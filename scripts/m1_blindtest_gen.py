# -*- coding: utf-8 -*-
"""对照盘实验 v2（10 号文档第五节）：50 组盲测材料生成。

每组 = 1 个查询片段（隐藏后续）+ 4 个候选（不标注通道）：
  C1 形状最近邻（去趋势+归一化）      —— v_shape 暴力检索 top-1
  C2 形状最近邻（不去趋势，带背景）    —— v_raw 暴力检索 top-1
  C3 同处境随机（同期、BTC 趋势/波动率相似、形状不同）
  C4 完全随机
你只做一件事：每组挑出"最像"的 1 个（可附一句话理由）。
做完 50 组 -> 导出 answers.json -> m1_blindtest_score.py 统计四通道被选频率。

输出：results/m1_blindtest/  （blindtest.html + key.json + 3 张样例 PNG）
用法：python m1_blindtest_gen.py [--groups 50] [--seed 42]
"""
import argparse
import base64
import io
import json
import os
import time

import numpy as np
import pandas as pd

import m1_lib as L

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402

UP, DOWN = "#26a69a", "#ef5350"
OUT = os.path.join(L.DIR_RESULTS, "m1_blindtest")
CHANNELS = ["C1_shape_detrend", "C2_shape_raw", "C3_context_random", "C4_random"]


# ---------- 数据访问（带缓存） ----------
_ot_cache = {}


def sym_ot(sym):
    if sym not in _ot_cache:
        df = L.load_15m(sym, columns=["open_time"])
        _ot_cache[sym] = df["open_time"].to_numpy(np.int64)
    return _ot_cache[sym]


def load_window(row):
    """meta 行 -> (窗口 df100, 未来 df24, symbol)"""
    sym = SYMS[int(META["sym"][row])]
    s_ot, e_ot = int(META["start"][row]), int(META["end"][row])
    df = L.load_15m(sym, columns=["open_time", "open", "high", "low", "close", "volume"])
    ot = df["open_time"].to_numpy(np.int64)
    p = int(np.searchsorted(ot, s_ot, side="left"))
    w = df.iloc[p:p + L.W].reset_index(drop=True)
    q = int(np.searchsorted(ot, e_ot, side="right"))
    fut = df.iloc[q:q + L.FUTURE_BARS].reset_index(drop=True)
    return w, fut, sym


def btc_context(s_ot, e_ot):
    """查询/候选片段同时段的 BTC 处境：(net_atr, atr_pct)。"""
    ot = sym_ot("BTCUSDT")
    btc = sym_btc()
    i_s = int(np.searchsorted(ot, s_ot, side="left"))
    i_e = int(np.searchsorted(ot, e_ot, side="right")) - 1
    if i_s < 0 or i_e <= i_s or i_e >= len(ot):
        return None
    atr = float(btc["cumtr"][i_e + 1] - btc["cumtr"][i_s]) / (i_e - i_s + 1)
    if atr <= 0:
        return None
    net = (float(btc["close"][i_e]) - float(btc["close"][i_s])) / atr
    return net, atr / float(btc["close"][i_s])


_btc = None


def sym_btc():
    global _btc
    if _btc is None:
        df = L.load_15m("BTCUSDT", columns=["open_time", "close", "high", "low"])
        tr = L.true_range(df["high"].to_numpy(float), df["low"].to_numpy(float),
                          df["close"].to_numpy(float))
        _btc = dict(close=df["close"].to_numpy(float),
                    cumtr=np.r_[0.0, np.cumsum(tr)])
    return _btc


# ---------- 候选挑选 ----------
def pick_nn(mat_name, q_row, banned, k=1):
    q = np.asarray(np.load(os.path.join(L.DIR_INDEX, mat_name + ".npy"),
                           mmap_mode="r")[q_row], dtype=np.float32)
    ex = np.zeros(len(META["sym"]), dtype=bool)
    ex[list(banned)] = True
    idx, dist = L.topk_bruteforce(os.path.join(L.DIR_INDEX, mat_name + ".npy"),
                                  None, q, 200, exclude=ex)
    for i in idx:
        if i >= 0 and int(i) not in banned:
            return int(i)
    raise RuntimeError("no candidate left")


def pick_context_random(q_row, banned, ctx_q, rng, pool_rows):
    """同处境随机：尽量同时段(渐进放宽) + BTC 趋势/波动率相似 + 形状距离远。
    返回 (row, degraded: bool)；找不到返回 (None, True)。"""
    qv = np.asarray(np.load(os.path.join(L.DIR_INDEX, "v_shape.npy"),
                            mmap_mode="r")[q_row], dtype=np.float32)
    q_start = int(META["start"][q_row])
    for max_days, need_ctx in ((45, True), (120, True), (365, False), (10**9, False)):
        cands = []
        for i in pool_rows:
            if int(i) in banned:
                continue
            s_i = int(META["start"][i])
            if abs(s_i - q_start) > max_days * 86_400_000:
                continue
            if need_ctx:
                c = btc_context(s_i, int(META["end"][i]))
                if c is None or ctx_q is None:
                    continue
                if abs(c[0] - ctx_q[0]) > 0.6:
                    continue
                if not (0.6 < c[1] / max(ctx_q[1], 1e-12) < 1.6):
                    continue
            cands.append(int(i))
        if len(cands) >= 5:
            break
    if len(cands) < 5:
        return None, True
    mat = np.load(os.path.join(L.DIR_INDEX, "v_shape.npy"), mmap_mode="r")
    dd = np.asarray(mat[np.array(cands)], dtype=np.float32) - qv
    dd = (dd * dd).sum(1)
    far = np.array(cands)[dd > np.percentile(dd, 60)]  # 形状不同
    return int(rng.choice(far)), False


def overlap_ban(row, banned_frac=0.5):
    """与 row 同币种且时间重叠 >50% 的全部窗口 -> bool 掩码。"""
    si = int(META["sym"][row])
    s_ot, e_ot = int(META["start"][row]), int(META["end"][row])
    same = np.asarray(META["sym"]) == si
    inter = np.minimum(np.asarray(META["end"]), e_ot) - np.maximum(np.asarray(META["start"]), s_ot)
    return same & (inter > banned_frac * (e_ot - s_ot))


# ---------- 绘图 ----------
def chart_png(w, title="", meta_line="", highlight=False):
    fig = plt.figure(figsize=(3.9, 2.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1])
    x = np.arange(len(w))
    o = w["open"].to_numpy(float)
    c = w["close"].to_numpy(float)
    h = w["high"].to_numpy(float)
    l = w["low"].to_numpy(float)
    col = np.where(c >= o, UP, DOWN)
    ax.vlines(x, l, h, color=col, lw=0.6)
    body_lo = np.minimum(o, c)
    body_h = np.maximum(np.abs(c - o), (h.max() - l.min()) * 2e-3)
    ax.bar(x, body_h, bottom=body_lo, width=0.65, color=col, linewidth=0)
    ax.set_ylim(l.min() - (h.max() - l.min()) * 0.04, h.max() + (h.max() - l.min()) * 0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    if highlight:
        for sp in ax.spines.values():
            sp.set_color("#f39c12")
            sp.set_linewidth(2.2)
    axv.bar(x, w["volume"].to_numpy(float), width=0.65,
            color=col, alpha=0.6, linewidth=0)
    axv.set_yscale("log")
    axv.set_xticks([])
    axv.set_yticks([])
    for sp in axv.spines.values():
        sp.set_color("#cccccc")
    if title:
        ax.set_title(title, fontsize=9)
    if meta_line:
        axv.set_xlabel(meta_line, fontsize=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=88, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fmt_ctx(ctx):
    if ctx is None:
        return "BTC: n/a"
    return f"BTC {ctx[0]:+.1f} ATR/25h · vol {ctx[1]*100:.2f}%/bar"


# ---------- 主流程 ----------
def main():
    global META, SYMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    meta = L.load_meta()
    META, SYMS = meta, meta["symbols"]
    N = len(META["sym"])
    os.makedirs(OUT, exist_ok=True)

    # 有效查询行：未来 24 根存在
    print("筛选有未来的查询行 ...", flush=True)
    t0 = time.time()
    ok_rows = []
    for row in rng.choice(N, size=3000, replace=False):
        sym = SYMS[int(META["sym"][row])]
        ot = sym_ot(sym)
        if int(META["end"][row]) + L.FUTURE_BARS * 900_000 <= ot[-1]:
            ok_rows.append(int(row))
    print(f"  {len(ok_rows)} 可用查询行 ({time.time()-t0:.0f}s)", flush=True)

    # 查询两两不同（同币种时间距离 > 300 根）
    queries, seen = [], []
    for row in ok_rows:
        sym = SYMS[int(META["sym"][row])]
        s = int(META["start"][row])
        if all(sy != sym or abs(s - s2) > 300 * 900_000 for sy, s2 in seen):
            queries.append(row)
            seen.append((sym, s))
        if len(queries) >= args.groups:
            break
    assert len(queries) == args.groups, f"只凑齐 {len(queries)} 组查询"

    # C3 池
    pool_rows = rng.choice(N, size=40_000, replace=False)

    groups, charts = [], {}
    saved_samples = 0
    degraded_groups = []
    qi = 0
    while len(groups) < args.groups and qi < len(queries):
        q_row = queries[qi]
        qi += 1
        try:
            ban = overlap_ban(q_row) | (np.arange(N) == q_row)
            banned = set(np.where(ban)[0].tolist())
            ctx_q = btc_context(int(META["start"][q_row]), int(META["end"][q_row]))

            picks = {}
            picks["C1_shape_detrend"] = pick_nn("v_shape", q_row, banned)
            banned.add(picks["C1_shape_detrend"])
            picks["C2_shape_raw"] = pick_nn("v_raw", q_row, banned)
            banned.add(picks["C2_shape_raw"])
            p3, deg = pick_context_random(q_row, banned, ctx_q, rng, pool_rows)
            if p3 is None:
                # 彻底降级：形状距离远的随机窗口（不计入处境通道的严格意义）
                mat = np.load(os.path.join(L.DIR_INDEX, "v_shape.npy"), mmap_mode="r")
                qv = np.asarray(mat[q_row], dtype=np.float32)
                alt = [int(i) for i in rng.choice(N, 500, replace=False)
                       if int(i) not in banned]
                dd = np.asarray(mat[np.array(alt)], dtype=np.float32) - qv
                dd = (dd * dd).sum(1)
                p3 = alt[int(np.argmax(dd))]
                deg = True
            if deg:
                degraded_groups.append(len(groups) + 1)
            picks["C3_context_random"] = p3
            banned.add(p3)
            p4 = int(rng.choice(np.where(~ban)[0]))
            while p4 in banned:
                p4 = int(rng.choice(np.where(~ban)[0]))
            picks["C4_random"] = p4

            order = list(CHANNELS)
            rng.shuffle(order)  # 展示顺序打乱
            qw, qfut, qsym = load_window(q_row)
            entry = dict(group=len(groups) + 1, query_row=int(q_row), query_sym=qsym,
                         query_start=int(META["start"][q_row]), order=order,
                         degraded_c3=deg,
                         candidates={ch: dict(row=int(r), sym=SYMS[int(META["sym"][r])],
                                              start=int(META["start"][r]))
                                     for ch, r in picks.items()})
            groups.append(entry)

            qmeta = f"{qsym} · " + fmt_ctx(ctx_q)
            charts[f"q{entry['group']}"] = chart_png(qw, "当前片段 QUERY", qmeta,
                                                     highlight=True)
            for ch, r in picks.items():
                cw, cfut, csym = load_window(r)
                c = btc_context(int(META["start"][r]), int(META["end"][r]))
                letter = "ABCD"[order.index(ch)]
                charts[f"{entry['group']}_{letter}"] = chart_png(
                    cw, f"候选 {letter}", f"{csym} · " + fmt_ctx(c))
                entry["candidates"][ch]["letter"] = letter
            if saved_samples < 2:
                os.makedirs(os.path.join(OUT, "sample"), exist_ok=True)
                for name, b64 in list(charts.items())[-5:]:
                    with open(os.path.join(OUT, "sample", f"g{entry['group']}_{name}.png"),
                              "wb") as f:
                        f.write(base64.b64decode(b64))
                saved_samples += 1
            if entry["group"] % 10 == 0:
                print(f"  group {entry['group']}/{args.groups}", flush=True)
        except Exception as e:
            print(f"  [skip] query_row={q_row} failed: {type(e).__name__}: {e}",
                  flush=True)
            if len(groups) + (len(queries) - qi) < args.groups:
                raise
    if degraded_groups:
        print(f"  [warn] degraded C3 groups: {degraded_groups}", flush=True)

    L.write_json(os.path.join(OUT, "key.json"),
                 dict(seed=args.seed, generated=time.strftime("%Y-%m-%d %H:%M"),
                      groups=groups))

    imgs_js = json.dumps(charts)
    # HTML 版只带字母映射，不带通道名/行号（防从源码偷看答案；真答案在 key.json）
    html_groups = [dict(group=g["group"], order=g["order"],
                        cand={"ABCD"[g["order"].index(ch)]:
                              dict(sym=v["sym"],
                                   start=pd.to_datetime(v["start"], unit="ms", utc=True)
                                        .strftime("%Y-%m-%d %H:%M"))
                              for ch, v in g["candidates"].items()})
                   for g in groups]
    groups_js = json.dumps(html_groups)
    html = HTML.replace("__GROUPS__", groups_js).replace("__IMGS__", imgs_js)
    html_path = os.path.join(OUT, f"blindtest_{time.strftime('%Y%m%d')}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    size = os.path.getsize(html_path) / 1e6
    print(f"done: {html_path} ({size:.1f} MB), key.json, sample PNG x{saved_samples*5}")


HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>M1 对照盘实验 v2</title>
<style>
body{font-family:"Microsoft YaHei",sans-serif;max-width:1080px;margin:24px auto;padding:0 12px;background:#111;color:#ddd}
.g{background:#1b1b1b;border-radius:10px;padding:14px;margin:18px 0}
.q{display:block;margin:0 auto;max-width:520px}
.cands{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
.cand{width:24%;min-width:230px;background:#222;border:2px solid #333;border-radius:8px;padding:6px;text-align:center;cursor:pointer}
.cand img{width:100%}
.cand.sel{border-color:#f39c12;background:#2a2416}
button{background:#f39c12;border:0;padding:10px 22px;border-radius:6px;font-size:15px;cursor:pointer}
textarea{width:100%;height:70px;background:#222;color:#ddd;border:1px solid #444}
h3{margin:4px 0 10px}
#bar{position:sticky;top:0;background:#111;padding:8px 0;border-bottom:1px solid #333;z-index:9}
</style></head><body>
<div id="bar">进度 <span id="prog">0</span>/50 · 每组挑一个"最像"的候选（点击卡片） · 做完点底部导出</div>
<h2>对照盘实验 v2 · 盲测</h2>
<p>每组给 1 个「当前片段」+ 4 个历史候选（<b>后续走势已隐藏</b>，通道不标注）。凭直觉选出你觉得"最像当前片段"的那一个。
一句话理由可选填（你用的线索：量能？位置？节奏？）。预计 30~40 分钟，中途可停，进度会保存在页面上（刷新不丢，但换浏览器/清缓存会丢，建议一次性做完就导出）。</p>
<div id="root"></div>
<button onclick="export_answers()">导出 answers.json（保存到 results/m1_blindtest/）</button>
<p>若下载不可用，复制下面文本框内容另存为 answers.json：</p>
<textarea id="fallback"></textarea>
<script>
const GROUPS=__GROUPS__, IMGS=__IMGS__;
const picks={}, reasons={};
document.addEventListener("DOMContentLoaded",()=>{
  const root=document.getElementById("root");
  GROUPS.forEach(g=>{
    const div=document.createElement("div");div.className="g";div.id="g"+g.group;
    div.innerHTML=`<h3>第 ${g.group} 组</h3><img class="q" src="data:image/png;base64,${IMGS["q"+g.group]}">
      <div class="cands">`+
      g.order.map((ch,i)=>{
        const L="ABCD"[i];
        return `<div class="cand" id="c${g.group}${L}" onclick="pick(${g.group},'${L}')">
          <img src="data:image/png;base64,${IMGS[g.group+"_"+L]}"></div>`;
      }).join("")+`</div>
      <p>理由（可选）：<input id="r${g.group}" style="width:80%;background:#222;color:#ddd;border:1px solid #444"
        oninput="reasons[${g.group}]=this.value"></p>`;
    root.appendChild(div);
  });
});
function pick(gr,L){
  picks[gr]=L;
  document.querySelectorAll(`#g${gr} .cand`).forEach(e=>e.classList.remove("sel"));
  document.getElementById("c"+gr+L).classList.add("sel");
  document.getElementById("prog").textContent=Object.keys(picks).length;
}
function export_answers(){
  const out={exported:new Date().toISOString(),answers:{}};
  GROUPS.forEach(g=>{ if(picks[g.group]) out.answers[g.group]={pick:picks[g.group],reason:reasons[g.group]||""}; });
  const txt=JSON.stringify(out,null,1);
  document.getElementById("fallback").value=txt;
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([txt],{type:"application/json"}));
  a.download="answers.json";a.click();
  alert(`已选 ${Object.keys(picks).length}/50 组。文件已下载，请放到 results/m1_blindtest/answers.json`);
}
</script></body></html>"""


if __name__ == "__main__":
    main()
