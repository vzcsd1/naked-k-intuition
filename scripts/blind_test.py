#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · 对照盘实验 v2（造靶子）

目的：把"哪种像说不清"变成可测量的问题。
  50 组盲测。每组给 1 个「当前片段」+ 4 个候选（**不告诉你是哪来的**），
  你只做一个动作：挑出最像的 1 个。
  统计四者被选频率 → 直接给出通道权重 → 也直接回答"要不要去趋势"。

四个候选（出处保密）：
  ① shape  去趋势最近邻（尺度无关的形态 + 成交量）
  ② raw    保留趋势最近邻（带着涨跌背景的形态 + 成交量）
  ③ regime 同处境随机（与当前片段同处 ±45 天内的市场环境，但形状随机）
  ④ random 完全随机

纪律（见 08 / 10）：
  · 盲测期间**不显示未来走势** —— 否则你的判断会被结果污染
  · 每个候选的**展示位置随机打乱** —— 否则你会被位置暗示
  · **做完这个实验之前不调任何相似度权重**

用法：python blind_test.py --n 50 --seed 42
输出：results/blind_test.html（自包含，双击即可用）
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import retrieve as R

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IDX = ROOT / "data" / "index"
RESULTS = ROOT / "results"
# 默认用**干净币池**索引（当时成交额前 20 名）。要用旧全池索引：BK_INDEX=15m_100
INDEX_NAME = os.environ.get("BK_INDEX", "15m_100_top20")

REGIME_DAYS = 45          # "同处境"的时间半窗
MIN_AMP = 2.0             # 查询窗口的最小波幅（%），太小的没有形态可言
MAX_AMP = 60.0


def pick_queries(ix: R.Index, n: int, seed: int) -> list[int]:
    m = ix.meta
    yrs = pd.to_datetime(m["t0"], unit="ms", utc=True).dt.year
    ok = (m["fwd_ret"].notna() & (m["amp_pct"] >= MIN_AMP) & (m["amp_pct"] <= MAX_AMP)
          & (m["t0"] > int(pd.Timestamp("2018-01-01", tz="UTC").timestamp() * 1000)))
    cand = np.flatnonzero(ok.to_numpy())
    rng = np.random.default_rng(seed)
    # 按年分层，保证不集中在某一段行情
    per_year = max(1, n // max(1, yrs[cand].nunique()))
    out = []
    for y in sorted(yrs[cand].unique()):
        pool = cand[(yrs[cand] == y).to_numpy()]
        k = min(per_year, len(pool))
        out.extend(rng.choice(pool, size=k, replace=False).tolist())
    rng.shuffle(out)
    return out[:n]


def _ohlc_for(ix, idx_list):
    """按币种分组，每个 parquet 只读一次，抽出需要的窗口。"""
    m = ix.meta
    need = {}
    for i in idx_list:
        s = str(m["symbol"].iloc[i])
        need.setdefault(s, []).append(i)
    out = {}
    for s, ids in need.items():
        p = RAW / "15m" / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["open_time", "open", "high", "low", "close", "quote_volume"])
        ot = d["open_time"].to_numpy()
        o = d["open"].to_numpy(); h = d["high"].to_numpy()
        l = d["low"].to_numpy(); c = d["close"].to_numpy()
        qv = d["quote_volume"].to_numpy()
        pos = {int(t): k for k, t in enumerate(ot)}
        for i in ids:
            t0 = int(m["t0"].iloc[i]); t1 = int(m["t1"].iloc[i])
            a = pos.get(t0)
            if a is None:
                continue
            b = pos.get(t1, a + 99) + 1
            bars = [[round(float(o[j]), 10), round(float(h[j]), 10), round(float(l[j]), 10),
                     round(float(c[j]), 10), round(float(qv[j]), 2)] for j in range(a, min(b, a + 100))]
            if len(bars) >= 50:
                out[i] = bars
    return out


def build(n_groups: int, seed: int):
    ix = R.Index(IDX / INDEX_NAME)
    m = ix.meta
    queries = pick_queries(ix, n_groups, seed)
    t0_all = m["t0"].to_numpy()
    sym_all = m["symbol"].to_numpy().astype(str)
    rng = np.random.default_rng(seed + 1)

    groups, used = [], set()
    for qi, i in enumerate(queries, 1):
        q = np.asarray(ix.vecs("shape")[i], dtype=np.float32)
        qr = np.asarray(ix.vecs("raw")[i], dtype=np.float32)
        pool = 600
        ci, _ = ix.search(q, "shape", topk=pool)
        keep = ~ix.exclude_mask(i, ci)
        ci = ci[keep]
        ri, _ = ix.search(qr, "raw", topk=pool)
        rkeep = ~ix.exclude_mask(i, ri)
        ri = ri[rkeep]

        picks = {}
        # ① shape 最近邻
        for j in ci:
            if j not in used:
                picks["shape"] = int(j); break
        # ② raw 最近邻（给出趋势背景更像的）
        for j in ri:
            if j not in used and picks.get("shape") != int(j):
                picks["raw"] = int(j); break
        # ③ 同处境随机：t0 落在查询 ±45 天内，形状随机
        win = REGIME_DAYS * 86400_000
        t = int(t0_all[i])
        mask = (np.abs(t0_all - t) <= win)
        mask[i] = False
        cand3 = np.flatnonzero(mask)
        rng.shuffle(cand3)
        for j in cand3[:2000]:
            if int(j) not in used and int(j) not in picks.values():
                picks["regime"] = int(j); break
        # ④ 完全随机
        for _ in range(5000):
            j = int(rng.integers(0, len(m)))
            if j != i and j not in used and j not in picks.values():
                picks["random"] = j; break
        if len(picks) < 4:
            continue
        used.update(picks.values())
        used.add(i)
        order = list(picks.keys())
        rng.shuffle(order)                      # 展示位置随机打乱
        groups.append({"qi": i, "cands": [{"src": s, "idx": picks[s]} for s in order]})
        if qi % 10 == 0:
            print(f"  已构造 {len(groups)} 组")

    all_ids = [g["qi"] for g in groups] + [c["idx"] for g in groups for c in g["cands"]]
    print(f"  读取 {len(set(all_ids))} 个窗口的 K 线数据…")
    bars = _ohlc_for(ix, all_ids)

    data = []
    for g in groups:
        i = g["qi"]
        if i not in bars:
            continue
        cs = []
        for c in g["cands"]:
            if c["idx"] in bars:
                cs.append({"src": c["src"], "sym": str(m["symbol"].iloc[c["idx"]]),
                           "t0": int(m["t0"].iloc[c["idx"]]), "bars": bars[c["idx"]]})
        if len(cs) < 4:
            continue
        data.append({
            "q": {"sym": str(m["symbol"].iloc[i]), "t0": int(m["t0"].iloc[i]),
                  "amp": round(float(m["amp_pct"].iloc[i]), 1), "bars": bars[i]},
            "cands": cs,
        })
    return data


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>裸K盘感 · 对照盘实验 v2</title>
<style>
:root{--bg:#fff;--fg:#2C2C2A;--mut:#5F5E5A;--line:#D3D1C7;--up:#D4534E;--dn:#1D9E75;--acc:#185FA5}
*{box-sizing:border-box}
body{margin:0;background:#F1EFE8;color:var(--fg);font:14px/1.6 system-ui,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px}
h1{font-size:17px;font-weight:500;margin:0 0 4px}
.sub{color:var(--mut);font-size:13px;margin-bottom:14px}
.bar{height:6px;background:#E6E4DC;border-radius:3px;overflow:hidden;margin-bottom:16px}
.bar>i{display:block;height:100%;background:var(--acc);width:0;transition:width .25s}
.card{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
.qhead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
.tag{font-size:12px;color:var(--mut)}
.qtag{font-size:12px;color:#0C447C;background:#E6F1FB;border:1px solid #B5D4F4;border-radius:20px;padding:1px 9px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.cand{background:#FBFAF7;border:1px solid var(--line);border-radius:9px;padding:8px;cursor:pointer;transition:border-color .15s,background .15s;position:relative}
.cand:hover{border-color:#85B7EB;background:#F7FBFF}
.cand.sel{border-color:var(--acc);background:#E6F1FB;box-shadow:inset 0 0 0 1px var(--acc)}
.cand .lab{position:absolute;top:6px;left:9px;font-size:12px;font-weight:500;color:var(--mut)}
.cand.sel .lab{color:#0C447C}
.cand .meta{position:absolute;top:6px;right:9px;font-size:11px;color:#888780}
svg{display:block;width:100%}
.act{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
input[type=text]{flex:1;min-width:260px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;font:13px inherit;background:#fff}
button{padding:8px 18px;border:0;border-radius:8px;background:var(--acc);color:#fff;font:14px inherit;cursor:pointer}
button:disabled{background:#B4B2A9;cursor:not-allowed}
button.ghost{background:#fff;color:var(--fg);border:1px solid var(--line)}
.hint{font-size:12px;color:var(--mut)}
#done{display:none}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left}
th{background:#E6F1FB;font-weight:500}
.big{font-size:15px;font-weight:500}
kbd{background:#E6E4DC;border-radius:4px;padding:1px 6px;font-size:12px}
</style></head><body><div class="wrap">
<h1>对照盘实验 v2 —— 你眼里的「像」到底是哪一种</h1>
<div class="sub" id="sub"></div>
<div class="bar"><i id="prog"></i></div>
<div id="main"></div>
<div id="done">
  <div class="card">
    <div class="big">🎉 全部完成</div>
    <p class="hint">下面是四个通道被选中的次数。请把这段结果发回给我，或点下方按钮导出。</p>
    <div id="sum"></div>
    <div class="act">
      <button onclick="dl()">导出结果 JSON</button>
      <button class="ghost" onclick="dlTxt()">复制为文本</button>
    </div>
  </div>
</div>
</div>
<script>
const DATA = __DATA__;
const KEY = 'lk_blind_v2';
let st = JSON.parse(localStorage.getItem(KEY) || 'null') || {i:0, picks:[], whys:[]};
const LAB = ['①','②','③','④'];
const NAME = {shape:'去趋势最近邻(形状)', raw:'保留趋势最近邻', regime:'同处境随机', random:'完全随机'};

function render(){
  const tot = DATA.length;
  if (st.i >= tot) { return finish(); }
  document.getElementById('done').style.display='none';
  document.getElementById('prog').style.width = (100*st.i/tot)+'%';
  document.getElementById('sub').textContent = `第 ${st.i+1} / ${tot} 组 —— 在下排四个候选里挑出「和上面最像」的那个`;
  const g = DATA[st.i];
  document.getElementById('main').innerHTML = `
    <div class="card">
      <div class="qhead"><span class="qtag">当前片段 · 你要匹配的就是它</span>
        <span class="tag">波幅 ${g.q.amp}% ｜ 100 根 15 分钟 K 线（约 25 小时）</span></div>
      <div id="qchart"></div>
    </div>
    <div class="card">
      <div class="qhead"><span class="tag">下面四个里，哪一个最像上面？</span>
        <span class="tag">出处已隐藏 · 位置已打乱 · 也不显示后续走势</span></div>
      <div class="grid">${g.cands.map((c,k)=>`
        <div class="cand" id="c${k}" onclick="pick(${k})">
          <span class="lab">${LAB[k]}</span><span class="meta">波幅 ${((Math.max(...c.bars.map(b=>b[1]))/Math.min(...c.bars.map(b=>b[2]))-1)*100).toFixed(1)}%</span>
          <div id="cc${k}" style="margin-top:18px"></div>
        </div>`).join('')}
      </div>
      <div class="act">
        <input type="text" id="why" placeholder="为什么选它？（一句话，可留空）" value="">
        <button id="next" onclick="next()" disabled>下一组</button>
        <button class="ghost" onclick="skip()">跳过这组</button>
      </div>
      <div class="hint">快捷键：<kbd>1</kbd><kbd>2</kbd><kbd>3</kbd><kbd>4</kbd> 选择，<kbd>Enter</kbd> 下一组</div>
    </div>`;
  draw(document.getElementById('qchart'), g.q.bars, 1100, 230);
  g.cands.forEach((c,k)=>draw(document.getElementById('cc'+k), c.bars, 520, 190));
  if (st.picks.length > st.i && st.picks[st.i] != null) {
    const el = document.getElementById('c'+st.picks[st.i]); if (el) el.classList.add('sel');
    document.getElementById('next').disabled = false;
  }
}

function draw(host, bars, w, h){
  const pad=6, vh=h*0.70, gap=8, volTop=vh+gap, volH=h-volTop-2;
  const hi=Math.max(...bars.map(b=>b[1])), lo=Math.min(...bars.map(b=>b[2]));
  const span=(hi-lo)||1, n=bars.length;
  const bw=Math.max(1.2,(w-2*pad)/n*0.62);
  const x=i=>pad+(i+0.5)*(w-2*pad)/n;
  const y=p=>pad+(hi-p)/span*(vh-2*pad);
  const vmax=Math.max(...bars.map(b=>b[4]))||1;
  let s=`<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="height:${h}px">`;
  s+=`<rect x="0" y="${volTop-1}" width="${w}" height="1" fill="#D3D1C7"/>`;
  bars.forEach((b,i)=>{
    const [o,hh,ll,c,v]=b;
    const up=c>=o, col=up?'#D4534E':'#1D9E75';
    const cx=x(i);
    s+=`<line x1="${cx}" y1="${y(hh)}" x2="${cx}" y2="${y(ll)}" stroke="${col}" stroke-width="1"/>`;
    const y1=y(Math.max(o,c)), y2=y(Math.min(o,c));
    s+=`<rect x="${cx-bw/2}" y="${y1}" width="${bw}" height="${Math.max(1,y2-y1)}" fill="${col}"/>`;
    const vh2=Math.max(0.5, v/vmax*(volH-2));
    s+=`<rect x="${cx-bw/2}" y="${volTop+volH-vh2}" width="${bw}" height="${vh2}" fill="${col}" opacity="0.55"/>`;
  });
  s+='</svg>';
  host.innerHTML=s;
}

function pick(k){
  document.querySelectorAll('.cand').forEach(e=>e.classList.remove('sel'));
  document.getElementById('c'+k).classList.add('sel');
  st.cur = k; document.getElementById('next').disabled=false;
}
function next(){
  if (st.cur==null) return;
  st.picks[st.i]=st.cur;
  st.whys[st.i]=document.getElementById('why')?.value||'';
  st.i++; delete st.cur; localStorage.setItem(KEY, JSON.stringify(st));
  render(); window.scrollTo(0,0);
}
function skip(){ st.picks[st.i]=null; st.whys[st.i]=''; st.i++; delete st.cur;
  localStorage.setItem(KEY, JSON.stringify(st)); render(); window.scrollTo(0,0); }

function finish(){
  document.getElementById('main').innerHTML='';
  document.getElementById('sub').textContent='实验完成，结果如下';
  document.getElementById('prog').style.width='100%';
  document.getElementById('done').style.display='block';
  const cnt={shape:0,raw:0,regime:0,random:0}, tot={shape:0,raw:0,regime:0,random:0};
  DATA.forEach((g,i)=>{ g.cands.forEach(c=>tot[c.src]++);
    const p=st.picks[i]; if(p!=null) cnt[g.cands[p].src]++; });
  const done=st.picks.filter(x=>x!=null).length;
  let rows='';
  Object.keys(NAME).forEach(k=>{
    const c=cnt[k], t=tot[k], pct=t? (100*c/t):0;
    rows+=`<tr><td>${NAME[k]}</td><td>${c}</td><td>${t}</td><td><b>${pct.toFixed(1)}%</b></td></tr>`;});
  document.getElementById('sum').innerHTML =
    `<p>有效作答 <b>${done}</b> / ${DATA.length} 组。四通道完全随机时应各约 25%。</p>
     <table><tr><th>通道</th><th>被选次数</th><th>出现次数</th><th>被选率</th></tr>${rows}</table>`;
}
function result(){ return {n: DATA.length, picks: st.picks, whys: st.whys,
  srcs: DATA.map(g=>g.cands.map(c=>c.src)), ver:2 }; }
function dl(){
  const b=new Blob([JSON.stringify(result(),null,1)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(b);
  a.download='blind_test_result.json'; a.click();
}
function dlTxt(){
  const r=result(); const lines=[`对照盘实验 v2 结果（${r.n} 组）`];
  DATA.forEach((g,i)=>{ const p=r.picks[i];
    lines.push(`第${i+1}组 选=${p==null?'跳过':LAB[p]+' '+g.cands[p].src}  理由=${r.whys[i]||''}`); });
  navigator.clipboard.writeText(lines.join('\n')).then(()=>alert('已复制到剪贴板'));
}
document.addEventListener('keydown',e=>{
  if(['1','2','3','4'].includes(e.key)) pick(+e.key-1);
  if(e.key==='Enter') next();
});
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    data = build(args.n, args.seed)
    if not data:
        raise SystemExit("没有构造出任何组")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "blind_test.html"
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    out.write_text(html, encoding="utf-8")
    print(f"[blind] {len(data)} 组 -> {out}")
    print(f"        HTML 大小 {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
