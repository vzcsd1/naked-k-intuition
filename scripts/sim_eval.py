#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · M1 相似度定义的「客观靶子」

## 为什么有这个脚本

原计划（10 第五节）用 50 组**人眼盲测**决定"哪一种像"，但用户反馈肉眼判断太困难。
日志里已记下这个反馈**动摇了实验前提**：
> 如果用户的肉眼判断本身不稳定，那么四通道被选率的统计就是在测噪声，
> 「人眼相似度」根本不是正确的靶子。

替代方案 —— 08 第六节 6.3「决策等价性」的**最小可测版本**：
> 两个片段"像" ⟺ 「如果在 A 上做某操作是对的，那么在 B 上做同样操作也是对的」

它不需要任何人内省，可用历史数据**自动标注**。

## 怎么测（严格守住 08 的红线）

红线：**检索时只准用当前可见信息，未来只作标签。**

对每个查询窗口 i：
  1. 只用**当前可见**的向量去检索 top-K 历史邻居；
  2. 邻居与查询在时间上**完全不重叠**（连各自的未来区间都不重叠）→ 排除机械相关；
  3. 再看「邻居的后续」能不能预测「查询自己的后续」。

## 三个必须同时看的东西

| 指标 | 读法 |
|---|---|
| IC(超额收益) | 邻居后续 → 查询后续的秩相关。**0 附近 = 形态没有预测力** |
| 邻居未来离散度 / 随机 | **最关键的一个**。若 ≈1，说明「检索出的相似段，后续分散程度和随机抽 40 段一样」→ 形态不约束结果 |
| 正对照·同币7天内 | **管道体检**。同币种时间相近的窗口，未来必然相关；若这行也是 0，说明脚本坏了不是结论 |

## 参与比较的定义（同一个库里取子空间，不重新建库、不调任何权重）

| 名称 | 内容 | 它在回答 |
|---|---|---|
| shape64 | 去趋势形态32 + 量32 | 当前主通道 |
| shape32 | 仅去趋势形态32 | 去掉量能会变好还是变差 |
| vol32   | 仅量32 | 量能单独有多大用 |
| raw64   | 保留趋势形态32 + 量32 | **要不要去趋势** |
| raw32   | 仅保留趋势形态32 | 趋势背景 vs 纯形态 |
| random  | 随机邻居 | 对照，IC 应≈0 |

⚠️ 这个靶子测的是「像 → 后续也像」，属于**可用性口径**，不是「人眼审美口径」。

用法：
  python sim_eval.py --n 2000 --topk 40
输出：
  results/sim_eval.csv · results/sim_eval_raw.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
IDX = ROOT / "data" / "index" / "15m_100"
RAW15 = ROOT / "data" / "raw" / "15m"
RESULTS = ROOT / "results"

BAR_MS = 15 * 60_000
HZ_BARS = 96                     # 与 build_index.py 的 hz 一致 = 24 小时
HZ_MS = HZ_BARS * BAR_MS
BAR_MS_H = 3_600_000
HORIZONS = {"1h": 4, "4h": 16, "12h": 48, "24h": 96}
MIN_AMP, MAX_AMP = 2.0, 60.0
T0_FLOOR = int(pd.Timestamp("2018-01-01", tz="UTC").timestamp() * 1000)
BENCH = "BTCUSDT"

VARIANTS: dict[str, tuple[str, tuple[int, int] | None]] = {
    "shape64": ("shape", None),
    "shape32": ("shape", (0, 32)),
    "vol32":   ("shape", (32, 64)),
    "raw64":   ("raw", None),
    "raw32":   ("raw", (0, 32)),
}
FILE_OF = {"shape": "vectors_shape.npy", "raw": "vectors_raw.npy"}

# ⭐ 「波幅加回去」的旋钮。
# 向量里每块的典型平方距离：形态/量能块 32 维 z-score 后约 2×32 = 64；
# 波幅块是 1 维标准化标量，随机两段之差平方约 2。
# 所以 w 让波幅贡献的比例约为 w*2 / (64 + w*2)：
#   w=4  → 11%    w=12 → 27%    w=32 → 50%    w=96 → 75%
AMP_WEIGHT = {"11%": 4.0, "27%": 12.0, "50%": 32.0, "75%": 96.0}
# (名称, 向量文件, 通道切片, 波幅权重)
SPECS: list[tuple[str, str, tuple[int, int] | None, float]] = [
    ("shape64", "shape", None, 0.0),
    ("shape64+波幅11%", "shape", None, AMP_WEIGHT["11%"]),
    ("shape64+波幅27%", "shape", None, AMP_WEIGHT["27%"]),
    ("shape64+波幅50%", "shape", None, AMP_WEIGHT["50%"]),
    ("shape64+波幅75%", "shape", None, AMP_WEIGHT["75%"]),
    ("shape32+波幅27%", "shape", (0, 32), AMP_WEIGHT["27%"]),
    ("raw64", "raw", None, 0.0),
    ("raw64+波幅27%", "raw", None, AMP_WEIGHT["27%"]),
]


# ---------------------------------------------------------------- 工具

def spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 30:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def pick_queries(m: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    yrs = pd.to_datetime(m["t0"], unit="ms", utc=True).dt.year
    ok = (m["fwd_ret"].notna() & m["amp_pct"].between(MIN_AMP, MAX_AMP)
          & (m["t0"] > T0_FLOOR)).to_numpy()
    cand = np.flatnonzero(ok)
    rng = np.random.default_rng(seed)
    uniq = sorted(pd.unique(yrs.iloc[cand]))
    per = max(1, n // max(1, len(uniq)))
    out: list[int] = []
    for y in uniq:
        pool = cand[(yrs.iloc[cand] == y).to_numpy()]
        out.extend(rng.choice(pool, size=min(per, len(pool)), replace=False).tolist())
    rng.shuffle(out)
    return np.array(out[:n], dtype=np.int64)


def q_spread(pred: np.ndarray, tgt: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(tgt)
    if ok.sum() < 50:
        return float("nan")
    s = pd.qcut(pd.Series(pred[ok]), 5, labels=False, duplicates="drop")
    g = pd.Series(tgt[ok]).groupby(s).mean()
    return float(g.iloc[-1] - g.iloc[0]) if len(g) >= 2 else float("nan")


# ---------------------------------------------------------------- 检索

class BatchSearch:
    """整个索引常驻显存，按查询批次做**精确** top-K。

    变体只是 64 维的子空间：查询向量未用通道补 0，
    于是 `blk @ q` 就等于子空间点积，不用真的切片；模长同理 `(blk**2) @ mask`。
    另可叠加「波幅」块：`D += w * (amp_z[候选] - amp_z[查询])²`。
    """

    def __init__(self, idx_dir, device: str = "cuda", amp_z: np.ndarray | None = None):
        self.dir = Path(idx_dir)
        self.device = device
        self.amp_z = amp_z
        self._amp_t = None
        self.cache: dict[str, torch.Tensor] = {}
        self.masks: dict[tuple[int, int], torch.Tensor] = {}

    def get_mask(self, sl: tuple[int, int] | None):
        key = (0, 64) if sl is None else (int(sl[0]), int(sl[1]))
        if key not in self.masks:
            mask = np.zeros(64, np.float32)
            mask[key[0]:key[1]] = 1.0
            self.masks[key] = torch.tensor(mask, device=self.device).view(64, 1)
        return self.masks[key]

    def amp_tensor(self):
        if self._amp_t is None:
            self._amp_t = torch.from_numpy(self.amp_z.astype(np.float32)).to(self.device)
        return self._amp_t

    def load(self, which: str):
        if which in self.cache:
            return self.cache[which]
        V = np.load(self.dir / FILE_OF[which], mmap_mode="r")
        n, d = V.shape
        g = torch.empty((n, d), dtype=torch.float32, device=self.device)
        step = 1_000_000
        for s in range(0, n, step):
            e = min(s + step, n)
            g[s:e] = torch.from_numpy(np.asarray(V[s:e], dtype=np.float32)).to(self.device)
        self.cache[which] = g
        return g

    def topk(self, which: str, sl: tuple[int, int] | None, Q: np.ndarray,
             pool: int, chunk: int, amp_w: float = 0.0, q_idx: np.ndarray | None = None):
        V = self.load(which)
        n = V.shape[0]
        qt = torch.from_numpy(np.asarray(Q, np.float32)).to(self.device)
        B = qt.shape[0]
        msk = self.get_mask(sl)
        qn = (qt.pow(2) @ msk).view(-1)                    # (B,)
        if amp_w > 0.0:
            A = self.amp_tensor()
            aq = A[torch.from_numpy(np.asarray(q_idx)).to(self.device)].view(-1, 1)  # (B,1)
        bv = torch.full((B, pool), float("inf"), device=self.device)
        bi = torch.zeros((B, pool), dtype=torch.long, device=self.device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            blk = V[s:e]
            v = ((blk.pow(2) @ msk).view(1, -1))
            D = qn[:, None] + v - 2.0 * (qt @ blk.t())
            if amp_w > 0.0:
                D = D + amp_w * (A[s:e].view(1, -1) - aq).pow(2)
            D.clamp_min_(0)
            k = min(pool, D.shape[1])
            cv, ci = torch.topk(D, k, dim=1, largest=False, sorted=True)
            ci = ci + s
            if s == 0:
                bv, bi = cv, ci
            else:
                av = torch.cat([bv, cv], 1)
                ai = torch.cat([bi, ci], 1)
                _, o = torch.topk(av, pool, dim=1, largest=False, sorted=True)
                bv = torch.gather(av, 1, o)
                bi = torch.gather(ai, 1, o)
            del D, cv, ci
        return bi.cpu().numpy()


# ---------------------------------------------------------------- 多周期后续

def fwd_multi(sym: np.ndarray, t1ms: np.ndarray, horizons: dict[str, int]):
    """按 (币种, 结束时刻) 取多个周期的后续收益与最高涨幅。返回 {h: (n,2)[ret, max]}。"""
    out = {h: np.full((len(sym), 2), np.nan) for h in horizons}
    order = np.argsort(sym, kind="stable")
    ss = np.asarray(sym, dtype=object)[order]
    cut = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    seg = np.r_[cut, len(ss)]
    for a, b in zip(seg[:-1], seg[1:]):
        s = str(ss[a])
        ks = order[a:b]
        p = RAW15 / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["open_time", "high", "close"])
        ot = d["open_time"].to_numpy()
        c = d["close"].to_numpy(np.float64)
        hi = d["high"].to_numpy(np.float64)
        n = len(c)
        pp = np.searchsorted(ot, t1ms[ks])                 # ot 已升序
        pp = np.clip(pp, 0, n - 1)
        good = ot[pp] == t1ms[ks]
        for k, i, g in zip(ks, pp, good):
            if not g or c[i] <= 0:
                continue
            for h, nb in horizons.items():
                if i + nb >= n:
                    continue
                out[h][k, 0] = c[i + nb] / c[i] - 1.0
                out[h][k, 1] = hi[i + 1:i + nb + 1].max() / c[i] - 1.0
    return out


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--pool", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--index", default="15m_100", help="data/index 下的索引目录名")
    ap.add_argument("--tag", default="", help="输出文件名后缀，避免覆盖")
    args = ap.parse_args()

    idx_dir = ROOT / "data" / "index" / args.index
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()
    print(f"[sim_eval] index={args.index} device={dev} n={args.n} topk={args.topk} pool={args.pool}")

    m = pd.read_parquet(idx_dir / "meta.parquet",
                        columns=["symbol", "t0", "t1", "amp_pct", "fwd_ret", "fwd_max"])
    n_all = len(m)
    codes = m["symbol"].cat.codes.to_numpy()
    cats = np.asarray(m["symbol"].cat.categories, dtype=object)
    t0a = m["t0"].to_numpy(); t1a = m["t1"].to_numpy()
    print(f"[sim_eval] 索引窗口 {n_all:,}")

    # ---- 「波幅」块：log 之后全库标准化（波幅分布右偏，必须先取 log）
    amp_a = m["amp_pct"].to_numpy(np.float64)
    la = np.log(np.maximum(amp_a, 1e-6))
    amp_z = (la - np.nanmean(la)) / np.nanstd(la)

    # 24h 横截面去均值（项目纪律：用超额不用绝对涨跌）
    m["exc"] = m["fwd_ret"] - m.groupby("t1", observed=True)["fwd_ret"].transform("mean")
    m["excmax"] = m["fwd_max"] - m.groupby("t1", observed=True)["fwd_max"].transform("mean")
    exc_a = m["exc"].to_numpy(np.float64)
    excmax_a = m["excmax"].to_numpy(np.float64)
    raw_a = m["fwd_ret"].to_numpy(np.float64)
    hit5_a = (m["fwd_max"].to_numpy(np.float64) >= 0.05).astype(np.float64)

    qi = pick_queries(m, args.n, args.seed)
    nq = len(qi)
    print(f"[sim_eval] 查询 {nq} 个（按年分层）")

    bs = BatchSearch(idx_dir, dev, amp_z=amp_z)
    res: dict[str, list] = {}
    for which in ("shape", "raw"):
        specs = [s for s in SPECS if s[1] == which]
        if not specs:
            continue
        for s0 in range(0, nq, args.batch):
            qsub = qi[s0:s0 + args.batch]
            Qall_m = bs.load(which)[torch.from_numpy(qsub).to(dev)].cpu().numpy()
            for name, f, sl, w in specs:
                Q = Qall_m.copy()
                if sl is not None:
                    Q[:, :sl[0]] = 0.0
                    Q[:, sl[1]:] = 0.0
                bi = bs.topk(which, sl, Q, args.pool, args.chunk, amp_w=w, q_idx=qsub)
                res.setdefault(name, []).append(bi)
        print(f"  [{which}] {len(specs)} 个变体完成  用时 {time.time() - t_start:.0f}s")

    rng = np.random.default_rng(args.seed + 99)
    # 正对照：同币种 ±7 天内的窗口（时间近 → 未来必然相关）。管道体检用。
    uniq_c, st = np.unique(codes, return_index=True)
    bnd = {int(u): (int(a), int(b)) for u, a, b in zip(uniq_c, st, np.append(st[1:], n_all))}
    NEAR = 7 * 86400_000
    ctrl = np.full((nq, args.pool), -1, np.int64)
    for k, i in enumerate(qi):
        a, b = bnd[int(codes[i])]
        cd = np.arange(a, b)
        cd = cd[np.abs(t1a[cd] - t1a[i]) <= NEAR]
        cd = cd[cd != i]
        if len(cd) > args.pool:
            cd = cd[rng.choice(len(cd), args.pool, replace=False)]
        ctrl[k, :len(cd)] = cd

    # ---- 时间硬隔离：候选必须「早于查询窗口起点」或「晚于查询未来终点」
    filt: dict[str, np.ndarray] = {}
    for v, parts in res.items():
        raw_idx = np.concatenate(parts, 0)
        f = np.full((nq, args.topk), -1, np.int64)
        for k in range(nq):
            c = raw_idx[k]
            keep = (t1a[c] <= t0a[qi[k]]) | (t0a[c] >= t1a[qi[k]] + HZ_MS)
            c = c[keep][: args.topk]
            f[k, :len(c)] = c
        filt[v] = f
    filt["random"] = rng.integers(0, n_all, size=(nq, args.topk))
    filt["正对照·同币7天内"] = ctrl[:, :args.topk]

    # 正对照②：同币种 ±12 小时，**故意不做时间隔离** —— 邻居的未来与查询大面积重叠，
    #            IC 必然接近 1。这是决定性的管道体检：这行不高说明脚本坏了，不是结论。
    ctrl2 = np.full((nq, args.topk), -1, np.int64)
    for k, i in enumerate(qi):
        a, b = bnd[int(codes[i])]
        cd = np.arange(a, b)
        cd = cd[np.abs(t1a[cd] - t1a[i]) <= 12 * BAR_MS_H]
        cd = cd[cd != i]
        if len(cd) > args.topk:
            cd = cd[rng.choice(len(cd), args.topk, replace=False)]
        ctrl2[k, :len(cd)] = cd
    filt["正对照·同币±12h(不隔离)"] = ctrl2

    # 诊断：**只用「波幅」这一个标量**做最近邻 —— 直接回答
    #       「64 维形态向量买到的东西，能不能超过一个标量？」
    ord_amp = np.argsort(amp_a, kind="stable")
    asort = amp_a[ord_amp]
    famp = np.full((nq, args.topk), -1, np.int64)
    for k, i in enumerate(qi):
        p = int(np.searchsorted(asort, amp_a[i]))
        lo, hi = max(0, p - 3000), min(n_all, p + 3000)
        cd = ord_amp[lo:hi]
        cd = cd[(t1a[cd] <= t0a[i]) | (t0a[cd] >= t1a[i] + HZ_MS)]
        cd = cd[cd != i][: args.topk]
        famp[k, :len(cd)] = cd
    filt["仅波幅最近邻(1维)"] = famp
    print(f"[sim_eval] 时间隔离完成  用时 {time.time() - t_start:.0f}s")

    # ================= 一、24h 详细指标（横截面去均值）=================
    randref = rng.integers(0, n_all, size=(nq, args.topk))
    disp_rand = float(np.nanmedian(np.nanstd(raw_a[randref], axis=1)))
    rows = []
    for v, f in filt.items():
        pe, pm, p5, pa, disp, cov, gap = [], [], [], [], [], [], []
        for k in range(nq):
            c = f[k]
            c = c[c >= 0]
            if len(c) < min(10, args.topk):
                pe.append(np.nan); pm.append(np.nan); p5.append(np.nan)
                pa.append(np.nan); disp.append(np.nan); cov.append(0)
                gap.append(np.nan)
                continue
            pe.append(float(np.nanmean(exc_a[c])))
            pm.append(float(np.nanmean(excmax_a[c])))
            p5.append(float(np.nanmean(hit5_a[c])))
            disp.append(float(np.nanstd(raw_a[c])))
            cov.append(len(c))
            gap.append(float(np.nanmean(np.abs(t0a[c] - t0a[qi[k]])) / 86400_000))
            cp = np.maximum(f[k], 0)
            cp = f[k][(f[k] >= 0) & (t1a[cp] <= t0a[qi[k]])][: args.topk]
            pa.append(float(np.nanmean(exc_a[cp])) if len(cp) >= 10 else np.nan)
        pe = np.asarray(pe); pm = np.asarray(pm); p5 = np.asarray(p5); pa = np.asarray(pa)
        q_e, q_m, q5 = exc_a[qi], excmax_a[qi], hit5_a[qi]
        okq = np.isfinite(pe) & np.isfinite(q_e)
        longm = okq & (pe > 0)
        rows.append({
            "定义": v,
            "IC(超额收益)": spearman(pe, q_e),
            "IC(只用更早邻居)": spearman(pa, q_e),
            "IC(命中+5%)": spearman(p5, q5),
            "+5%命中率首尾差": q_spread(p5, q5),
            "IC(后续最高涨幅)": spearman(pm, q_m),
            "五分位首尾差(超额)": q_spread(pe, q_e),
            "同向率": float(np.mean(np.sign(pe[okq]) == np.sign(q_e[okq]))),
            "做多规则均超额": float(q_e[longm].mean()) if longm.sum() >= 30 else np.nan,
            "邻居未来离散度": float(np.nanmedian(disp)),
            "离散度/随机": float(np.nanmedian(disp)) / disp_rand,
            "邻居平均时间距离(天)": float(np.nanmedian(gap)),
            "有效邻居数": float(np.mean(cov)),
        })
    df = pd.DataFrame(rows)
    print(f"\n=== 一、24h 后续的一致性（横截面去均值，n={nq}）===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"（随机抽 {args.topk} 段的未来离散度基准 = {disp_rand:.4f}）")

    # ================= 二、多周期表（BTC 调整）=================
    uni = np.unique(np.concatenate([f.ravel() for f in filt.values()] + [qi]))
    uni = uni[uni >= 0]
    sym_u = cats[codes[uni]] if cats is not None else None
    print(f"\n[sim_eval] 计算 {len(uni):,} 个窗口的多周期后续…")
    fw = fwd_multi(sym_u, t1a[uni], HORIZONS)

    # BTC 基准：对所有出现的 t1 取 BTC 同期后续
    t_uni = np.unique(t1a[uni])
    bw = fwd_multi(np.full(len(t_uni), BENCH, dtype=object), t_uni, HORIZONS)
    bmap = {h: {int(t): float(bw[h][k, 0]) for k, t in enumerate(t_uni)} for h in HORIZONS}

    pos = {int(j): k for k, j in enumerate(uni)}
    qi_pos = np.array([pos[int(i)] for i in qi])
    multi = []
    for v, f in filt.items():
        row = {"定义": v}
        for h, nb in HORIZONS.items():
            fp = np.full((nq, args.topk), -1, np.int64)
            for k in range(nq):
                cc = f[k][f[k] >= 0]
                cc = cc[np.isfinite(fw[h][[pos[int(x)] for x in cc], 0])] if len(cc) else cc
                if len(cc):
                    fp[k, :len(cc)] = [pos[int(x)] for x in cc]
            pred = np.full(nq, np.nan); tgt = np.full(nq, np.nan)
            for k in range(nq):
                cc = fp[k][fp[k] >= 0]
                if len(cc) < 10:
                    continue
                nb_ret = fw[h][cc, 0]
                nb_b = np.array([bmap[h].get(int(t1a[uni[p]]), np.nan) for p in cc])
                pred[k] = np.nanmean(nb_ret - nb_b)
                bq = bmap[h].get(int(t1a[qi[k]]), np.nan)
                tgt[k] = fw[h][qi_pos[k], 0] - bq if np.isfinite(bq) else np.nan
            row[h] = spearman(pred, tgt)
        multi.append(row)
    dm = pd.DataFrame(multi)
    print(f"\n=== 二、不同持仓周期下的一致性（IC，BTC 调整超额）===")
    print(dm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.index
    f_csv = RESULTS / f"sim_eval_{tag}.csv"
    f_hz = RESULTS / f"sim_eval_{tag}_horizon.csv"
    f_js = RESULTS / f"sim_eval_{tag}.json"
    df.to_csv(f_csv, index=False, encoding="utf-8-sig")
    dm.to_csv(f_hz, index=False, encoding="utf-8-sig")
    f_js.write_text(json.dumps({
        "index": args.index, "n_query": int(nq), "topk": args.topk, "pool": args.pool,
        "n_windows": int(n_all), "seed": args.seed,
        "disp_rand": disp_rand, "horizons": HORIZONS,
        "table_24h": df.to_dict("records"),
        "table_horizon_ic": dm.to_dict("records"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[sim_eval] -> results/{f_csv.name} · {f_hz.name}  用时 {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
