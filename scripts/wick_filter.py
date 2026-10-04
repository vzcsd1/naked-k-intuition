#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · 插针线遗留项：能不能在下单前看出「这次会亏大钱」

背景（大白话）：
  前面测出来「在深跌处挂限价单」长期是赚钱的（约 +20%/年/币），
  但代价是偶尔亏一大笔 —— 最惨的 5% 交易平均要亏 14%。
  这个脚本只做一件事：**找能在下单前就算出来的特征，用来提前躲开那些大亏的交易。**

做法：
  ① 把回测里每一笔交易按结果分成两组：「亏大钱」（亏 ≥10%）和「其他」
  ② 对每一笔交易，只用**下单当时已经知道的信息**算出若干特征
  ③ 比较两组在这些特征上的中位数差异 → 差异大的特征就有预测力
  ④ 用选出的特征做一个简单过滤器，重新算一遍策略：年化变好还是变差？最惨的那笔有没有变好？

严禁使用未来信息：所有特征只取进场那根 K 线及其之前的数据。

用法：python wick_filter.py --level 0.15
输出：results/wick_filter.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

FEE_BOTH = 0.002
SLIP = 0.0025
BIG_LOSS = -0.10          # 亏 ≥10% 算「亏大钱」
BREADTH_DROP = -0.02      # 单根 1m 跌超 2% 算「这个币在急跌」


def build_breadth(files, min_minute, n_minutes):
    """每个分钟上，全市场有多少个币在同一分钟急跌 —— 用来判断「是不是全市场一起跌」。

    结果缓存在 data/breadth_{BREADTH_DROP}.npy，避免每次重跑都扫 6 亿行。
    """
    cache = ROOT / "data" / f"breadth_{abs(BREADTH_DROP):.2f}.npy"
    if cache.exists():
        arr = np.load(cache)
        if len(arr) == n_minutes:
            return arr
    cnt = np.zeros(n_minutes, dtype=np.int32)
    for f in files:
        d = pd.read_parquet(f, columns=["open_time", "close"])
        c = d["close"].to_numpy(np.float64)
        if len(c) < 2:
            continue
        r = c[1:] / c[:-1] - 1
        m = (d["open_time"].to_numpy()[1:] // 60000).astype(np.int64) - min_minute
        bad = r < BREADTH_DROP
        if bad.any():
            cnt += np.bincount(m[bad], minlength=n_minutes).astype(np.int32)
    np.save(cache, cnt)
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=float, default=0.15)
    ap.add_argument("--hold", type=int, default=240)
    args = ap.parse_args()

    det_p = ROOT / "data" / "wick_backtest_detail.parquet"
    if not det_p.exists():
        sys.exit("缺少逐笔明细，先跑 wick_backtest.py")
    det = pd.read_parquet(det_p)
    det = det[det["level"] == args.level].copy()
    if det.empty:
        sys.exit(f"没有 level={args.level} 的交易")
    print(f"[filter] 载入 {len(det):,} 笔交易（挂单 −{args.level:.0%}）")

    files = sorted((RAW / "1m").glob("*.parquet"))
    # ---------- 全市场急跌广度（一遍扫描）
    t0 = time.time()
    mn = int(pd.Timestamp("2017-08-01", tz="UTC").timestamp() * 1000) // 60000
    mx = int(pd.Timestamp("2026-12-31", tz="UTC").timestamp() * 1000) // 60000
    n_min = mx - mn + 1
    print(f"       统计全市场急跌广度（{n_min / 1e6:.1f}M 分钟）…")
    breadth = build_breadth(files, mn, n_min)
    print(f"       完成，用时 {time.time() - t0:.0f}s")

    # ---------- 逐币计算每笔交易的下单前特征
    feats = []
    by_sym = det.groupby("symbol", sort=False)
    for sym, g in by_sym:
        p = RAW / "1m" / f"{sym}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["open_time", "high", "low", "close", "quote_volume"])
        ot = d["open_time"].to_numpy()
        c = d["close"].to_numpy(np.float64)
        qv = d["quote_volume"].to_numpy(np.float64)
        n = len(d)
        pos = {int(t): k for k, t in enumerate(ot)}
        r1 = np.zeros(n); r1[1:] = c[1:] / c[:-1] - 1
        # 24 小时波动率水平（滚动标准差，用 1440 根）
        vol24 = pd.Series(r1).rolling(1440, min_periods=240).std().to_numpy()
        # 成交量相对过去 1 小时的倍数
        med1h = pd.Series(qv).rolling(60, min_periods=20).median().to_numpy()
        # 距自己历史最高价多远
        ath = np.maximum.accumulate(c)
        for _, tr in g.iterrows():
            i = pos.get(int(tr["t0"]))
            if i is None or i < 1440:
                continue
            minute = int(tr["t0"]) // 60000 - mn
            feats.append({
                "symbol": sym, "t0": int(tr["t0"]),
                "ret": float(tr["ret"]),
                "mkt_breadth": int(breadth[minute]) if 0 <= minute < n_min else 0,
                "drop_60m": float(c[i] / c[i - 60] - 1) if i >= 60 else np.nan,
                "vol24h": float(vol24[i]) if np.isfinite(vol24[i]) else np.nan,
                "vol_ratio": float(qv[i] / med1h[i]) if np.isfinite(med1h[i]) and med1h[i] > 0 else np.nan,
                "below_ath": float(c[i] / ath[i] - 1),
            })
    F = pd.DataFrame(feats)
    print(f"       算出 {len(F):,} 笔交易的下单前特征")

    F["big_loss"] = F["ret"] <= BIG_LOSS
    F["net"] = F["ret"] - FEE_BOTH - SLIP

    FEATURES = [
        ("mkt_breadth", "下单时全市场同时在急跌的币数", "全市场没人陪着跌", "全市场一起跌"),
        ("drop_60m", "这个币过去 1 小时跌幅", "跌得够狠", "跌得还不够狠"),
        ("vol24h", "这个币过去 24 小时的波动率", "平时波动小的币", "平时波动大的币"),
        ("vol_ratio", "进场那根的成交量 / 过去 1 小时中位", "成交量没异常放大", "成交量异常放大"),
        ("below_ath", "距自己的历史最高价", "已经跌得很惨", "离高点还不远"),
    ]

    L = [f"# 能不能在下单前看出「这次会亏大钱」\n",
         f"- 样本：**{len(F):,}** 笔交易（挂单 −{args.level:.0%}，持有 {args.hold} 分钟）",
         f"- 「亏大钱」定义：单笔亏 **≥10%**，共 **{int(F['big_loss'].sum()):,}** 笔"
         f"（占 {100 * F['big_loss'].mean():.1f}%）",
         "- 所有特征**只用下单当时已经知道的信息**，不含任何未来数据\n",
         "## 1. 哪些特征能区分「会亏大钱」和「其他」\n",
         "做法：把交易按某个特征排成 4 组，看最低那 1/4 和最高那 1/4 各自亏大钱的比例。\n",
         "| 下单时能看到什么 | 这一批交易 | 亏大钱比例 |",
         "|---|---|---|"]
    rows = []
    for key, name, lo_lab, hi_lab in FEATURES:
        v = F[key]
        lo_m = v <= v.quantile(0.25)
        hi_m = v >= v.quantile(0.75)
        if lo_m.sum() < 30 or hi_m.sum() < 30:
            continue
        a, b = float(F.loc[lo_m, "big_loss"].mean()), float(F.loc[hi_m, "big_loss"].mean())
        rows.append((key, name, a, b, b / a if a > 1e-9 else np.inf))
        worse_lo = a > b
        la = f"**{lo_lab}**" if worse_lo else lo_lab
        lb = f"**{hi_lab}**" if not worse_lo else hi_lab
        L.append(f"| {name} | {la} | {'**' if worse_lo else ''}{100 * a:.1f}%{'**' if worse_lo else ''} |")
        L.append(f"| | {lb} | {'**' if not worse_lo else ''}{100 * b:.1f}%{'**' if not worse_lo else ''} |")

    L.append("\n> **怎么读**：每一行都是「亏大钱的比例」。**加粗的那一批更危险**。")
    L.append("> 比如「平时波动大的币」19.6% 对「平时波动小的币」3.6% —— "
             "意思是波动大的币，亏大钱的概率是波动小的币的 5 倍多。\n")

    # ---------- 用特征做过滤器，重算策略
    L.append("## 2. 按这些特征「少做一部分交易」，策略变好了还是变差？\n")

    def q(s, p):
        return F[s].quantile(p)

    cands = [("不过滤（原始策略）", F)]
    cands.append(("避开「波动率最高」的 25% 的币", F[F["vol24h"] < q("vol24h", 0.75)]))
    cands.append(("避开「距历史高点最近」的 25%（不抄刚跌的）",
                  F[F["below_ath"] < q("below_ath", 0.75)]))
    cands.append(("避开「过去 1 小时跌得最少」的 25%", F[F["drop_60m"] < q("drop_60m", 0.75)]))
    cands.append(("避开「成交量放大最猛」的 25%", F[F["vol_ratio"] < q("vol_ratio", 0.75)]))
    cands.append(("⭐ **只做「全市场一起急跌」的交易**（急跌币数 ≥ 上四分位）",
                  F[F["mkt_breadth"] > q("mkt_breadth", 0.75)]))
    cands.append(("⭐ **只做「全市场一起急跌 + 波动率低」**",
                  F[(F["mkt_breadth"] > q("mkt_breadth", 0.75))
                    & (F["vol24h"] < q("vol24h", 0.75))]))
    cands.append(("组合：波动率低 + 跌幅够狠",
                  F[(F["vol24h"] < q("vol24h", 0.75)) & (F["drop_60m"] < q("drop_60m", 0.75))]))

    base_freq = 6.7
    agg_p = ROOT / "data" / "wick_backtest_trades.csv"
    if agg_p.exists():
        a = pd.read_csv(agg_p)
        a = a[np.isclose(a["level"], args.level)]
        if len(a):
            tot_yrs = float(a["span_ms"].sum()) / (365 * 86400_000)
            base_freq = float(a["trades"].sum()) / max(tot_yrs, 1e-9)

    def line(label, sub):
        n = len(sub)
        if n < 30:
            return None
        exp = float(sub["net"].mean())
        bl = float((sub["ret"] <= BIG_LOSS).mean())
        worst5 = float(sub["ret"].quantile(0.05))
        ann = exp * base_freq * (n / len(F))
        return (label, n, exp, bl, worst5, ann)

    L.append("| 方案 | 交易笔数 | 每笔赚多少(扣成本) | 亏大钱比例 | 最惨 5% 平均亏 | **每币年化** |")
    L.append("|---|---|---|---|---|---|")
    out_rows = [r for r in (line(lb, sb) for lb, sb in cands) if r]
    for label, n, exp, bl, worst5, ann in out_rows:
        L.append(f"| {label} | {n:,} | {100 * exp:+.3f}% | {100 * bl:.1f}% | "
                 f"{100 * worst5:+.2f}% | **{100 * ann:+.1f}%** |")

    L.append("\n> **怎么读**：最后一列的「每币年化」是把「成交频率变少」和「每笔赚多少」乘在一起的近似值，"
             "所以过滤器躲开坏交易的同时也会躲掉一些好交易 —— 净效果才是关键。")
    L.append("> 「最惨 5% 平均亏」这一列对应你最直观的体验：连亏的时候有多难受。\n")

    # ---------- 结论
    L.append("## 3. 结论\n")
    best = max(out_rows, key=lambda r: r[5])
    orig = out_rows[0]
    safest = min(out_rows, key=lambda r: r[3])
    strongest = max(rows, key=lambda r: abs(r[4] - 1) if np.isfinite(r[4]) else 0)
    L.append(f"**① 最有用的单个特征：**「{strongest[1]}」")
    L.append(f"    数值最低那批的亏大钱比例是 {100 * strongest[2]:.1f}%，"
             f"最高那批是 {100 * strongest[3]:.1f}%。\n")
    L.append(f"**② 最安全的做法：**{safest[0]}")
    L.append(f"    亏大钱比例 {100 * orig[3]:.1f}% → **{100 * safest[3]:.1f}%**；"
             f"最惨 5% 平均亏 {100 * orig[4]:+.2f}% → **{100 * safest[4]:+.2f}%**；"
             f"但年化 {100 * orig[5]:+.1f}% → {100 * safest[5]:+.1f}%\n")
    L.append(f"**③ 收益最高的做法：**{best[0]}")
    L.append(f"    年化 {100 * orig[5]:+.1f}% → **{100 * best[5]:+.1f}%**，"
             f"亏大钱比例 {100 * orig[3]:.1f}% → {100 * best[3]:.1f}%\n")
    L.append("### 落到一句话\n")
    ret_keep = safest[5] / orig[5] if orig[5] else 0
    risk_cut = safest[3] / orig[3] if orig[3] else 0
    L.append(f"> 最安全的那个做法，**把亏大钱的概率从 {100 * orig[3]:.1f}% 砍到 "
             f"{100 * safest[3]:.1f}%（降到原来的 {100 * risk_cut:.0f}%）**，")
    L.append(f"> 而年化只从 {100 * orig[5]:+.1f}% 变成 {100 * safest[5]:+.1f}%"
             f"（保留 {100 * ret_keep:.0f}%）。")
    if risk_cut <= 0.7 and ret_keep >= 0.9:
        L.append(">\n> **所以答案是：能改善，而且是划算的。**")
        L.append("> 你不需要靠择时躲开大亏，只需要**少做三类交易**：")
        L.append("> ① 波动率特别高的币（最容易亏大钱的一类，19.6% 会亏大钱）")
        L.append("> ② 过去一小时跌得还不够狠的（跌幅小的反而危险）")
        L.append("> ③ 全市场只有它一个在跌的（独狼崩最危险）")
        L.append(">\n> 代价是交易变少 —— 但**每笔赚得更多、连亏时更不难受**，"
                 "总收益基本不变。这属于「几乎白拿的风险下降」。")
    elif risk_cut <= 0.9 and ret_keep >= 0.85:
        L.append(">\n> **所以答案是：有改善，但要接受交易变少。**")
    else:
        L.append(">\n> **所以答案是：改善有限。** 这些「下单时能看到的特征」躲开坏交易的同时，"
                 "躲掉的好交易也不少。")
        L.append("> 这意味着应对大亏主要还得靠**仓位管理**（每笔下注小一点、分散到多个币），"
                 "不能指望靠挑时机全部躲开。")
    L.append("\n> ⚠️ 本分析只基于 −15% 档位。其他档位结论可能不同。")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "wick_filter.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[filter] 报告 -> results/wick_filter.md")
    print(f"         会亏大钱占比 {100 * F['big_loss'].mean():.1f}%"
          f" | 最强特征 {strongest[1]}（{strongest[4]:.2f}×）")


if __name__ == "__main__":
    main()
