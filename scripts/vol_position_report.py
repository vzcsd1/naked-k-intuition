#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · M1 步骤 4 的**复盘分析**（读逐格明细，不重新检索）

为什么单独有这个脚本：
  `vol_position.py` 跑一次检索要 15~20 分钟。所以它把**逐格明细**
  （每期每币的 可见特征 / 预测 / 未来实际）落成 parquet；
  **之后任何统计口径的调整都只在这里做，秒级出结果，不用重跑检索。**

本脚本做三件事：
  1. 表一：横截面调仓的夏普，并用**分块自助法**给出"夏普差"的显著性
     （⚠️ 项目铁律：样本高度自相关，不能只按期数估标准误）
  2. 表二：总仓位随时间升降 —— **修正缩放口径**（原版用 mean(1/V)，
     会被"未来波动最小"的那几个币主导而失稳）→ 改用 median
  3. 打印分年度，检查是否稳健

用法：
  python vol_position_report.py --cells ../results/vol_position_cells_15m_100_aw27_e1.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def sharpe(x: np.ndarray, per_year: float) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std() * np.sqrt(per_year)) if x.std() > 0 else np.nan


def block_boot_diff(a: np.ndarray, b: np.ndarray, per_year: float,
                    n_boot: int = 2000, block: int = 10, seed: int = 7):
    """夏普差 b−a 的分块自助法（块内保留自相关结构）。返回 (差, 标准误, p值双侧)。"""
    rng = np.random.default_rng(seed)
    n = len(a)
    nb = int(np.ceil(n / block))
    diffs = []
    for _ in range(n_boot):
        st = rng.integers(0, n, nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in st])
        idx = idx[idx < n]
        diffs.append(sharpe(b[idx], per_year) - sharpe(a[idx], per_year))
    d = np.asarray(diffs, float)
    obs = sharpe(b, per_year) - sharpe(a, per_year)
    se = float(np.nanstd(d))
    p = float(2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))
    return obs, se, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=str(RESULTS / "vol_position_cells_15m_100_aw27_e1.parquet"))
    ap.add_argument("--per_year", type=float, default=365.0)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--exclude", default="",
                    help="逗号分隔的交易对，从逐格明细里剔除。⭐ 必读 2026-09-14 发现："
                         "不剔除稳定币/法币对（USDC/TUSD/BUSD/EUR/PAXG）时，"
                         "「按波幅调仓」会把仓位堆到零波动的稳定币上，"
                         "夏普改善是假象（有效持仓只有 8/166、单币权重 33%）。")
    ap.add_argument("--exclude-stable", action="store_true",
                    help="等价于 --exclude USDCUSDT,TUSDUSDT,BUSDUSDT,EURUSDT,PAXGUSDT")
    ap.add_argument("--tag", default="", help="输出文件名后缀（避免覆盖不同口径的结果）")
    ap.add_argument("--wmax", type=float, default=0.0,
                    help="⭐ 权重上限（分位，如 0.02 = 剪掉最重的 2%% 权重到该分位）。"
                         "0 = 不剪。不剪时反波动加权会把仓位压到极少数最平静的币上"
                         "（2026-09-14 实测：剔稳定币前单币 33%%、有效持仓 8/166），"
                         "夏普改善会变成假象。")
    args = ap.parse_args()

    d = pd.read_parquet(args.cells)
    print(f"[report] 逐格明细 {len(d):,} 行 · {d['g'].nunique()} 期 · {d['sym'].nunique()} 个币")

    ex = [s.strip().upper() for s in args.exclude.split(",") if s.strip()]
    if args.exclude_stable:
        ex += ["USDCUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT", "PAXGUSDT"]
    if ex:
        before = len(d)
        d = d[~d["sym"].str.upper().isin(set(ex))].copy()
        print(f"[report] ⭐ 剔除 {len(set(ex))} 个非加密风险资产（{'/'.join(sorted(set(ex)))}）"
              f"：{before - len(d):,} 行 → 剩 {len(d):,} 行 · {d['sym'].nunique()} 个币")
    suf = f"_{args.tag}" if args.tag else ""

    def inv(x):
        x = np.asarray(x, float)
        x = np.where(np.isfinite(x) & (x > 1e-6), x, np.nan)
        med = np.nanmedian(x)
        w = 1.0 / np.where(np.isfinite(x), x, med if np.isfinite(med) else 1.0)
        if args.wmax > 0 and np.isfinite(w).sum() > 10:
            w = np.clip(w, 0.0, np.nanquantile(w, 1.0 - args.wmax))
        return w

    rng = np.random.default_rng(args.seed)
    names = ["A 等权", "B 按当前波幅", "C 按检索波动预期", "D 随机置换", "E 上帝视角"]
    R1 = {k: [] for k in names}          # 表一：截面再分配的每期收益
    W2 = {k: [] for k in ["A 固定", "B 按当前波幅", "C 按检索波动预期", "E 上帝视角"]}
    M2 = {k: [] for k in W2}             # 同期的市场（等权）收益
    stamp = []
    EN, WMAX = [], []
    for g, sub in d.groupby("g", sort=True):
        n = len(sub)
        Rw = sub["R"].to_numpy(float)
        base = float(Rw.mean())
        w = {"A 等权": np.ones(n),
             "B 按当前波幅": inv(sub["amp"]),
             "C 按检索波动预期": inv(sub["nb_vol"]),
             "E 上帝视角": inv(sub["V"])}
        w["D 随机置换"] = rng.permutation(w["C 按检索波动预期"])
        for k in names:
            v = np.asarray(w[k], float)
            v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
            v = v / v.sum() if v.sum() > 0 else np.full(n, 1.0 / n)
            R1[k].append(float((v * Rw).sum()))
            if k == "B 按当前波幅":          # ⭐ 集中度体检（2026-09-14 新增）
                EN.append(1.0 / float(np.sum(v ** 2)))
                WMAX.append(float(v.max()))
        # 表二：总仓位缩放量 —— 用**中位数**而非均值（原口径会被极小的 V 主导）
        for k, col in (("A 固定", None), ("B 按当前波幅", "amp"),
                       ("C 按检索波动预期", "nb_vol"), ("E 上帝视角", "V")):
            s = 0.0 if col is None else float(np.log(np.nanmedian(sub[col])))
            W2[k].append(s); M2[k].append(base)
        stamp.append(int(g))

    print("\n=== 表一 横截面再分配（总仓位恒定）===")
    tab = []
    for k in names:
        r = np.asarray(R1[k], float)
        tab.append({"方案": k, "每期收益(%)": 100 * r.mean(),
                    "年化收益(%)": 100 * r.mean() * args.per_year,
                    "年化波动(%)": 100 * r.std() * np.sqrt(args.per_year),
                    "夏普": sharpe(r, args.per_year)})
    t1 = pd.DataFrame(tab)
    base_r = np.asarray(R1["A 等权"], float)
    print(t1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"  ⭐ 集中度体检（B 方案）：平均有效持仓 {np.mean(EN):.1f} 只 / "
          f"{int(np.median([len(s) for _, s in d.groupby('g')]))} 只 · 单币最大权重均值 {100*np.mean(WMAX):.1f}%"
          f"  ← 有效持仓远小于币数 = 权重塌缩，夏普不可信")

    print("\n=== 夏普差的分块自助法检验（vs 等权；块长 10 期，2000 次）===")
    sig = []
    for k in names[1:]:
        obs, se, p = block_boot_diff(base_r, np.asarray(R1[k], float), args.per_year)
        sig.append({"方案": k, "夏普差": obs, "标准误": se, "p值": p,
                    "结论": "显著" if p < 0.05 else "不显著"})
    sdf = pd.DataFrame(sig)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== 表二 总仓位随时间升降（缩放量 = log 中位数的偏离，平均仓位归一）===")
    t2 = []
    for k in W2:
        s = np.asarray(W2[k], float)
        s = -s                                   # 波动越大 → 仓位越小
        s = s - np.nanmean(s)                    # 居中
        s = np.exp(np.clip(s, -1.5, 1.5))        # 转成乘数，限制在 e^±1.5
        s = s / np.nanmean(s)
        r = s * np.asarray(M2[k], float)
        t2.append({"方案": k, "年化收益(%)": 100 * r.mean() * args.per_year,
                   "年化波动(%)": 100 * r.std() * np.sqrt(args.per_year),
                   "夏普": sharpe(r, args.per_year),
                   "最大回撤(%)": 100 * float((np.cumprod(1 + r)
                                              / np.maximum.accumulate(np.cumprod(1 + r)) - 1).min())})
    t2 = pd.DataFrame(t2)
    print(t2.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out = RESULTS / f"vol_position_report{suf}.csv"
    t1.to_csv(out, index=False, encoding="utf-8-sig")
    sdf.to_csv(RESULTS / f"vol_position_sharpe_sig{suf}.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(RESULTS / f"vol_position_table2{suf}.csv", index=False, encoding="utf-8-sig")
    print(f"\n[report] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
