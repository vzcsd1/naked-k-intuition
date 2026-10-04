#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 verify ⑥：流动性表"老日期被改写"到底是真差异还是检查方法的 bug。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

b = pd.read_parquet(DATA / "_snap_liq_before.parquet")
a = pd.read_parquet(DATA / "pool_liquidity.parquet",
                    columns=["day", "sym", "liq30", "liq60"])

print("=" * 76)
print(f"快照 b: {len(b):,} 行 · day {int(b['day'].min())}~{int(b['day'].max())} · "
      f"{b['sym'].nunique()} 币 · dtypes day={b['day'].dtype} sym={b['sym'].dtype}")
print(f"新表 a: {len(a):,} 行 · day {int(a['day'].min())}~{int(a['day'].max())} · "
      f"{a['sym'].nunique()} 币 · dtypes day={a['day'].dtype} sym={a['sym'].dtype}")
print(f"b 侧 liq30 为 NaN 的行：{int(b['liq30'].isna().sum()):,}")
print(f"a 侧 liq30 为 NaN 的行：{int(a['liq30'].isna().sum()):,}")

# ---- 行集合差异（不看值，只看"有没有这一格"）
kb = set(map(tuple, b[["day", "sym"]].to_numpy()))
ka = set(map(tuple, a[["day", "sym"]].to_numpy()))
only_b = kb - ka
print(f"\n[行集合] b 有 a 没有的格：{len(only_b):,}")
if only_b:
    ob = pd.DataFrame(sorted(only_b), columns=["day", "sym"])
    print(f"          day 范围 {int(ob['day'].min())}~{int(ob['day'].max())} · "
          f"涉及 {ob['sym'].nunique()} 个币")
    print(f"          样例：{ob.head(8).to_dict('records')}")

x = b.merge(a, on=["day", "sym"], how="left", suffixes=("_b", "_a"))
nan_a = x["liq30_a"].isna() & x["liq30_b"].notna()
print(f"\n[值] b 有值但 a 缺失（真差异）：{int(nan_a.sum()):,}")
d = x.loc[nan_a]
if len(d):
    print(f"     day 范围 {int(d['day'].min())}~{int(d['day'].max())} · {d['sym'].nunique()} 个币")
    print(f"     样例：{d.head(8).to_dict('records')}")
    print(f"     币分布（前 10）：{d['sym'].value_counts().head(10).to_dict()}")

# ---- 只看 b 侧也有值的那些行（排除"两边都 NaN"的中性行）
m = x["liq30_b"].notna() & x["liq30_a"].notna()
print(f"\n[共同有值的行] {int(m.sum()):,} / {len(x):,}")
print(f"     liq30 最大偏差 {np.abs(x.loc[m,'liq30_b']-x.loc[m,'liq30_a']).max():.12f}")
print(f"     liq60 最大偏差 {np.abs(x.loc[m,'liq60_b']-x.loc[m,'liq60_a']).max():.12f}")
print("=" * 76)
