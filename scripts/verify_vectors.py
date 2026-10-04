#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量口径校验 —— 证明「实时查询用的归一化」与「建库时用的归一化」**完全一致**。

为什么必须做这一步：
    实时拉取要把「交易所刚返回的 100 根 K 线」转成和库里同样的 64 维向量才能检索。
    如果两边口径有任何差异（截断值、去趋势方式、成交量归一），
    检索出的"距离"就**不可比**，界面上的"像度条"和"距离标尺"全部作废，
    而这种错误**不会报错、只会静默地给你错的答案**。

做法（拿库当中介物，不需要真值）：
    索引里的 `vectors_shape.npy` 是建库时算出来的 → 它就是基准。
    从原始 15m parquet 重新算一遍同样的窗口，两者必须**逐位相同**
    （都是 float16 存储 → 理论上差值为 0，出现非零就说明口径漂移）。

用法：
    python scripts/verify_vectors.py                    # 默认抽 20 币 × 10 窗口
    python scripts/verify_vectors.py --coins 40 --per 12
    python scripts/verify_vectors.py --index 15m_100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import build_index as B          # noqa: E402
import retrieve as R             # noqa: E402

RAW = ROOT / "data" / "raw"
IDX = ROOT / "data" / "index"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="15m_100_top20")
    ap.add_argument("--coins", type=int, default=20, help="抽多少个币")
    ap.add_argument("--per", type=int, default=10, help="每个币抽多少个窗口")
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    ix = R.Index(IDX / args.index)
    meta = ix.meta
    win = int(ix.fp["win"])
    dst = int(ix.fp["dst"])
    clip = float(ix.fp["clip"])
    Vshape = ix.vecs("shape")
    Vraw = ix.vecs("raw")

    print(f"[校验] 索引 {args.index}：{len(meta):,} 窗口 · win={win} dst={dst} clip={clip}")
    print("=" * 74)

    rng = np.random.default_rng(args.seed)
    syms = np.array(sorted(set(meta["symbol"].astype(str))))
    pick = rng.choice(syms, min(args.coins, len(syms)), replace=False)

    n_checked = n_missing = 0
    worst_s = worst_r = 0.0
    worst_at = None
    bad_rows = []

    for sym in pick:
        fp = RAW / "15m" / f"{sym}.parquet"
        if not fp.exists():
            n_missing += 1
            continue
        d = pd.read_parquet(fp, columns=["open_time", "close", "quote_volume"])
        ot = d["open_time"].to_numpy(np.int64)
        c_all = d["close"].to_numpy(np.float32)
        qv_all = d["quote_volume"].to_numpy(np.float32)

        rows = meta.index[meta["symbol"].astype(str) == sym]
        if len(rows) == 0:
            continue
        take = rng.choice(rows.to_numpy(), min(args.per, len(rows)), replace=False)

        for i in take:
            t0 = int(meta["t0"].iloc[i])
            pos = int(np.searchsorted(ot, t0))
            if pos >= len(ot) or ot[pos] != t0 or pos + win > len(ot):
                n_missing += 1
                continue
            c = c_all[pos:pos + win]
            qv = qv_all[pos:pos + win]
            lc = np.log(c)
            vlog = np.log1p(np.maximum(qv, 0))
            vs, vr = B.vectors_from_matrix(lc, vlog, dst, clip)

            ref_s = np.asarray(Vshape[i], dtype=np.float32)
            ref_r = np.asarray(Vraw[i], dtype=np.float32)
            ds = float(np.abs(vs[0].astype(np.float32) - ref_s).max())
            dr = float(np.abs(vr[0].astype(np.float32) - ref_r).max())
            if ds > worst_s:
                worst_s, worst_at = ds, (sym, int(meta["t0"].iloc[i]))
            worst_r = max(worst_r, dr)
            if ds > 0 or dr > 0:
                bad_rows.append((sym, t0, ds, dr))
            n_checked += 1

    print(f"抽查窗口            {n_checked}（覆盖 {len(pick)} 个币，跳过 {n_missing}）")
    print(f"shape 通道 最大偏差  {worst_s:.8f}   {'✅ 逐位一致' if worst_s == 0 else '❌ 有偏差'}")
    print(f"raw   通道 最大偏差  {worst_r:.8f}   {'✅ 逐位一致' if worst_r == 0 else '❌ 有偏差'}")
    if worst_at and worst_s:
        print(f"最差样本            {worst_at[0]} @ {pd.Timestamp(worst_at[1], unit='ms', tz='UTC')}")
    if bad_rows:
        print("-" * 74)
        print(f"⚠️ 有 {len(bad_rows)} 个窗口不一致，前 5 个：")
        for sym, t0, ds, dr in bad_rows[:5]:
            print(f"   {sym} {pd.Timestamp(t0, unit='ms', tz='UTC')}  shape差={ds:.8f} raw差={dr:.8f}")
    print("=" * 74)
    if worst_s == 0 and worst_r == 0:
        print("结论：✅ 实时查询与建库**同一口径**，距离可直接比较。")
        return 0
    print("结论：❌ 口径不一致 —— 在修好之前，不要相信实时检索的距离。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
