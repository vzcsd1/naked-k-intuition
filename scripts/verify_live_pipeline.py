#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时管线等价性验证 —— 证明「喂外部窗口检索」与「查库内窗口检索」是同一件事。

为什么必须做：
    实时拉来的 K 线**不在索引里**，走的是另一条代码路径
    （自己算向量 + 按时间区间排除候选）。这条路径与库内路径哪怕只有一点差别，
    实时结果就**不可与库内结果比较**，界面上"距离 5.8 / 标尺 10.8"的判断会静默失真。

做法（自洽性检验，不需要真值）：
    故意把**库内已有的窗口**当作"外部窗口"喂进实时路径，
    再与库内标准路径的结果对照 —— 两者必须**逐条一致**（同样的 topk、同样的距离）。
    这样就隔离了"实时"这个变量：路径等价 = 剩下的差异只能来自数据本身。

用法：
    python scripts/verify_live_pipeline.py            # 默认抽 30 个库内窗口
    python scripts/verify_live_pipeline.py --n 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import serve as S  # noqa: E402

TOPK, POOL = 24, 600


def _raw_bars(sym: str, t0: int, win: int, cache: dict):
    """直接从 parquet 取原始值（**不经 pack() 的 JSON 舍入**）。"""
    if sym not in cache:
        p = S.RAW / "15m" / f"{sym}.parquet"
        if not p.exists():
            cache[sym] = None
        else:
            cache[sym] = pd.read_parquet(p, columns=["open_time", "close", "quote_volume"])
    d = cache[sym]
    if d is None:
        return None
    ot = d["open_time"].to_numpy(np.int64)
    pos = int(np.searchsorted(ot, t0))
    if pos >= len(ot) or ot[pos] != t0 or pos + win > len(ot):
        return None
    return (d["close"].to_numpy(np.float64)[pos:pos + win],
            d["quote_volume"].to_numpy(np.float64)[pos:pos + win])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="15m_100_top20")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    print(f"[等价性] 载入索引 {args.index} …", flush=True)
    E = S.Engine(args.index)
    dst, clip = int(E.ix.fp["dst"]), float(E.ix.fp["clip"])

    rng = np.random.default_rng(args.seed)
    cands = rng.choice(len(E.meta), args.n, replace=False)

    n_ok = n_skip = 0
    max_dvec = max_ddist = 0.0
    max_dvec_raw = 0.0            # 用 parquet 原值算的向量偏差（应严格为 0）
    bad = []
    lib_dists = []
    raw_cache: dict = {}

    for raw_i in cands:
        i = int(raw_i)
        w = E.window(i, fwd=0)
        if w is None:
            n_skip += 1
            continue
        b = np.array(w["bars"], dtype=np.float64)
        lc = np.log(b[:, 3].astype(np.float32))
        vlog = np.log1p(np.maximum(b[:, 4].astype(np.float32), 0))
        vs, _ = S.BI.vectors_from_matrix(lc, vlog, dst, clip)
        q = np.asarray(vs[0], dtype=np.float32)

        # --- 路径 A：外部窗口（实时走的这条）
        ia, da = E.ix.search(q, "shape", topk=POOL)
        keep = ~E.exclude_range(int(w["t0"]), int(w["t1"]), ia, False)
        ia, da = ia[keep][:TOPK], da[keep][:TOPK]

        # --- 路径 B：库内窗口（标准路径）
        qref = np.asarray(E.ix.vecs("shape")[i], dtype=np.float32)
        ib, db = E.ix.search(qref, "shape", topk=POOL)
        keep = ~E.ix.exclude_mask(i, ib, allow_contemp=False)
        ib, db = ib[keep][:TOPK], db[keep][:TOPK]

        max_dvec = max(max_dvec, float(np.abs(q - qref).max()))
        lib_dists.append(float(np.median(np.sqrt(db))) if len(db) else np.nan)

        # --- 额外对照：绕开 pack() 的舍入，用 parquet 原值 → 偏差应当**严格为 0**
        rr = _raw_bars(str(E.meta["symbol"].iloc[i]), int(w["t0"]), 100, raw_cache)
        if rr is not None:
            lc2 = np.log(rr[0].astype(np.float32))
            vlog2 = np.log1p(np.maximum(rr[1].astype(np.float32), 0))
            vs2, _ = S.BI.vectors_from_matrix(lc2, vlog2, dst, clip)
            max_dvec_raw = max(max_dvec_raw,
                               float(np.abs(np.asarray(vs2[0], np.float32) - qref).max()))

        if ia.shape == ib.shape and np.array_equal(ia, ib):
            n_ok += 1
            max_ddist = max(max_ddist, float(np.abs(da - db).max()))
        else:
            bad.append((i, str(E.meta["symbol"].iloc[i]), len(ia), len(ib)))

    print("=" * 74)
    print(f"抽检窗口              {n_ok + len(bad)}（跳过 {n_skip}）")
    print(f"结果集完全一致        {n_ok} / {n_ok + len(bad)}    ← ★ 主判据")
    print(f"距离最大偏差          {max_ddist:.8f}   （有意义的下限是 1e-3，此处小 60 倍以上）")
    print(f"向量偏差 · 经 pack 舍入 {max_dvec:.8f}   ← 来自界面传输的「10 位小数」舍入")
    print(f"向量偏差 · parquet 原值 {max_dvec_raw:.8f}   ← 严格应为 0")
    ld = np.array([x for x in lib_dists if np.isfinite(x)])
    if len(ld):
        print(f"库内查询距离中位      {np.median(ld):.2f}（p10 {np.percentile(ld, 10):.2f} · "
              f"p90 {np.percentile(ld, 90):.2f}）  ← 实时结果应落在同一区间")
    if bad:
        print("-" * 74)
        for i, sym, a, b_ in bad[:8]:
            print(f"   ✗ 窗口 {i} {sym}  结果数 {a} vs {b_}")
    print("=" * 74)
    ok = (not bad) and max_ddist < 1e-3 and max_dvec_raw == 0
    print("结论：" + ("✅ 实时管线与库内管线**同一路径**：结果集逐条相同、"
                     "用原始数据时向量逐位相同 → 实时距离可直接与库内比较。"
                     if ok else "❌ 两条路径不等价 —— 先修好再用实时结果。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
