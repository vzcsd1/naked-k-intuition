#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""滚动补数据的进度盘点（只读，不改任何东西）。"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"
IDX = DATA / "index" / "15m_100_top20"


def ms2s(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def main():
    print("=" * 74)
    print(f"盘点时间 {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")

    # ---- 1. 原始 1m 现状
    run = sorted((RAW / "1m").glob("*_run.json"))
    metas = []
    for p in run:
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            metas.append((p.stem.replace("_run", ""), int(j["last"]), int(j["first"]),
                          int(j.get("rows", 0))))
        except Exception:                                          # noqa: BLE001
            pass
    if metas:
        last_all = max(m[1] for m in metas)
        first_all = min(m[2] for m in metas)
        print(f"\n[1] raw/1m  {len(metas)} 个币")
        print(f"    最早 {ms2s(first_all)}   最晚 {ms2s(last_all)}")
        print(f"    最后一根收盘 {ms2s(last_all + 60_000)}")

    # ---- 2. 每日归档目录
    dd = RAW / "1m_days"
    if dd.exists():
        syms = sorted(p.name for p in dd.iterdir() if p.is_dir())
        ndays = {}
        for s in syms:
            for f in (dd / s).glob("*.parquet"):
                ndays[f.stem.replace(f"{s}-1m-", "")] = ndays.get(
                    f.stem.replace(f"{s}-1m-", ""), 0) + 1
        print(f"\n[2] data/raw/1m_days/  {len(syms)} 个币")
        print(f"    日期覆盖 {min(ndays)} ~ {max(ndays)}  共 {len(ndays)} 天")
    else:
        print("\n[2] ⚠️ 没有 1m_days 目录")

    # ---- 3. touched / state
    t = DATA / "touched_syms.txt"
    print(f"\n[3] touched_syms.txt  {'存在' if t.exists() else '❌ 不存在（merge 未跑）'}")
    st = DATA / "refresh_state.json"
    if st.exists():
        j = json.loads(st.read_text(encoding="utf-8"))
        print(f"    refresh_state：days {len(j.get('days', []))} 天 · "
              f"touched {len(j.get('touched', []))} 个 · 更新于 {j.get('updated_at')}")

    # ---- 4. 快照
    for nm, p in (("向量指纹", DATA / "_snap_index_before.parquet"),
                  ("快照元数据", DATA / "_snap_index_before.json"),
                  ("流动性快照", DATA / "_snap_liq_before.parquet")):
        print(f"\n[4] {nm:<8} {'✅ 在' if p.exists() else '❌ 缺'}  "
              f"{p.stat().st_size / 1e6:.2f} MB" if p.exists() else f"[4] {nm:<8} ❌ 缺")

    # ---- 5. 索引现状
    m = pd.read_parquet(IDX / "meta.parquet")
    print(f"\n[5] 索引 {IDX.name}")
    print(f"    {len(m):,} 窗口 · {m['symbol'].nunique()} 币 · "
          f"t0 {ms2s(m['t0'].min())} ~ t1 {ms2s(m['t1'].max())}")
    print(f"    落后最新数据 {(max(x[1] for x in metas) - m['t1'].max()) / 86400000:.1f} 天"
          if metas else "")

    # ---- 6. 流动性表
    lp = DATA / "pool_liquidity.parquet"
    if lp.exists():
        L = pd.read_parquet(lp, columns=["day"])
        print(f"\n[6] pool_liquidity.parquet  {len(L):,} 行 · "
              f"{L['day'].min()} ~ {L['day'].max()}")

    # ---- 7. 磁盘
    d = shutil.disk_usage("C:/")
    print(f"\n[7] C: 可用 {d.free / 1e9:.1f} GB / 共 {d.total / 1e9:.1f} GB")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
