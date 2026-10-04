#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""滚动补数据 · 补之前先看这一眼（只读，不改任何东西）。

回答三个问题：
  1. **该不该补** —— 库/原始数据离最新归档差几天？
  2. **有没有坏文件** —— 每日归档里有没有 0 字节（沙箱写入拦截的产物，
     `refresh.py fetch` 的"已存在就跳过"**看不出来**，会让 merge 中途炸）。
  3. **上次跑到哪一步** —— 快照/下载/合并/索引各自的状态。

用法：
    python scripts/refresh_status.py              # 含联网探活
    python scripts/refresh_status.py --offline    # 不联网（只看本地）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refresh as RF                                              # noqa: E402  口径唯一

RAW, DATA = RF.RAW, RF.DATA
IDX = RF.IDX_DIR
DAY_MS = RF.DAY_MS


def ms2s(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="不联网（跳过归档探活）")
    args = ap.parse_args()

    print("=" * 76)
    print(f"滚动补数据盘点 · {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")

    # ---------- 1. 原始 1m 现状（口径来自 refresh.py，与补数据脚本完全一致）
    metas = RF.all_sym_meta()
    if not metas:
        print("\n✗ 找不到 data/raw/1m/*_run.json")
        return 1
    last_all = max(m["last"] for m in metas.values())
    print(f"\n[1] data/raw/1m   {len(metas)} 个币 · 最新一根 {ms2s(last_all)}"
          f"（{len([m for m in metas.values() if m['last'] >= last_all - 3 * DAY_MS])} 个币还活着）")

    # ---------- 2. 该不该补（要联网）
    if not args.offline:
        latest = RF._latest_archive_day()
        if latest:
            gap_ms = int(pd.Timestamp(latest, tz="UTC").value // 10**6) - last_all
            days = gap_ms // DAY_MS
            print(f"\n[2] 每日归档最新   {latest}")
            if days <= 0:
                print("     ✅ 没有缺口 —— 不需要补")
            else:
                print(f"     ⚠️ 缺 {days} 天（{ms2s(last_all + RF.MIN_MS)} 起）")
                print(f"        → 补：python scripts/refresh.py --stage fetch && "
                      f"... --stage merge/derive/liquidity/index/verify")
        else:
            print("\n[2] ⚠️ 探不到每日归档（最近 8 天全 404）→ 换数据源或稍后再试")

    # ---------- 3. 坏文件护栏（0 字节）
    bad = [p for p in RF.DAYS_DIR.rglob("*.parquet") if p.stat().st_size == 0] \
        if RF.DAYS_DIR.exists() else []
    n_day = len(list(RF.DAYS_DIR.rglob("*.parquet"))) if RF.DAYS_DIR.exists() else 0
    print(f"\n[3] 每日归档文件   {n_day:,} 个")
    if bad:
        print(f"     ❌ 0 字节文件 {len(bad)} 个（fetch 的『已存在就跳过』看不出它们）：")
        for p in bad[:5]:
            print(f"        {p.relative_to(RAW)}")
        syms = sorted({p.name.split('-')[0] for p in bad})
        print(f"        → 定点重下：python scripts/refresh.py --stage fetch "
              f"--symbols {','.join(syms)} --force")
    else:
        print("     ✅ 没有 0 字节文件")

    # ---------- 4. 快照（没有它就无法证明"老数据没被改动"）
    print("\n[4] 改动前快照（改索引前必须存过）")
    for nm, p in (("向量指纹", DATA / "_snap_index_before.parquet"),
                  ("元数据  ", DATA / "_snap_index_before.json"),
                  ("流动性  ", DATA / "_snap_liq_before.parquet")):
        print(f"     {nm}  {'✅ ' + str(round(p.stat().st_size / 1e6, 2)) + ' MB' if p.exists() else '❌ 缺（跑 --stage snapshot）'}")

    st = RF.load_state()
    print(f"\n[5] 上次跑到哪     days {len(st.get('days', []))} 天 · "
          f"touched {len(st.get('touched', []))} 个币 · 更新于 {st.get('updated_at', '—')}")

    # ---------- 6. 索引现状
    m = pd.read_parquet(IDX / "meta.parquet")
    tmax = int(m["t1"].max())
    print(f"\n[6] 索引 {IDX.name}")
    print(f"     {len(m):,} 窗口 · {m['symbol'].nunique()} 币 · "
          f"t0 {ms2s(m['t0'].min())} ~ t1 {ms2s(tmax)}")
    lag = (last_all - tmax) / DAY_MS
    print(f"     索引落后原始数据 {lag:.2f} 天   "
          f"{'✅' if lag < 1 else '⚠️ 索引没跟上（要跑 --stage index）'}")
    for f in ("vectors_shape.npy", "vectors_raw.npy", "norms_shape.npy", "norms_raw.npy"):
        p = IDX / f
        print(f"     {f:<20} {'✅ ' + str(round(p.stat().st_size / 1e6, 1)) + ' MB' if p.exists() else '❌ 缺'}")

    # ---------- 7. 流动性表 + 磁盘
    lp = DATA / "pool_liquidity.parquet"
    if lp.exists():
        L = pd.read_parquet(lp, columns=["day"])
        d0 = pd.to_datetime(int(L["day"].min()) * DAY_MS, unit="ms").date()
        d1 = pd.to_datetime(int(L["day"].max()) * DAY_MS, unit="ms").date()
        print(f"\n[7] 流动性表 {len(L):,} 行 · {d0} ~ {d1}")
    d = shutil.disk_usage("C:/")
    print(f"\n[8] C: 可用 {d.free / 1e9:.1f} GB / 共 {d.total / 1e9:.1f} GB")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
