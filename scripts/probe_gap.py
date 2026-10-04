#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
滚动补数据 · 前置事实核查（只读，不改任何东西）

回答三个问题：
  1. 现有 raw/1m 和 raw/15m 各到哪一刻？（缺口多长）
  2. 每日归档端点（data.binance.vision 的 daily klines）现在通不通？
     —— 缺口在月中，月度归档还没生成，只能走每日归档
  3. 缺口期需要补哪些币？覆盖得住吗？

用法：python scripts/probe_gap.py
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"
CDN = "https://data.binance.vision"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "lK-pangen/0.1 (research)"})

lines: list[str] = []


def say(s=""):
    print(s)
    lines.append(s)


def ms2s(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------ 1. 现状
def step1():
    say("=" * 76)
    say("1. 现有数据到哪一刻")
    say("=" * 76)
    out = {}
    for iv in ("1m", "15m"):
        d = RAW / iv
        files = sorted(d.glob("*.parquet"))
        if not files:
            say(f"  {iv}: 无数据")
            continue
        rows = []
        for p in files:
            t = pd.read_parquet(p, columns=["open_time"])
            rows.append((p.stem, int(t["open_time"].min()), int(t["open_time"].max()), len(t)))
        rows.sort(key=lambda r: r[2])           # 按最后一根排序
        n_bars = sum(r[3] for r in rows)
        say(f"  {iv}: {len(rows)} 个币 · 合计 {n_bars:,} 根")
        say(f"      全体最早 {ms2s(min(r[1] for r in rows))}")
        say(f"      全体最晚 {ms2s(rows[-1][2])}   币={rows[-1][0]}")
        # 看看"最晚"附近的分布：有多少币真的到了最新
        lastmax = rows[-1][2]
        for lag_d in (0, 1, 7, 30):
            cut = lastmax - lag_d * 86_400_000
            k = sum(1 for r in rows if r[2] >= cut)
            say(f"      最后一根在 {lag_d:>2} 天以内的币: {k:>3} / {len(rows)}")
        out[iv] = {"n": len(rows), "last": rows[-1][2], "first": min(r[1] for r in rows)}
    return out


# ------------------------------------------------------------ 2. 端点连通性
def step2(latest_known_month: str):
    say("")
    say("=" * 76)
    say("2. 归档端点连通性（缺口在月中 → 只能走每日归档）")
    say("=" * 76)

    # 2a 月度归档：看 2026-09 有没有
    ok_month = None
    try:
        url = f"{CDN}/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-09.zip"
        t0 = time.time()
        r = SESSION.get(url, timeout=20, stream=True)
        dt = time.time() - t0
        say(f"  月度归档 2026-09        HTTP {r.status_code}  {dt:.2f}s  "
            f"{'（已生成）' if r.status_code == 200 else '（还没生成，符合预期）'}")
        r.close()
        ok_month = r.status_code == 200
    except Exception as e:                                        # noqa: BLE001
        say(f"  月度归档 2026-09        连接失败 {type(e).__name__}: {e}")

    # 2b 每日归档：试最近 5 天
    days = []
    for k in range(14, -1, -1):
        d = datetime.now(timezone.utc).timestamp() - k * 86400
        days.append(datetime.fromtimestamp(d, timezone.utc).strftime("%Y-%m-%d"))
    say("")
    say("  每日归档探测（BTCUSDT 1m，最近 15 天）:")
    got = []
    for day in days:
        url = f"{CDN}/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{day}.zip"
        try:
            t0 = time.time()
            r = SESSION.get(url, timeout=25)
            dt = time.time() - t0
            if r.status_code == 200 and len(r.content) > 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                nm = [n for n in z.namelist() if n.endswith(".csv")]
                nrow = 0
                if nm:
                    with z.open(nm[0]) as fh:
                        nrow = sum(1 for _ in fh) - 1
                got.append(day)
                say(f"    {day}  ✅ HTTP200  {len(r.content)/1024:7.0f} KB  {nrow:>5} 行  {dt:.2f}s")
            else:
                say(f"    {day}  ❌ HTTP {r.status_code}  （还没生成 / 不存在）")
        except Exception as e:                                    # noqa: BLE001
            say(f"    {day}  ❌ {type(e).__name__}: {e}")
    say("")
    say(f"  可用每日归档最新一天: {got[-1] if got else '无'}")
    return got


# ------------------------------------------------------------ 3. 缺口覆盖
def step3(info):
    say("")
    say("=" * 76)
    say("3. 缺口与覆盖率")
    say("=" * 76)
    last = info.get("1m", {}).get("last")
    if not last:
        say("  没有 1m 数据，跳过")
        return {}
    gap_start = last + 60_000
    now = int(time.time() * 1000)
    gap_days = (now - gap_start) / 86_400_000
    say(f"  1m 数据末尾     {ms2s(last)}")
    say(f"  此刻（本机钟）  {ms2s(now)}")
    say(f"  缺口            {gap_start} 起，约 {gap_days:.1f} 天")
    return {"gap_start": gap_start, "gap_days": gap_days}


def main():
    info = step1()
    step2("")
    step3(info)
    say("")
    say("=" * 76)
    (ROOT / "results" / "_probe_gap.txt").write_text("\n".join(lines), encoding="utf-8")
    say("明细 -> results/_probe_gap.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
