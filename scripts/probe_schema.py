#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一次性事实核查：raw/1m 表结构 + 缺口期每日归档是否齐全（含 09-01/09-02）。"""
import io
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
S = requests.Session()
S.headers.update({"User-Agent": "lK-pangen/0.1 (research)"})
CDN = "https://data.binance.vision"

L = []
L.append("=== raw/1m 表结构 ===")
p = ROOT / "data" / "raw" / "1m" / "BTCUSDT.parquet"
d = pd.read_parquet(p)
L.append(f"文件 {p.name}  {len(d):,} 行")
L.append("列: " + ", ".join(f"{c}({d[c].dtype})" for c in d.columns))
L.append("尾部 3 行:")
L.append(d.tail(3).to_string())

L.append("")
L.append("=== 缺口期每日归档是否齐全（BTCUSDT 1m / 09-01~09-17）===")
for day in [f"2026-09-{i:02d}" for i in range(1, 18)]:
    url = f"{CDN}/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{day}.zip"
    try:
        r = S.get(url, timeout=25)
        if r.status_code == 200 and len(r.content) > 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            nm = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(nm) as fh:
                txt = fh.read().decode("utf-8", "replace")
            lines = [x for x in txt.splitlines() if x.strip()]
            hdr = not lines[0][:1].isdigit()
            body = lines[1:] if hdr else lines
            first = body[0].split(",")[0]
            last = body[-1].split(",")[0]
            # 时间戳单位：微秒还是毫秒
            unit = "us" if int(first) > 1e14 else "ms"
            t0 = pd.to_datetime(int(first) if unit == "ms" else int(first) // 1000, unit="ms", utc=True)
            t1 = pd.to_datetime(int(last) if unit == "ms" else int(last) // 1000, unit="ms", utc=True)
            L.append(f"  {day}  OK  {len(r.content)/1024:6.0f}KB  {len(body):>5} 行  "
                     f"表头={hdr} 单位={unit}  {t0:%H:%M} ~ {t1:%H:%M}")
        else:
            L.append(f"  {day}  HTTP {r.status_code}")
    except Exception as e:                                        # noqa: BLE001
        L.append(f"  {day}  ERR {type(e).__name__}: {e}")

(ROOT / "results" / "_probe_schema.txt").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L[:8]))
sys.exit(0)
