# -*- coding: utf-8 -*-
"""M1 步骤1：1m -> 15m 派生。断点续跑：输出存在则跳过。
用法: python m1_derive_15m.py [SYMBOL ...]   # 缺省=全部
"""
import os
import sys
import time

import pandas as pd

import m1_lib as L

PY = __file__


def derive_one(sym):
    out = L.f15_path(sym)
    if os.path.exists(out):
        return "skip"
    src = os.path.join(L.DIR_1M, sym + ".parquet")
    df = pd.read_parquet(src, columns=["open_time", "open", "high", "low", "close",
                                       "volume", "quote_volume", "trades"])
    df = df.sort_values("open_time").reset_index(drop=True)
    out_df = L.derive_15m_from_1m(df)
    out_df.to_parquet(out, compression="zstd")
    return f"{len(out_df)} bars"


def main():
    syms = sys.argv[1:] or L.list_symbols()
    os.makedirs(L.DIR_15M, exist_ok=True)
    L.write_json(os.path.join(L.DIR_15M, "_run.json"), {
        "script": os.path.basename(PY),
        "bucket_ms": 900_000,
        "agg": "open=first high=max low=min close=last volume=sum trades=sum, n_bars=count",
        "empty_buckets": "dropped (gap = missing)",
        "source": "data/raw/1m",
        "n_symbols": len(syms),
    })
    t0 = time.time()
    done = 0
    for i, sym in enumerate(syms, 1):
        try:
            r = derive_one(sym)
        except Exception as e:  # 单币失败不阻塞整体，断点续跑补
            r = f"ERROR {type(e).__name__}: {e}"
        done += not r.startswith(("skip", "ERROR"))
        print(f"[{i}/{len(syms)}] {sym}: {r}", flush=True)
    print(f"derive done: {done} newly derived, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
