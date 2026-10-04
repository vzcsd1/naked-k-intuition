# -*- coding: utf-8 -*-
"""M1 检索工具：看到一段 K 线 -> 联想历史相似片段（叠图）。

入口（三选一）：
  --live  SYMBOL          实时拉取交易所最近 100 根 15m（沙箱内不可达，本机终端可用）
  --csv   PATH            数值 CSV（列 open_time,open,high,low,close,volume；1m 自动聚合 15m）
  --image PATH            截图先经 m1_screenshot_ohlcv.py 转 CSV，再走 --csv

常用选项：
  --symbol SYM    查询片段属于哪个币（用于排除自身时段重叠）
  --shape-only    只用形状通道（去趋势归一化路径）检索；缺省用 V_full 等权四通道
  --keep N        去重后保留 N 个（默认 8）
  --top N         去重前召回 N 个（默认 500）
  --future N      叠图向后延伸根数（默认 24 = 6h）
  --exclude SYM   额外排除的币种（可多次）

输出：results/m1_query_<时间戳>/ （叠图 PNG + matches.json + summary.md + input_15m.csv）
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import m1_lib as L

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402

UP, DOWN = "#26a69a", "#ef5350"


def klines_live(symbol, interval, limit):
    import requests
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params=dict(symbol=symbol, interval=interval, limit=limit),
                     timeout=15)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close",
                                     "volume", "close_time", "qv", "trades",
                                     "tb", "tq", "ig"])
    df["open_time"] = df["open_time"].astype(np.int64)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df[["open_time", "open", "high", "low", "close", "volume"]]


def load_query_df(args):
    if args.live:
        try:
            df = klines_live(args.live, "15m", L.W)
            df1m = klines_live(args.live, "1m", 60)
            sym = args.live
        except Exception as e:
            raise SystemExit(
                f"实时拉取失败（{e}）。\n沙箱内 api.binance.com 不可达——"
                f"请在沙箱外的本机终端运行，或改用 --csv / --image 入口。")
    elif args.csv:
        df = pd.read_csv(args.csv)
        cols = {"open_time", "open", "high", "low", "close", "volume"}
        if not cols.issubset(df.columns):
            raise SystemExit(f"CSV 需要列 {sorted(cols)}，实际 {list(df.columns)}")
        sym = args.symbol or os.path.splitext(os.path.basename(args.csv))[0]
        df1m = None
    elif args.image:
        raise SystemExit("先用 scripts/m1_screenshot_ohlcv.py 把截图转 CSV，再 --csv 输入。")
    else:
        raise SystemExit("必须指定 --live / --csv / --image 之一")
    df = df.sort_values("open_time").reset_index(drop=True)
    # 1m 粒度自动聚合为 15m
    diffs = np.diff(df["open_time"].to_numpy(np.int64))
    med = np.median(diffs) if len(diffs) else 60_000
    if med < 900_000:
        df = L.derive_15m_from_1m(df)
    if len(df) < L.W:
        raise SystemExit(f"聚合后只有 {len(df)} 根 15m，不足 {L.W} 根")
    df = df.tail(L.W).reset_index(drop=True)
    return df, sym, df1m


def recall(qvec, args, meta):
    mat_name = "v_shape" if args.shape_only else "v_full"
    mat_path = os.path.join(L.DIR_INDEX, mat_name + ".npy")
    if args.shape_only:
        q = qvec["v_shape"][0]
    else:
        scales = json.load(open(os.path.join(L.DIR_INDEX, "blocks_scale.json"),
                                encoding="utf-8"))
        q = np.concatenate([
            qvec["v_shape"][0] / np.float32(scales["v_shape"]),
            qvec["v_vol"][0] / np.float32(scales["v_vol"]),
            qvec["v_rng"][0] / np.float32(scales["v_rng"]),
            qvec["v_trd"][0] / np.float32(scales["v_trd"]),
        ])
    exclude = np.zeros(meta["sym"].shape[0], dtype=bool)
    if args.symbol and args.symbol in meta["symbols"]:
        si = meta["symbols"].index(args.symbol)
        # 查询时段本身不存在于索引（近端），同币全时段照常可召回；仅显式排除
        pass
    for s in args.exclude or []:
        if s in meta["symbols"]:
            exclude |= meta["sym"] == meta["symbols"].index(s)
    idx, dist = L.topk_bruteforce(mat_path, None, q, args.top, exclude=exclude)
    return idx, dist


def dedup(idx, dist, meta, keep):
    kept = []
    for i, d in zip(idx, dist):
        if i < 0:
            continue
        si = int(meta["sym"][i])
        s_ot, e_ot = int(meta["start"][i]), int(meta["end"][i])
        ok = True
        for j in kept:
            if int(meta["sym"][j]) != si:
                continue
            inter = min(int(meta["end"][j]), e_ot) - max(int(meta["start"][j]), s_ot)
            if inter > 0.5 * (e_ot - s_ot):
                ok = False
                break
        if ok:
            kept.append(int(i))
        if len(kept) >= keep:
            break
    return kept


def future_bars(sym, e_ot, n):
    df = L.load_15m(sym, columns=["open_time", "open", "high", "low", "close", "volume"])
    ot = df["open_time"].to_numpy(np.int64)
    p = int(np.searchsorted(ot, e_ot, side="right"))
    out = df.iloc[p:p + n].reset_index(drop=True)
    return out


def draw_candles(ax, df, width=0.6):
    x = np.arange(len(df))
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    col = np.where(c >= o, UP, DOWN)
    ax.vlines(x, l, h, color=col, linewidth=0.7, zorder=1)
    body_lo = np.minimum(o, c)
    body_h = np.abs(c - o)
    body_h = np.where(body_h <= 0, (h - l).max() * 1e-4, body_h)
    ax.bar(x, body_h, bottom=body_lo, width=width, color=col, linewidth=0, zorder=2)


def draw_vol(ax, df, width=0.6):
    x = np.arange(len(df))
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    col = np.where(c >= o, UP, DOWN)
    ax.bar(x, df["volume"].to_numpy(float), width=width, color=col, alpha=0.55, linewidth=0)
    ax.set_yscale("log")
    ax.tick_params(labelleft=False, left=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("vol", fontsize=7)


def norm_path(df_close, atr):
    c = df_close.to_numpy(float)
    return (c - c[0]) / atr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live")
    ap.add_argument("--csv")
    ap.add_argument("--symbol")
    ap.add_argument("--shape-only", action="store_true")
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--future", type=int, default=L.FUTURE_BARS)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    qdf, sym, df1m = load_query_df(args)
    qvec = L.vectorize_single(qdf)
    meta = L.load_meta()
    idx, dist = recall(qvec, args, meta)
    kept = dedup(idx, dist, meta, args.keep)

    out = args.out or os.path.join(
        L.DIR_RESULTS, "m1_query_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    qdf.to_csv(os.path.join(out, "input_15m.csv"), index=False)
    if df1m is not None:
        df1m.to_csv(os.path.join(out, "input_1m.csv"), index=False)

    matches = []
    for rank, i in enumerate(kept, 1):
        si = int(meta["sym"][i])
        msym = meta["symbols"][si]
        s_ot, e_ot, atr = int(meta["start"][i]), int(meta["end"][i]), float(meta["atr"][i])
        wdf = L.load_15m(msym, columns=["open_time", "open", "high", "low", "close", "volume"])
        ot = wdf["open_time"].to_numpy(np.int64)
        p = int(np.searchsorted(ot, s_ot, side="left"))
        wdf = wdf.iloc[p:p + L.W].reset_index(drop=True)
        fut = future_bars(msym, e_ot, args.future)
        fwd = ((float(fut["close"].iloc[-1]) - float(wdf["close"].iloc[-1])) / atr
               if len(fut) else np.nan)
        matches.append(dict(rank=rank, symbol=msym, start_utc=pd.to_datetime(
                                s_ot, unit="ms", utc=True).isoformat(),
                            end_utc=pd.to_datetime(e_ot, unit="ms", utc=True).isoformat(),
                            dist=float(dist[kept.index(i)]), atr=atr,
                            forward_atr=fwd, window=wdf.to_dict("list"),
                            future=fut[["open_time", "close", "high", "low"]]
                                   .to_dict("list")))

    # ---- 渲染 ----
    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.4, 0.7, 2.2],
                          hspace=0.16, left=0.07, right=0.98, top=0.94, bottom=0.07)
    ax1 = fig.add_subplot(gs[0])
    ax1v = fig.add_subplot(gs[1], sharex=ax1)
    draw_candles(ax1, qdf)
    draw_vol(ax1v, qdf)
    ax1.tick_params(labelbottom=False)
    ax1v.tick_params(labelbottom=False)
    ax1.set_title(f"QUERY  {sym}  {pd.to_datetime(qdf['open_time'].iloc[0], unit='ms', utc=True):%Y-%m-%d %H:%M} -> "
                  f"{pd.to_datetime(qdf['open_time'].iloc[-1], unit='ms', utc=True):%H:%M} UTC  (15m x {L.W})",
                  fontsize=10)
    ax1.grid(alpha=0.2)
    ax2 = fig.add_subplot(gs[2])
    cmap = plt.get_cmap("viridis")
    for m in matches:
        wclose = pd.Series(m["window"]["close"])
        y = norm_path(wclose, m["atr"])
        color = cmap((m["rank"] - 1) / max(1, len(matches) - 1))
        ax2.plot(np.arange(len(y)), y, color=color, lw=1.4,
                 label=f"#{m['rank']} {m['symbol']} {m['start_utc'][:16]}")
        fcl = pd.Series(m["future"]["close"]) if m["future"]["close"] else None
        if fcl is not None and len(fcl):
            fy = (fcl.to_numpy(float) - float(wclose.iloc[0])) / m["atr"]
            ax2.plot(np.arange(len(y) - 1, len(y) - 1 + len(fy)), fy, color=color,
                     lw=1.0, ls="--", alpha=0.55)
    ax2.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax2.axvline(L.W - 1, color="k", lw=0.5, alpha=0.3)
    ax2.text(L.W - 1, ax2.get_ylim()[1], " now ", va="top", fontsize=8, alpha=0.6)
    ax2.legend(fontsize=7, ncol=2, loc="best")
    ax2.set_title("normalized matches (solid=window, dashed=future, unit=ATR)",
                  fontsize=9)
    ax2.grid(alpha=0.2)
    fig.savefig(os.path.join(out, "overlay.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    L.write_json(os.path.join(out, "matches.json"), matches)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# M1 检索结果\n\n查询：{sym} "
                f"{pd.to_datetime(qdf['open_time'].iloc[0], unit='ms', utc=True):%Y-%m-%d %H:%M}"
                f" ~ {pd.to_datetime(qdf['open_time'].iloc[-1], unit='ms', utc=True):%Y-%m-%d %H:%M} UTC"
                f"（15m x {L.W}，ATR={qvec['atr']:.6g}）\n\n")
        f.write(f"通道：{'形状 only' if args.shape_only else 'V_full 等权四通道'}；"
                f"召回 {args.top} -> 重叠去重 -> 保留 {len(kept)}\n\n"
                f"| # | 币种 | 起点(UTC) | dist | 后续{args.future}根(±ATR) |\n|---|---|---|---|---|\n")
        for m in matches:
            f.write(f"| {m['rank']} | {m['symbol']} | {m['start_utc'][:16]} | "
                    f"{m['dist']:.2f} | {m['forward_atr']:+.2f} |\n")
    print(f"done in {time.time()-t0:.0f}s -> {out}")
    for m in matches:
        print(f"  #{m['rank']} {m['symbol']:14s} {m['start_utc'][:16]} "
              f"dist={m['dist']:.2f} fwd={m['forward_atr']:+.2f} ATR")


if __name__ == "__main__":
    main()
