# -*- coding: utf-8 -*-
"""M1 检索层公共库：路径、常量、15m 派生、窗口切片、向量化。

设计依据：10_M1检索层设计.md + 08_需求校准与相似度设计.md
- 背景层窗口：15m x 100 根，步长 5
- 向量分块：shape 48（去趋势+ATR归一）· vol 8 · rng 4 · trd 4 => V_full 64 维
- 另存 V_raw 48（不去趋势，带涨跌背景），供对照盘实验 C2 用
- 窗口有效性：100 根 15m 桶全部存在，且每桶 n_bars>=13（容忍停机边界缺<=2分钟）
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_1M = os.path.join(ROOT, "data", "raw", "1m")
DIR_15M = os.path.join(ROOT, "data", "derived", "15m")
DIR_INDEX = os.path.join(ROOT, "data", "index")
DIR_RESULTS = os.path.join(ROOT, "results")

W = 100        # 窗口长度（15m 根数）
STEP = 5       # 切片步长
N_BARS_MIN = 13  # 桶内 1m 根数下限（15 根中至少 13）

DIM_SHAPE = 48  # 形状块：去趋势 + ATR 归一后的收盘路径
DIM_RAW = 48    # 原始路径（不去趋势）
DIM_VOL = 8     # 量能块：log(vol / median(vol, W))
DIM_RNG = 4     # 波动结构块：TR/ATR 路径
DIM_TRD = 4     # 趋势块：net_atr, r2, pos_in_range, range_atr
DIM_FULL = DIM_SHAPE + DIM_VOL + DIM_RNG + DIM_TRD  # = 64

FUTURE_BARS = 24  # 叠图延伸：24 根 15m = 6 小时


def list_symbols():
    fs = [f for f in os.listdir(DIR_1M) if f.endswith(".parquet")]
    return sorted(f[:-8] for f in fs)


def f15_path(sym):
    return os.path.join(DIR_15M, sym + ".parquet")


def derive_15m_from_1m(df):
    """1m DataFrame(open_time int64 ms, open/high/low/close/volume/quote_volume/trades)
    -> 15m DataFrame(open_time, open, high, low, close, volume, quote_volume, trades, n_bars)。
    只保留至少有 1 根 1m 的桶；缺失桶即缺口，不补。"""
    ot = df["open_time"].to_numpy()
    bucket = (ot // 900_000).astype(np.int64) * 900_000
    g = df.assign(open_time=bucket).groupby("open_time", sort=True)
    out = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        trades=("trades", "sum"),
        n_bars=("open", "size"),
    ).reset_index()
    return out


def load_15m(sym, columns=None):
    df = pd.read_parquet(f15_path(sym), columns=columns)
    return df


def interp_matrix(d_out, n_in):
    """线性插值重采样矩阵 (d_out x n_in)：把长度 n_in 的路径降采样到 d_out。"""
    pos = np.linspace(0.0, n_in - 1.0, d_out)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n_in - 1)
    w = pos - lo
    P = np.zeros((d_out, n_in), dtype=np.float64)
    r = np.arange(d_out)
    P[r, lo] += 1.0 - w
    P[r, hi] += w
    return P


def true_range(high, low, close):
    tr = np.empty(len(close), dtype=np.float64)
    tr[0] = high[0] - low[0]
    pc = close[:-1]
    tr[1:] = np.maximum.reduce([
        (high[1:] - low[1:]),
        np.abs(high[1:] - pc),
        np.abs(low[1:] - pc),
    ])
    return tr


def window_starts(ot, n_bars):
    """有效窗口起点（按全局 STEP 网格）。
    有效 = 100 个连续桶全部存在且每桶 n_bars >= N_BARS_MIN。ot 必须升序。"""
    ot = np.asarray(ot, dtype=np.int64)
    nb = np.asarray(n_bars)
    ok_bar = nb >= N_BARS_MIN
    # 桶间隔必须正好 15m（ot 本身就是桶起点）
    m = len(ot)
    if m < W:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, m - W + 1, STEP, dtype=np.int64)
    # 每个起点检查 [i, i+W) 全部 ok_bar 且时间连续
    idx = starts[:, None] + np.arange(W)[None, :]
    ok = ok_bar[idx].all(axis=1)
    cont = (ot[idx] - ot[idx][:, :1] == np.arange(W, dtype=np.int64) * 900_000).all(axis=1)
    return starts[ok & cont]


_P_CACHE = {}


def _P(d):
    if d not in _P_CACHE:
        _P_CACHE[d] = interp_matrix(d, W)
    return _P_CACHE[d]


def vectorize_windows(df):
    """对单币 15m DataFrame 计算全部有效窗口向量。
    返回 dict(v_shape Kx48 f32, v_raw Kx48, v_vol Kx8, v_rng Kx4, v_trd Kx4,
              start_ot K i64, end_ot K i64, atr K f32)"""
    from numpy.lib.stride_tricks import sliding_window_view as swv

    ot = df["open_time"].to_numpy(dtype=np.int64)
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)

    starts = window_starts(ot, df["n_bars"].to_numpy())
    if len(starts) == 0:
        return None
    take = lambda a: swv(a, W)[starts]  # noqa: E731

    cw = take(c)
    hw = take(h)
    lw = take(l)
    vw = take(v)
    tr_all = true_range(h, l, c)
    trw = take(tr_all)

    atr = trw.mean(axis=1)
    keep = atr > 0
    if not keep.all():
        cw, hw, lw, vw, trw, atr, starts = (
            a[keep] for a in (cw, hw, lw, vw, trw, atr, starts))
    K = len(starts)

    x = np.arange(W, dtype=np.float64)
    xc = x - x.mean()
    Sxx = float((xc * xc).sum())
    mean_y = cw.mean(axis=1)
    slope = (cw @ xc) / Sxx
    resid = cw - mean_y[:, None] - np.outer(slope, xc)
    total_var = ((cw - mean_y[:, None]) ** 2).mean(axis=1)
    r2 = 1.0 - (resid ** 2).mean(axis=1) / np.maximum(total_var, 1e-18)

    v_shape = ((resid / atr[:, None]) @ _P(DIM_SHAPE).T).astype(np.float32)

    v_raw = (((cw - cw[:, :1]) / atr[:, None]) @ _P(DIM_RAW).T).astype(np.float32)

    med = np.median(vw, axis=1)
    rel = (vw + np.maximum(med, 1e-12)[:, None] * 1e-9) / np.maximum(med, 1e-12)[:, None]
    lv = np.log(np.maximum(rel, 1e-9))
    v_vol = (lv @ _P(DIM_VOL).T).astype(np.float32)

    v_rng = ((trw / atr[:, None]) @ _P(DIM_RNG).T).astype(np.float32)

    net = (cw[:, -1] - cw[:, 0]) / atr
    rng_hi = hw.max(axis=1) - lw.min(axis=1)
    pos = (cw[:, -1] - lw.min(axis=1)) / np.maximum(rng_hi, 1e-18)
    v_trd = np.stack([net, r2, pos, rng_hi / atr], axis=1).astype(np.float32)

    return dict(
        v_shape=v_shape, v_raw=v_raw, v_vol=v_vol, v_rng=v_rng, v_trd=v_trd,
        start_ot=ot[starts].astype(np.int64),
        end_ot=ot[starts + W - 1].astype(np.int64),
        atr=atr.astype(np.float32),
    )


def vectorize_single(df100):
    """单窗口（恰好 100 根 15m 的 DataFrame，含 volume/n_bars 列）-> 同 vectorize_windows 结构，K=1。"""
    from numpy.lib.stride_tricks import sliding_window_view as swv
    assert len(df100) == W, f"query window must be {W} bars, got {len(df100)}"
    c = df100["close"].to_numpy(dtype=np.float64)
    h = df100["high"].to_numpy(dtype=np.float64)
    l = df100["low"].to_numpy(dtype=np.float64)
    v = df100["volume"].to_numpy(dtype=np.float64)
    tr = true_range(h, l, c)
    atr = float(tr.mean())
    if not np.isfinite(atr) or atr <= 0:
        raise ValueError("ATR<=0：窗口内无有效波动，无法归一化")
    x = np.arange(W, dtype=np.float64)
    xc = x - x.mean()
    Sxx = float((xc * xc).sum())
    mean_y = float(c.mean())
    slope = float((c @ xc) / Sxx)
    resid = c - mean_y - slope * xc
    total_var = float(((c - mean_y) ** 2).mean())
    r2 = 1.0 - float((resid ** 2).mean()) / max(total_var, 1e-18)
    v_shape = ((resid / atr) @ _P(DIM_SHAPE).T).astype(np.float32)[None, :]
    v_raw = (((c - c[0]) / atr) @ _P(DIM_RAW).T).astype(np.float32)[None, :]
    med = float(np.median(v))
    rel = (v + max(med, 1e-12) * 1e-9) / max(med, 1e-12)
    lv = np.log(np.maximum(rel, 1e-9))
    v_vol = (lv @ _P(DIM_VOL).T).astype(np.float32)[None, :]
    v_rng = ((tr / atr) @ _P(DIM_RNG).T).astype(np.float32)[None, :]
    net = (c[-1] - c[0]) / atr
    rng_hi = float(h.max() - l.min())
    pos = (c[-1] - l.min()) / max(rng_hi, 1e-18)
    v_trd = np.array([[net, r2, pos, rng_hi / atr]], dtype=np.float32)
    return dict(v_shape=v_shape, v_raw=v_raw, v_vol=v_vol, v_rng=v_rng, v_trd=v_trd,
                atr=atr)


BLOCKS = [("v_shape", DIM_SHAPE), ("v_raw", DIM_RAW), ("v_vol", DIM_VOL),
          ("v_rng", DIM_RNG), ("v_trd", DIM_TRD)]


def topk_bruteforce(mat_path, dim, q, k, exclude=None, chunk=1_000_000):
    """分块暴力 top-k（L2^2 越小越相似）。mat_path: .npy memmap (N, dim) f32。
    exclude: 长度 N 的 bool 数组（True=排除）。返回 (idx, dist)，按 dist 升序。"""
    mat = np.load(mat_path, mmap_mode="r")
    N = mat.shape[0]
    q = np.asarray(q, dtype=np.float32).ravel()
    best_idx = np.full(k, -1, dtype=np.int64)
    best_d = np.full(k, np.inf, dtype=np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        blk = np.asarray(mat[s:e], dtype=np.float32)
        d = (blk * blk).sum(1) - 2.0 * (blk @ q)
        if exclude is not None:
            d[exclude[s:e]] = np.inf
        loc = np.argpartition(d, min(k, len(d) - 1))[:k]
        cand_d = d[loc]
        cand_i = loc + s
        all_d = np.concatenate([best_d, cand_d])
        all_i = np.concatenate([best_idx, cand_i])
        order = np.argpartition(all_d, k)[:k]
        best_d, best_idx = all_d[order], all_i[order]
    order = np.argsort(best_d)
    return best_idx[order], best_d[order]


def load_meta():
    """返回 dict(sym, start_ot, end_ot, atr) 及 symbols 列表。"""
    m_sym = np.load(os.path.join(DIR_INDEX, "m_sym.npy"), mmap_mode="r")
    m_start = np.load(os.path.join(DIR_INDEX, "m_start.npy"), mmap_mode="r")
    m_end = np.load(os.path.join(DIR_INDEX, "m_end.npy"), mmap_mode="r")
    m_atr = np.load(os.path.join(DIR_INDEX, "m_atr.npy"), mmap_mode="r")
    with open(os.path.join(DIR_INDEX, "symbols.json"), encoding="utf-8") as f:
        symbols = json.load(f)
    return dict(sym=np.asarray(m_sym), start=np.asarray(m_start),
                end=np.asarray(m_end), atr=np.asarray(m_atr), symbols=symbols)


def overlap_mask(meta, idx, sym_i, s_ot, e_ot, frac=0.5):
    """与 (sym_i, [s_ot, e_ot]) 时间重叠超过 frac 的窗口掩码（同币种才判重叠）。"""
    same = meta["sym"] == sym_i
    inter = (np.minimum(meta["end"], e_ot) - np.maximum(meta["start"], s_ot))
    span = min(int(e_ot - s_ot), 99 * 900_000) * frac
    return same & (inter > span)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
