#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · M1 检索库构建

两步：
  derive  1m → 15m（200 币，6.3 亿根 1m 聚合）
  build   切片 + 归一化 + 落库（840 万窗口 × 64 维）

设计要点（见 10_M1检索层设计.md）：
  · 窗口 100 根 15m、步长 5 → 约 840 万窗口
  · 每个窗口产出 **两个** 64 维向量：
      shape  去趋势后的形态（32 维）+ 成交量（32 维）← 主向量
      raw    保留趋势的形态（32 维）+ 成交量（32 维）← 用于盲测的对照候选
  · 幅度归一化用「窗口自身波动」，所以形态向量是**尺度无关**的；
    绝对幅度另存为元数据（amp_pct），不混进向量 —— 否则"涨 5%"和"涨 50%"的同一形态会被判为不同
  · 后续路径**从窗口结束的下一根算起**（不把窗口内走势当未来收益）

## ⭐ 币池过滤（2026-09-14 新增，`--pool_top`）

旧索引把 200 个交易对**全塞进去**，里面混着三类不该上场的：
稳定币/法币/黄金 5 个（USDC/TUSD/BUSD/EUR/PAXG）、68 个已下架的、以及冷到没法交易的。
→ 检索会返回近似直线的窗口；你在界面上也会看到一堆僵尸样本。

过滤规则（**只用当时可见信息**，见 `pool_governance.py`）：
  对每个窗口，取其**结束时刻的前一天**，看这个币在当时的
  「过去 30 天中位日成交额」横截面里排第几，**只保留前 `--pool_top` 名**。
  （用前一天而不是当天 → 严格 point-in-time，避免"当天成交额"这种事后信息）

死币会自动掉出去：没有撮合的日子流动性表补 0 → 30 天中位数归零 → 排名掉到最底。

全程向量化：用 sliding_window_view + 预计算插值矩阵，逐窗口循环会慢 100 倍。

用法：
  python build_index.py derive --workers 6
  python build_index.py build  --interval 15m --win 100 --stride 5
  python build_index.py build  --interval 15m --win 100 --stride 5 --pool_top 20   # 干净币池
输出：data/raw/15m/*.parquet · data/index/15m_100{,_topK}/{vectors_shape.npy, vectors_raw.npy, meta.parquet}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IDX = ROOT / "data" / "index"
DATA = ROOT / "data"

COLS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]
CLIP = 3.0          # 归一化后截断，防止极端点主导距离
DAY = 86_400_000

# 非加密风险资产（稳定币 / 法币 / 贵金属）——同 `pool_governance.py`，增删必须有理由
NON_CRYPTO = ["USDCUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT", "PAXGUSDT"]


# ---------------------------------------------------------------- 币池过滤

class PoolFilter:
    """按「当时可见的前 K 名流动性」过滤窗口。

    数据来自 `pool_governance.py` 的产物 `data/pool_liquidity.parquet`
    （币 × 天：日成交额 / 过去 30·60 天中位成交额）。

    ⚠️ 两条纪律：
      1. **排名要剔除 5 个非加密风险资产之后再排** —— 否则它们会占掉名额。
      2. **用窗口结束时刻的前 `delay` 天查表** —— 严格 point-in-time，不许用当天成交额。
    """

    def __init__(self, top: int, liq_floor: float = 0.0, delay: int = 1,
                 exclude: list[str] | None = None, src: Path | None = None):
        src = src or (DATA / "pool_liquidity.parquet")
        if not src.exists():
            sys.exit(f"没有流动性表 {src}，先跑：python pool_governance.py --stage liquidity")
        L = pd.read_parquet(src, columns=["day", "sym", "liq30"])
        drop = set(NON_CRYPTO if exclude is None else exclude)
        L = L[~L["sym"].isin(drop)]
        W = L.pivot_table(index="day", columns="sym", values="liq30")
        # ⚠️ 坑（2026-09-14 抽查发现，已量化为全期 8.2% 的名额）：
        #    流动性表把"没有撮合的日子"补成 0。如果**照 0 一起排名**，
        #    那么当天有成交的币不足 20 只时（2017-08 ~ 2019-03 共 553 天），
        #    零成交的币会靠 rank 挤进前 20 —— 等于把死币放了回来。
        #    → 先置为 NaN，让它们**不参与排名**（NaN 永远排不进前 K）。
        W = W.where(W > 0)
        self.days = W.index.to_numpy(np.int64)
        self.syms = np.asarray(W.columns, dtype=object)
        self.col = {str(s): j for j, s in enumerate(self.syms)}
        self.liq = np.nan_to_num(W.to_numpy(np.float64), nan=0.0)
        # 排名：只要 day 与 sym 的交点；NaN（零成交 / 整表极早期）当作没有流动性
        self.rank = W.rank(axis=1, ascending=False, method="first").to_numpy(np.float64)
        self.top = top
        self.liq_floor = liq_floor
        self.delay = delay
        self.day0 = int(self.days[0])

    def keep(self, sym: str, t1_ms: np.ndarray) -> np.ndarray:
        """返回与 t1_ms 同长的布尔数组：该窗口是否属于当时的前 K 名。"""
        j = self.col.get(str(sym))
        n = len(t1_ms)
        if j is None:
            return np.zeros(n, bool)
        di = (t1_ms // DAY).astype(np.int64) - self.delay - self.day0
        ok = (di >= 0) & (di < len(self.days))
        r = np.full(n, np.inf)
        q = np.zeros(n)
        r[ok] = self.rank[di[ok], j]
        q[ok] = self.liq[di[ok], j]
        return ok & (r <= self.top) & (q >= self.liq_floor)

    def describe(self) -> dict:
        return {"pool_top": int(self.top), "pool_delay_days": int(self.delay),
                "pool_liq_floor": float(self.liq_floor),
                "pool_exclude": list(NON_CRYPTO),
                "pool_src": "data/pool_liquidity.parquet"}


# ---------------------------------------------------------------- derive: 1m → Nm

def _derive_step_ms(interval: str) -> int:
    """'15m' -> 900000, '1h' -> 3600000"""
    unit = interval[-1].lower()
    num = int(interval[:-1])
    if unit == "m":
        return num * 60_000
    if unit == "h":
        return num * 3_600_000
    raise ValueError(f"不支持的周期: {interval}")


def _agg_one(args):
    src_name, interval, out_dir, force = args
    src = RAW / "1m" / f"{src_name}.parquet"
    out = Path(out_dir) / f"{src_name}.parquet"
    if out.exists() and not force:
        return src_name, "skip", 0
    step_ms = _derive_step_ms(interval)
    d = pd.read_parquet(src, columns=COLS)
    if d.empty:
        return src_name, "empty", 0
    b = (d["open_time"] // step_ms).astype("int64")
    g = d.groupby(b, sort=True)
    o = pd.DataFrame({
        "open_time": (g["open_time"].first().to_numpy() // step_ms) * step_ms,
        "open": g["open"].first().to_numpy(),
        "high": g["high"].max().to_numpy(),
        "low": g["low"].min().to_numpy(),
        "close": g["close"].last().to_numpy(),
        "volume": g["volume"].sum().to_numpy(),
        "quote_volume": g["quote_volume"].sum().to_numpy(),
        "trades": g["trades"].sum().to_numpy(),
        "n_1m": g.size().to_numpy(),      # 每根 15m 由几根 1m 聚合而来；<15 说明有缺口
    })
    o.to_parquet(out, index=False, compression="zstd")
    return src_name, "ok", len(o)


def pick_syms(all_syms: list[str], spec: str | None) -> list[str]:
    """把 `--symbols` 参数解析成实际存在的币列表。

    spec 三种写法（供滚动补数据时"只重算被改动的币"用）：
      None / ""   全部
      "A,B,C"     逗号分隔
      "@path"     从文件读，每行一个（# 开头为注释）—— 便于把上一阶段的产物直接喂进来
    """
    if not spec:
        return list(all_syms)
    if spec.startswith("@"):
        p = Path(spec[1:])
        if not p.exists():
            sys.exit(f"币清单文件不存在：{p}")
        want = set()
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                want.add(line.upper())
    else:
        want = {x.strip().upper() for x in spec.split(",") if x.strip()}
    keep = [s for s in all_syms if s.upper() in want]
    missing = want - {s.upper() for s in keep}
    if missing:
        print(f"  ⚠️ 指定了但本地没有 1m 数据的币 {len(missing)} 个（忽略）："
              f"{', '.join(sorted(missing)[:8])}")
    return keep


def cmd_derive(args):
    src_dir = RAW / "1m"
    all_syms = sorted(p.stem for p in src_dir.glob("*.parquet"))
    if not all_syms:
        sys.exit("没有 1m 数据")
    syms = pick_syms(all_syms, getattr(args, "symbols", None))
    if not syms:
        sys.exit("筛选后没有要派生的币")
    out_dir = RAW / args.interval
    out_dir.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor

    force = bool(getattr(args, "force", False))
    print(f"[derive] 待处理 {len(syms)}/{len(all_syms)} 个币"
          + ("（强制重算）" if force else "（已有则跳过）"))
    jobs = [(s, args.interval, str(out_dir), force) for s in syms]
    t0, ok, skip, tot = time.time(), 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, st, n in ex.map(_agg_one, jobs):
            if st == "ok":
                ok += 1
                tot += n
            elif st == "skip":
                skip += 1
            else:
                print(f"  {name}: {st}")
    print(f"[derive] {args.interval}: 新派生 {ok} / 跳过 {skip} / 合计 {tot:,} 根"
          f"  用时 {time.time() - t0:.0f}s -> {out_dir}")
    return 0


# ---------------------------------------------------------------- build: 切片+归一化

def interp_matrix(src_n, dst_n):
    """把 src_n 点线性重采样到 dst_n 点的稀疏矩阵。"""
    pos = np.linspace(0, src_n - 1, dst_n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, src_n - 1)
    w = (pos - lo).astype(np.float32)
    M = np.zeros((dst_n, src_n), dtype=np.float32)
    M[np.arange(dst_n), lo] = 1 - w
    M[np.arange(dst_n), hi] += w
    return M


def _zscore_clip(W: np.ndarray, clip: float = CLIP) -> np.ndarray:
    mu = W.mean(axis=1, keepdims=True)
    sd = W.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return np.clip((W - mu) / sd, -clip, clip)


def _detrend(W: np.ndarray, xc: np.ndarray, denom: float) -> np.ndarray:
    x = xc + xc.mean()
    b = (W * xc).sum(axis=1, keepdims=True) / denom
    a = W.mean(axis=1, keepdims=True) - b * x.mean()
    return W - (a + b * x)


def windows_of(a: np.ndarray, win: int, stride: int):
    """返回形状 (n_win, win) 的视图矩阵。"""
    v = np.lib.stride_tricks.sliding_window_view(a, win)[::stride]
    return v


def vectors_from_matrix(Wp: np.ndarray, Wv: np.ndarray, dst: int,
                        clip: float = CLIP):
    """⭐ **归一化的唯一实现** —— 建库与实时查询都必须走这里，否则口径会漂移。

    输入：
      Wp  log(close) 的滑窗矩阵，形状 (m, win)；单窗口请传 1 维数组
      Wv  log1p(quote_volume) 的滑窗矩阵，同形状
      dst 目标维度（本库 = 32，因此每条向量 = 32 形态 + 32 成交量 = 64 维）

    输出：(vec_shape, vec_raw)，各为 (m, 2*dst) 的 float16
      shape 去趋势（尺度无关）← 主通道，界面默认
      raw   保留趋势         ← 对照通道

    ⚠️ 传进来的矩阵若已是 float32 则**不复制**（8 亿窗口的矩阵复制一次就是 3 GB）。
    """
    Wp = np.asarray(Wp, dtype=np.float32)
    Wv = np.asarray(Wv, dtype=np.float32)
    if Wp.ndim == 1:
        Wp, Wv = Wp[None, :], Wv[None, :]
    win = int(Wp.shape[1])

    x = np.arange(win, dtype=np.float32)
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    M = interp_matrix(win, dst)

    # ---- 形态：去趋势（尺度无关）
    shape = (M @ _zscore_clip(_detrend(Wp, xc, denom), clip).T).T

    # ---- 形态：保留趋势（对照用）。窗口均值去中心后按自身波动归一
    rawp = (M @ _zscore_clip(Wp - Wp.mean(axis=1, keepdims=True), clip).T).T

    # ---- 成交量通道：相对窗口自身中位数
    vol = (M @ _zscore_clip(Wv - np.median(Wv, axis=1, keepdims=True), clip).T).T

    return (np.concatenate([shape, vol], axis=1).astype(np.float16),
            np.concatenate([rawp, vol], axis=1).astype(np.float16))


def _build_one(sym_file: Path, interval: str, win: int, stride: int, dst: int,
               pool: "PoolFilter | None" = None):
    d = pd.read_parquet(sym_file, columns=["open_time", "close", "quote_volume", "high", "low"])
    n = len(d)
    if n < win + dst + 10:
        return None, None, None
    c = d["close"].to_numpy(np.float32)
    if np.any(c <= 0):
        return None, None, None
    lc = np.log(c)
    qv = d["quote_volume"].to_numpy(np.float32)
    vlog = np.log1p(np.maximum(qv, 0))
    ot = d["open_time"].to_numpy(np.int64)

    Wp = windows_of(lc, win, stride).astype(np.float32)      # (m, win)
    Wv = windows_of(vlog, win, stride).astype(np.float32)
    m = Wp.shape[0]
    if m == 0:
        return None, None, None

    # ---- 向量（归一化的唯一实现 → 见 vectors_from_matrix，实时查询共用同一支）
    vec_shape, vec_raw = vectors_from_matrix(Wp, Wv, dst)

    # ---- 元数据（注意：必须用 log 后的高低价，直接对原始价格取 exp 会溢出）
    hi = d["high"].to_numpy(np.float32)
    lo = d["low"].to_numpy(np.float32)
    bad = (~np.isfinite(hi)) | (~np.isfinite(lo)) | (hi <= 0) | (lo <= 0)
    if bad.any():                                   # 脏值置为邻近有效值后再算
        hi = np.where(bad, c, hi)
        lo = np.where(bad, c, lo)
    loghi, loglo = np.log(hi), np.log(lo)
    Whi = windows_of(loghi, win, stride)[:m]
    Wlo = windows_of(loglo, win, stride)[:m]
    amp = (np.exp(Whi.max(axis=1) - Wlo.min(axis=1)) - 1) * 100          # 窗口总波幅 %
    vol = ((hi - lo) / np.where(c > 0, c, np.nan))
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0)
    vol_pct = windows_of(vol.astype(np.float32), win, stride)[:m].mean(axis=1) * 100

    # 后续走势：从窗口结束的下一根算起（不把窗口内当未来）
    end_i = np.arange(m) * stride + win - 1
    hz = dst * 3
    valid = end_i + hz < n
    fwd_ret = np.full(m, np.nan, np.float32)
    fwd_max = np.full(m, np.nan, np.float32)
    fwd_min = np.full(m, np.nan, np.float32)
    ii = end_i[valid]
    base = c[ii]
    idx = ii[:, None] + np.arange(1, hz + 1)[None, :]
    hh = hi[idx]
    ll = lo[idx]
    cc = c[idx[:, -1]]
    fwd_ret[valid] = cc / base - 1
    fwd_max[valid] = hh.max(axis=1) / base - 1
    fwd_min[valid] = ll.min(axis=1) / base - 1

    t0_arr = ot[np.arange(m) * stride]
    t1_arr = ot[end_i]

    # ---- 币池过滤（必须在构造 meta 之前裁，保证向量/元数据同长度）
    if pool is not None:
        keep = pool.keep(sym_file.stem, t1_arr)
        if keep.sum() == 0:
            return None, None, None
        vec_shape, vec_raw = vec_shape[keep], vec_raw[keep]
        amp, vol_pct = amp[keep], vol_pct[keep]
        fwd_ret, fwd_max, fwd_min = fwd_ret[keep], fwd_max[keep], fwd_min[keep]
        t0_arr, t1_arr = t0_arr[keep], t1_arr[keep]

    meta = pd.DataFrame({
        "symbol": sym_file.stem,
        "t0": t0_arr,
        "t1": t1_arr,
        "amp_pct": amp.astype(np.float32),
        "vol_pct": vol_pct.astype(np.float32),
        "fwd_ret": fwd_ret, "fwd_max": fwd_max, "fwd_min": fwd_min,
    })
    return vec_shape, vec_raw, meta


def cmd_build(args):
    """内存安全版：逐币落分片，再用 memmap 拼接成大数组。

    本机可用内存只有约 6 GB，而两个变体共约 2.15 GB、concat 峰值翻倍 ——
    所以不做一次性 concat，改为分片写盘 + memmap 顺序填充。
    """
    src_dir = RAW / args.interval
    files = sorted(src_dir.glob("*.parquet"))
    if not files:
        sys.exit(f"没有 {src_dir} 数据，先跑 derive")
    suf = f"_top{args.pool_top}" if args.pool_top > 0 else ""
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) \
        else (IDX / f"{args.interval}_{args.win}{suf}")
    parts = out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    syms = [f.stem for f in files]

    # ---- 滚动补数据用：只对指定币强制重算，其余复用已有分片（增量重建）
    force_syms = set()
    if getattr(args, "force_syms", None):
        force_syms = {s.upper() for s in pick_syms(syms, args.force_syms)}
        if force_syms:
            print(f"[build] 增量模式：强制重算 {len(force_syms)} 个币，其余复用分片")

    pool = None
    if args.pool_top > 0:
        pool = PoolFilter(args.pool_top, args.liq_floor, args.pool_delay)
        print(f"[build] 币池过滤：当时前 {args.pool_top} 名（截至前 {args.pool_delay} 天）"
              f"· 门槛 {args.liq_floor:.0e} USDT/天 · 剔除 {','.join(NON_CRYPTO)}")

    t0 = time.time()
    metas, n_total, n_used = [], 0, 0
    for k, f in enumerate(files, 1):
        sp, rp = parts / f"{f.stem}_shape.npy", parts / f"{f.stem}_raw.npy"
        mp = parts / f"{f.stem}_meta.parquet"
        force_this = bool(args.force) or (f.stem.upper() in force_syms)
        if sp.exists() and rp.exists() and mp.exists() and not force_this:
            metas.append(pd.read_parquet(mp))
            n_total += len(metas[-1])
            continue
        a, b, mta = _build_one(f, args.interval, args.win, args.stride, args.dst, pool)
        if a is None:
            # ⚠️ 重算后变成"空"（该币被剔出币池 / 数据不足）时，**必须把旧分片覆盖成空数组**。
            #    沙箱里删不掉文件，留着旧分片会让下面的拼接多写几行 →
            #    `vectors` 比 `meta` 长 → 索引不自洽（而且不报错，只是距离全错）。
            if force_this and sp.exists():
                print(f"  ⚠️ {f.stem} 重算后为空（已剔出币池？）→ 旧分片置空")
                np.save(sp, np.zeros((0, args.dst * 2), dtype=np.float16))
                np.save(rp, np.zeros((0, args.dst * 2), dtype=np.float16))
                # meta 也必须一起清空：只清向量会让 meta 多出旧行数 → 两边行数不一致。
                # 用 `iloc[0:0]` 取空表，schema / category 原样保留（沙箱删不掉文件，只能覆盖）。
                if mp.exists():
                    pd.read_parquet(mp).iloc[0:0].to_parquet(mp, index=False, compression="zstd")
            continue
        np.save(sp, a)
        np.save(rp, b)
        mta["symbol"] = pd.Categorical(mta["symbol"], categories=syms)
        mta.to_parquet(mp, index=False, compression="zstd")
        metas.append(mta)
        n_total += len(a)
        n_used += 1
        if k % 40 == 0:
            print(f"  [{k}/{len(files)}] 累计窗口 {n_total:,}  用时 {time.time() - t0:.0f}s", flush=True)

    if n_total == 0:
        sys.exit("没有生成任何窗口")
    dim = args.dst * 2

    for out_name, suffix in (("vectors_shape.npy", "_shape"), ("vectors_raw.npy", "_raw")):
        out = np.lib.format.open_memmap(out_dir / out_name, mode="w+",
                                        dtype=np.float16, shape=(n_total, dim))
        off = 0
        for f in files:
            p = parts / f"{f.stem}{suffix}.npy"
            if not p.exists():
                continue
            arr = np.load(p)
            out[off:off + len(arr)] = arr
            off += len(arr)
        out.flush()
        del out
        if off != n_total:
            print(f"  ⚠️ {out_name} 写入 {off} 行，与预期 {n_total} 不一致")

    M = pd.concat(metas, ignore_index=True) if len(metas) > 1 else metas[0]
    M.to_parquet(out_dir / "meta.parquet", index=False, compression="zstd")
    (out_dir / "symbols.json").write_text(json.dumps(syms, ensure_ascii=False), encoding="utf-8")
    n_used = int(M["symbol"].nunique())

    # ---- 顺手刷新「向量模长」缓存（retrieve.py 查询时用）
    #      2026-09-22 踩坑：增量重建会让向量行数变化，而旧缓存长度没跟着变 →
    #      检索时切片越界（报错，算走运）或**静默按错位取范数**（距离全错却不报错，最坏）。
    #      在这里重算，保证界面一载入就有正确缓存。
    for v_name, v_file in (("shape", "vectors_shape.npy"), ("raw", "vectors_raw.npy")):
        Vv = np.load(out_dir / v_file, mmap_mode="r")
        nn = int(Vv.shape[0])
        nrm = np.empty(nn, dtype=np.float32)
        for s in range(0, nn, 400_000):          # 与 retrieve.CHUNK 同口径
            e = min(s + 400_000, nn)
            blk = np.asarray(Vv[s:e], dtype=np.float32)
            nrm[s:e] = (blk ** 2).sum(axis=1)
        np.save(out_dir / f"norms_{v_name}.npy", nrm)
        del Vv
    print(f"        模长缓存已刷新：norms_shape/raw 各 {nn:,} 条")

    fp = {"interval": args.interval, "win": args.win, "stride": args.stride,
          "dst": args.dst, "dim": dim, "clip": CLIP,
          "n_symbols": len(set(M["symbol"].astype(str))), "n_windows": int(n_total),
          "n_coins_used": int(n_used),
          "built_at": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"}
    if pool is not None:
        fp.update(pool.describe())
    if force_syms:
        fp["last_refresh_at"] = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
        fp["last_refresh_syms"] = sorted(force_syms)
    (out_dir / "_run.json").write_text(json.dumps(fp, indent=1, ensure_ascii=False),
                                       encoding="utf-8")

    print(f"[build] {args.interval} win={args.win} stride={args.stride} dst={args.dst}")
    print(f"        窗口 {n_total:,} ｜ 维度 {dim} ｜ 交易对 {len(syms)}"
          + (f"（有窗口的 {n_used}）" if pool is not None else ""))
    print(f"        占用 shape {n_total * dim * 2 / 2**30:.2f} GB + raw "
          f"{n_total * dim * 2 / 2**30:.2f} GB ｜ 用时 {time.time() - t0:.0f}s")
    print(f"        -> {out_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("derive")
    p.set_defaults(func=cmd_derive)
    p.add_argument("--interval", default="15m")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--symbols", default=None,
                   help="只派生这些币（'A,B,C' 或 '@清单文件'）；不传=全部")
    p.add_argument("--force", action="store_true", help="已有 15m 文件也重算（滚动补数据必用）")
    p = sub.add_parser("build")
    p.set_defaults(func=cmd_build)
    p.add_argument("--interval", default="15m")
    p.add_argument("--win", type=int, default=100)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--dst", type=int, default=32)
    p.add_argument("--force", action="store_true")
    p.add_argument("--force_syms", default=None,
                   help="只强制重算这些币的分片，其余复用已有分片（增量重建）")
    p.add_argument("--out_dir", default=None, help="输出目录（默认按参数自动命名；验证时用）")
    # 币池过滤（0 = 不过滤，即旧口径全池）
    p.add_argument("--pool_top", type=int, default=0,
                   help="只保留「当时（截至前一天）过去30天中位日成交额」前 N 名；0=不过滤")
    p.add_argument("--pool_delay", type=int, default=1,
                   help="查表用的天数回退（1=用前一天，严格 point-in-time）")
    p.add_argument("--liq_floor", type=float, default=0.0,
                   help="流动性下限（USDT/天），低于它直接剔除")
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
