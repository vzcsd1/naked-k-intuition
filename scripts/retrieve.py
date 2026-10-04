#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · M1 检索引擎

暴力精确检索（不做 ANN 近似）：
  840 万 × 64 维一条查询 = 5.4 亿次乘加。分块 + 预存模长，内存安全，单条查询 ~百毫秒级。
  精确解的好处：没有召回遗漏，不用调 HNSW 的 ef/M。

两个向量通道（见 build_index.py）：
  shape —— 去趋势后的形态 + 成交量（主通道，尺度无关）
  raw   —— 保留趋势的形态 + 成交量（对照通道，用于盲测第 2 候选）

用法：
  python retrieve.py info
  python retrieve.py query --index-idx 123456            # 用库内某窗口当查询
  python retrieve.py query --symbol BTCUSDT --time "2026-08-01 00:00" --topk 10
  python retrieve.py similar --index-idx 123456 --topk 20 --variant shape

⚠️ `--index` 默认 **`15m_100_top20`（干净币池：当时成交额前 20 名，约 116 万窗口）**。
   旧的全池索引 `15m_100`（842 万窗口）仍在，但里面混着稳定币与已死掉的币，**只用于对照**。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IDX = ROOT / "data" / "index"
CHUNK = 400_000          # 分块大小：400k × 64 float32 ≈ 100 MB


class Index:
    def __init__(self, idx_dir: Path):
        self.dir = Path(idx_dir)
        fp = self.dir / "_run.json"
        if not fp.exists():
            sys.exit(f"索引不存在：{self.dir}（先跑 build_index.py build）")
        self.fp = json.loads(fp.read_text(encoding="utf-8"))
        self.dim = int(self.fp["dim"])
        self.meta = pd.read_parquet(self.dir / "meta.parquet")
        self._vecs: dict[str, np.memmap] = {}
        self._norms: dict[str, np.ndarray] = {}
        self.symbols = json.loads((self.dir / "symbols.json").read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- 内部
    def vecs(self, variant: str):
        if variant not in self._vecs:
            name = "vectors_shape.npy" if variant == "shape" else "vectors_raw.npy"
            self._vecs[variant] = np.load(self.dir / name, mmap_mode="r")
        return self._vecs[variant]

    def norms(self, variant: str):
        """预存模长平方（float32, 全量 34 MB），避免每次查询重算。

        ⚠️ **必须校验缓存长度与向量行数一致**（2026-09-22 踩坑，增量重建索引后暴露）：
        `norms_*.npy` 是上一次建库留下的缓存，而增量重建会让 `vectors` 变长 →
        最坏情况是**静默按错位的位置取范数**：距离全错、却不报错，看起来"检索还能用"。
        这里把长度当作缓存有效性的一部分 —— 对不上就重算并覆盖。
        """
        if variant not in self._norms:
            cache = self.dir / f"norms_{variant}.npy"
            n_vec = int(self.vecs(variant).shape[0])
            if cache.exists():
                cached = np.load(cache)
                if int(cached.shape[0]) == n_vec:
                    self._norms[variant] = cached
                    return cached
                print(f"[retrieve] ⚠️ {cache.name} 有 {cached.shape[0]:,} 条，但向量有 "
                      f"{n_vec:,} 行（索引被增量重建过）→ 丢弃旧缓存，重新计算", flush=True)
            V = self.vecs(variant)
            n = V.shape[0]
            out = np.empty(n, dtype=np.float32)
            for s in range(0, n, CHUNK):
                e = min(s + CHUNK, n)
                blk = np.asarray(V[s:e], dtype=np.float32)
                out[s:e] = (blk ** 2).sum(axis=1)
            np.save(cache, out)
            self._norms[variant] = out
        return self._norms[variant]

    # ---------------------------------------------------------------- 检索
    def search(self, q: np.ndarray, variant="shape", topk=200, pool=3000):
        V = self.vecs(variant)
        nrm = self.norms(variant)
        n = V.shape[0]
        q32 = np.asarray(q, dtype=np.float32).reshape(-1)
        qn = float(q32 @ q32)
        cand_d = np.full(pool, np.inf, dtype=np.float32)
        cand_i = np.full(pool, -1, dtype=np.int64)
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            blk = np.asarray(V[s:e], dtype=np.float32)
            d = qn + nrm[s:e] - 2.0 * (blk @ q32)
            np.maximum(d, 0, out=d)
            k = min(pool, len(d))
            part = np.argpartition(d, k - 1)[:k]
            dv = d[part]
            keep = dv < cand_d.max() if cand_d.max() < np.inf else np.ones(len(dv), bool)
            if keep.any():
                idx = np.concatenate([cand_i, part[keep] + s])
                dd = np.concatenate([cand_d, dv[keep]])
                order = np.argsort(dd)[:pool]
                cand_i, cand_d = idx[order], dd[order]
        cand_i = cand_i[cand_i >= 0]
        order = np.argsort(cand_d[: len(cand_i)])
        return cand_i[order][:topk], cand_d[order][:topk]

    # ---------------------------------------------------------------- 辅助
    def exclude_mask(self, i: int, cand_idx: np.ndarray, allow_self=True,
                     allow_contemp=False) -> np.ndarray:
        """返回 True 表示该候选应被剔除。

        两层排除（都是"独立行情段去重"的实现）：
          1. 同币种 + 时间重叠  → 自相关（同一波行情的重叠窗口）
          2. 任意币种 + 时间重叠 → **同一个市场事件**。BTC 和 ETH 同时段走势几乎一样，
             但"同时间的 ETH"不是历史相似，是同一个事件 —— 对联想毫无价值。
        """
        m = self.meta
        s0, t0, t1 = str(m["symbol"].iloc[i]), int(m["t0"].iloc[i]), int(m["t1"].iloc[i])
        cs = m["symbol"].to_numpy().astype(str)[cand_idx]
        c0 = m["t0"].to_numpy()[cand_idx]
        c1 = m["t1"].to_numpy()[cand_idx]
        time_overlap = (c0 <= t1) & (c1 >= t0)
        bad = time_overlap if not allow_contemp else np.zeros(len(cand_idx), bool)
        if not allow_self:
            bad |= (cs == s0) & time_overlap
        return bad


def find_window(ix: Index, symbol: str, time_str: str) -> int:
    t = int(pd.Timestamp(time_str, tz="UTC").timestamp() * 1000)
    m = ix.meta
    sel = m[(m["symbol"].astype(str) == symbol) & (m["t0"] <= t) & (m["t1"] >= t)]
    if sel.empty:
        sys.exit(f"没有覆盖 {symbol} {time_str} 的窗口")
    return int(sel.index[0])


def fmt(ms: int) -> str:
    return str(pd.Timestamp(int(ms), unit="ms", tz="UTC"))[:16]


def cmd_info(args):
    ix = Index(IDX / args.index)
    print(f"索引 {args.index}")
    for k, v in ix.fp.items():
        print(f"  {k:<12} {v}")
    print(f"  meta 行数    {len(ix.meta):,}")
    print(f"  覆盖区间     {fmt(ix.meta['t0'].min())} ~ {fmt(ix.meta['t1'].max())}")
    return 0


def cmd_query(args):
    ix = Index(IDX / args.index)
    if args.symbol and args.time:
        i = find_window(ix, args.symbol.upper(), args.time)
    else:
        i = int(args.window)
    q = np.asarray(ix.vecs(args.variant)[i], dtype=np.float32)
    idx, dist = ix.search(q, args.variant, topk=args.pool_k)
    keep = ~ix.exclude_mask(i, idx, allow_contemp=args.allow_contemp)
    idx, dist = idx[keep][: args.topk], dist[keep][: args.topk]
    m = ix.meta
    print(f"查询窗口 #{i}  {m['symbol'].iloc[i]}  {fmt(m['t0'].iloc[i])} ~ {fmt(m['t1'].iloc[i])}"
          f"  波幅 {m['amp_pct'].iloc[i]:.1f}%")
    print(f"{'#':>3} {'币种':<12}{'开始时间':<18}{'距离':>7}{'波幅%':>8}{'未来24h':>9}")
    for k, (j, d) in enumerate(zip(idx, dist), 1):
        r = m.iloc[j]
        fr = r["fwd_ret"]
        print(f"{k:>3} {str(r['symbol']):<12}{fmt(r['t0']):<18}{d:>7.3f}"
              f"{r['amp_pct']:>8.1f}{(100 * fr if np.isfinite(fr) else float('nan')):>9.2f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info")
    p.set_defaults(func=cmd_info)
    p.add_argument("--index", default="15m_100_top20")
    p = sub.add_parser("query")
    p.set_defaults(func=cmd_query)
    p.add_argument("--index", default="15m_100_top20")
    p.add_argument("--variant", default="shape", choices=["shape", "raw"])
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--symbol", default=None)
    p.add_argument("--time", default=None)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--pool-k", dest="pool_k", type=int, default=400)
    p.add_argument("--allow-contemp", dest="allow_contemp", action="store_true",
                   help="允许同时段跨币候选（默认排除：那是同一个市场事件，不是历史相似）")
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
