#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · M1 相似 K 线联想界面（本地服务）

做一件事：**给一段 K 线，立刻列出历史上最像的几十段**（并排 + 叠图）。

引擎 = `retrieve.Index`（暴力精确检索，干净币池 116 万窗口，一次查询 <1 秒）。

⚠️ 这个界面**不预测涨跌**。实测结论（见 `results/pool_recheck.md`）：
   8 种形态定义里 **7 种还不如"随便挑一段"**。
   它的价值是「**更快看到该看的东西**」和「**更守纪律**」，不是「告诉你接下来会涨」。

用法：
    python scripts/serve.py                    # 默认 http://127.0.0.1:8765
    python scripts/serve.py --open             # 启动并自动开浏览器
    python scripts/serve.py --port 9000
    BK_INDEX=15m_100 python scripts/serve.py   # 切到旧全池索引（仅对照）

接口（前端用，也可单独 curl）：
    GET /                 前端页面
    GET /api/info         索引信息
    GET /api/coins        可查币种
    GET /api/search?symbol=BTCUSDT&time=2025-03-01+12:00&topk=24&variant=shape
    GET /api/search?idx=123456&topk=24
    GET /api/search?random=1&amp_min=2&amp_max=60&topk=24
    GET /api/live?symbol=BTCUSDT&topk=24
        ⭐ **实时入口**：去交易所拉「此刻的最近 100 根已收盘 15m」再检索。
        数据源 = data-api.binance.vision（2026-09-17 实测：api.binance.com 本机不通）

    POST /api/shot
        ⭐ **截图入口**（2026-10-04 补上 `10` §三 的入口 B）：body 为 JSON
        {img_b64, box:[x1,y1,x2,y2], t0:"YYYY-MM-DD HH:MM"(UTC), interval:"15m",
         p_top, p_bottom, log_scale, variant, topk, contemp, after}
        截图 -> 颜色掩码反解 OHLCV -> **与库同一支归一化**（vectors_from_matrix）-> 检索。
        ⚠️ 截图读不出量能 -> 查询向量只取形态前 32 维（`_search32`），
        量能通道不参与 —— 不许拿"中性量能"混进 64 维里假装参与了。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import build_index as BI  # noqa: E402
import live_feed as LF    # noqa: E402
import m1_screenshot_ohlcv as SHOT  # noqa: E402
import retrieve as R      # noqa: E402

RAW = ROOT / "data" / "raw"
IDX = ROOT / "data" / "index"
WEB = ROOT / "web"
SHOT_DIR = ROOT / "results" / "shot_uploads"
INDEX_NAME = os.environ.get("BK_INDEX", "15m_100_top20")

WIN = 100          # 窗口长度（根）
FWD = 96           # 窗口结束后的 24h（96 根 15m），供"显示后续走势"开关


def pack(c: dict, a: int, b: int) -> list:
    """抽出 [a, b) 的 K 线： [open, high, low, close, quote_volume]。"""
    return [[round(float(c["o"][j]), 10), round(float(c["h"][j]), 10),
             round(float(c["l"][j]), 10), round(float(c["cl"][j]), 10),
             round(float(c["qv"][j]), 2)] for j in range(a, b)]


class Engine:
    def __init__(self, name: str, pool_k: int = 600, topk: int = 24):
        t = time.time()
        self.name = name
        self.ix = R.Index(IDX / name)
        self.meta = self.ix.meta
        self.pool_k = pool_k
        self.topk = topk
        self._cache: dict[str, dict | None] = {}
        self._live_cache: dict[str, dict] = {}      # 实时拉取的 K 线（短缓存，免连点打网络）
        self._lock = threading.Lock()
        self.syms = sorted(set(self.meta["symbol"].astype(str)))
        self.t_min = int(self.meta["t0"].min())
        self.t_max = int(self.meta["t1"].max())
        self.ix.norms("shape")          # 预热，避免第一查慢
        self.dist_random = self._random_scale()   # "随便挑一段"的距离标尺
        print(f"[serve] 索引 {name}：{len(self.meta):,} 窗口 · {len(self.syms)} 币 · "
              f"随机距离中位 {np.sqrt(self.dist_random):.1f} · 载入 {time.time() - t:.1f}s",
              flush=True)

    def _random_scale(self, nq=24, ncand=3000) -> float:
        """标尺：**随便挑两段**时的距离中位数。检索结果远小于它才算"真的像"。"""
        V = self.ix.vecs("shape")
        n = len(self.meta)
        rng = np.random.default_rng(20260916)
        cs = np.sort(rng.choice(n, min(ncand, n), replace=False))
        C = np.asarray(V[cs], dtype=np.float32)
        nc = (C ** 2).sum(1)
        out = []
        for i in rng.choice(n, nq, replace=False):
            q = np.asarray(V[i], dtype=np.float32)
            d = nc + float(q @ q) - 2.0 * (C @ q)
            out.append(float(np.median(np.maximum(d, 0))))
        return float(np.median(out))

    # ------------------------------------------------------------ 行情缓存
    def _sym(self, s: str):
        with self._lock:
            if s in self._cache:
                return self._cache[s]
        p = RAW / "15m" / f"{s}.parquet"
        c = None
        if p.exists():
            d = pd.read_parquet(p, columns=["open_time", "open", "high", "low",
                                            "close", "quote_volume"])
            c = {"ot": d["open_time"].to_numpy(np.int64),
                 "o": d["open"].to_numpy(np.float64),
                 "h": d["high"].to_numpy(np.float64),
                 "l": d["low"].to_numpy(np.float64),
                 "cl": d["close"].to_numpy(np.float64),
                 "qv": d["quote_volume"].to_numpy(np.float64)}
            c["pos"] = {int(x): k for k, x in enumerate(c["ot"])}
        with self._lock:
            if len(self._cache) > 240:      # 简单封顶，避免内存无限涨
                self._cache.clear()
            self._cache[s] = c
        return c

    # ------------------------------------------------------------ 单窗口
    def window(self, i: int, fwd: int = 0):
        m = self.meta
        s = str(m["symbol"].iloc[i])
        c = self._sym(s)
        if c is None:
            return None
        a = c["pos"].get(int(m["t0"].iloc[i]))
        if a is None:
            return None
        b = min(a + WIN, len(c["ot"]))
        if b - a < 20:
            return None
        fr = m["fwd_ret"].iloc[i]
        out = {"idx": int(i), "sym": s,
               "t0": int(m["t0"].iloc[i]), "t1": int(m["t1"].iloc[i]),
               "amp": round(float(m["amp_pct"].iloc[i]), 2),
               "vol": round(float(m["vol_pct"].iloc[i]), 3),
               "fwd_ret": (round(float(fr), 4) if np.isfinite(fr) else None),
               "fwd_max": (round(float(m["fwd_max"].iloc[i]), 4) if np.isfinite(m["fwd_max"].iloc[i]) else None),
               "fwd_min": (round(float(m["fwd_min"].iloc[i]), 4) if np.isfinite(m["fwd_min"].iloc[i]) else None),
               "bars": pack(c, a, b)}
        if fwd:
            out["after"] = pack(c, b, min(b + fwd, len(c["ot"])))
        return out

    # ------------------------------------------------------------ 检索
    def search(self, i: int, variant="shape", topk=None, pool_k=None, allow_contemp=False):
        topk = self.topk if topk is None else topk
        pool_k = self.pool_k if pool_k is None else pool_k
        q = np.asarray(self.ix.vecs(variant)[i], dtype=np.float32)
        idx, dist = self.ix.search(q, variant, topk=pool_k)
        keep = ~self.ix.exclude_mask(i, idx, allow_contemp=allow_contemp)
        idx, dist = idx[keep], dist[keep]
        if allow_contemp:                       # 别把自己当结果
            keep2 = idx != i
            idx, dist = idx[keep2], dist[keep2]
        return idx[:topk], dist[:topk]

    # ------------------------------------------------------------ 截图查询
    def _norms32(self, variant: str) -> np.ndarray:
        """库向量前 32 维（形态部分）的模长平方缓存。

        为什么需要：截图读不出量能，查询向量只有形态 32 维是真数据。
        若把"中性量能"混进 64 维检索，距离里会多出逐窗口不同的 ‖v_vol‖² 项，
        **排序会被"谁的量能模长小"带偏** —— 所以截图检索只在形态 32 维上做。
        缓存模式与 retrieve.Index.norms 相同：长度对不上就重算覆盖。
        """
        d = int(self.ix.fp["dst"])
        key = f"_n32_{variant}"
        if hasattr(self, key):
            return getattr(self, key)
        n = len(self.meta)
        cache = IDX / self.name / f"norms_{variant}32.npy"
        vals = None
        if cache.exists():
            c = np.load(cache)
            if int(c.shape[0]) == n:
                vals = c
        if vals is None:
            V = self.ix.vecs(variant)
            vals = np.empty(n, dtype=np.float32)
            for s in range(0, n, R.CHUNK):
                e = min(s + R.CHUNK, n)
                blk = np.asarray(V[s:e], dtype=np.float32)[:, :d]
                vals[s:e] = (blk ** 2).sum(axis=1)
            try:
                np.save(cache, vals)
            except OSError:
                pass                    # 索引目录只读也能用，只是每次重算
        setattr(self, key, vals)
        return vals

    def _search32(self, q32: np.ndarray, variant: str, pool: int):
        """只在形态前 dst 维上做暴力精确检索（数学同 Index.search，维数不同）。"""
        V = self.ix.vecs(variant)
        nrm = self._norms32(variant)
        d = int(self.ix.fp["dst"])
        n = V.shape[0]
        qn = float(q32 @ q32)
        cand_d = np.full(pool, np.inf, dtype=np.float32)
        cand_i = np.full(pool, -1, dtype=np.int64)
        for s in range(0, n, R.CHUNK):
            e = min(s + R.CHUNK, n)
            blk = np.asarray(V[s:e], dtype=np.float32)[:, :d]
            dist = qn + nrm[s:e] - 2.0 * (blk @ q32)
            np.maximum(dist, 0, out=dist)
            k = min(pool, len(dist))
            part = np.argpartition(dist, k - 1)[:k]
            dv = dist[part]
            keep = dv < cand_d.max() if cand_d.max() < np.inf else np.ones(len(dv), bool)
            if keep.any():
                idx = np.concatenate([cand_i, part[keep] + s])
                dd = np.concatenate([cand_d, dv[keep]])
                order = np.argsort(dd)[:pool]
                cand_i, cand_d = idx[order], dd[order]
        cand_i = cand_i[cand_i >= 0]
        order = np.argsort(cand_d[: len(cand_i)])
        return cand_i[order][:pool], cand_d[order][:pool]

    def shot(self, img_bytes: bytes, box, t0_ms: int, interval: str,
             p_top: float, p_bottom: float, variant="shape", topk=None,
             pool_k=None, allow_contemp=False, fwd=0, log_scale=False):
        """📷 截图入口：颜色掩码反解 OHLCV -> 同库归一化 -> 形态 32 维检索。

        返回 (payload, 提取出的 df)——df 由调用方落盘存档（出问题时能复盘标定）。
        """
        topk = self.topk if topk is None else topk
        pool_k = self.pool_k if pool_k is None else pool_k
        if not (np.isfinite(p_top) and np.isfinite(p_bottom)) or p_top <= p_bottom:
            raise ValueError("价格轴标定不对：需要 p_top > p_bottom")
        step = BI._derive_step_ms(interval)
        df = SHOT.extract_bytes(img_bytes, box, p_top, p_bottom, t0_ms, step,
                                log_scale=log_scale)
        if len(df) != WIN:
            raise ValueError(f"从截图识别出 {len(df)} 根蜡烛，需要恰好 {WIN} 根 ——"
                             f"请核对首根时间/周期，或重新框选（只框蜡烛绘图区）")
        ot = df["open_time"].to_numpy(np.int64)
        if len(ot) > 1 and not np.all(np.diff(ot) == step):
            raise ValueError("识别出的时间轴不连续（缺根/重叠）——核对周期设置或框选区域")

        o = df["open"].to_numpy(np.float64)
        h = df["high"].to_numpy(np.float64)
        l = df["low"].to_numpy(np.float64)
        cl = df["close"].to_numpy(np.float64)
        bad = (h < np.maximum(o, cl) - 1e-9) | (l > np.minimum(o, cl) + 1e-9)
        warn = []
        if bad.any():
            warn.append(f"{int(bad.sum())} 根蜡烛 high/low 与实体矛盾（像素识别边缘误差）")

        # ⭐ 与库内/实时完全同一支归一化（口径不漂移），量能以中性值占位
        lc = np.log(cl.astype(np.float32))
        vlog = np.log1p(np.ones(len(cl), dtype=np.float32))
        vs, vr = BI.vectors_from_matrix(lc, vlog, int(self.ix.fp["dst"]),
                                        float(self.ix.fp["clip"]))
        qfull = vs[0] if variant != "raw" else vr[0]
        q32 = np.asarray(qfull[: int(self.ix.fp["dst"])], dtype=np.float32)

        idx, dist = self._search32(q32, variant, pool_k)
        keep = ~self.exclude_range(int(ot[0]), int(ot[-1]), idx, allow_contemp)
        idx, dist = idx[keep][:topk], dist[keep][:topk]

        amp, volp = LF.amp_and_vol(cl, h, l)
        qbars = [[round(float(o[k]), 10), round(float(h[k]), 10),
                  round(float(l[k]), 10), round(float(cl[k]), 10), 1.0]
                 for k in range(len(ot))]

        res = []
        for j, dd in zip(idx, dist):
            w = self.window(int(j), fwd=fwd)   # 结果在库里 -> "之后 24h" 取得到
            if w is None:
                continue
            w["dist"] = round(float(dd), 4)
            res.append(w)

        note = ("📷 截图查询：量能读不出来 -> 只比对形态（量能通道不参与）；"
                "识别误差未用真实截图量化过（合成图回环 ≈1 像素/根）")
        if warn:
            note += " ｜ ⚠️ " + "；".join(warn)

        payload = {
            "query": {"idx": -1, "sym": "截图", "t0": int(ot[0]), "t1": int(ot[-1]),
                      "amp": amp, "vol": volp,
                      "fwd_ret": None, "fwd_max": None, "fwd_min": None,
                      "bars": qbars},
            "results": res, "variant": variant,
            "n_windows": int(len(self.meta)), "index": self.name,
            "dist_random": round(float(np.sqrt(self.dist_random)), 3),
            "topk": int(topk), "pool_k": int(pool_k),
            "allow_contemp": bool(allow_contemp), "note": note,
            "shot": {"interval": interval, "log_scale": bool(log_scale),
                     "p_top": float(p_top), "p_bottom": float(p_bottom),
                     "n_bars": int(len(ot)),
                     "t0_given": int(t0_ms), "first_bar": int(ot[0])},
        }
        return payload, df

    # ------------------------------------------------------------ 定位窗口
    def find(self, symbol: str, time_str: str):
        symbol = symbol.upper()
        try:
            t = int(pd.Timestamp(time_str, tz="UTC").timestamp() * 1000)
        except Exception:
            raise ValueError(f"时间格式看不懂：{time_str}（示例 2025-03-01 12:00，按 UTC）")
        m = self.meta
        s2 = m[m["symbol"].astype(str) == symbol]
        if s2.empty:
            raise ValueError(f"索引里没有这个币：{symbol}")
        sel = s2[(s2["t0"] <= t) & (s2["t1"] >= t)]
        if not sel.empty:
            return int(sel.index[0]), True
        k = (s2["t0"] - t).abs().idxmin()
        return int(k), False                    # 退一步：最接近的窗口

    def latest(self, symbol: str):
        m = self.meta
        s2 = m[m["symbol"].astype(str) == symbol]
        if s2.empty:
            raise ValueError(f"索引里没有这个币：{symbol}")
        return int(s2["t0"].idxmax())

    def rnd(self, amp_min=2.0, amp_max=60.0, after: int | None = None):
        m = self.meta
        ok = (m["amp_pct"].between(amp_min, amp_max)).to_numpy()
        if after:
            ok &= (m["t0"].to_numpy() >= after)
        cand = np.flatnonzero(ok)
        if len(cand) == 0:
            raise ValueError("没有符合波幅条件的窗口")
        return int(np.random.default_rng().choice(cand))

    # ------------------------------------------------------------ 组装响应
    def payload(self, i: int, variant="shape", topk=None, pool_k=None,
                allow_contemp=False, fwd=FWD, note: str = ""):
        q = self.window(i, fwd=fwd)
        if q is None:
            raise ValueError(f"窗口 {i} 的行情数据缺失（可能该币的 15m 文件不完整）")
        idx, dist = self.search(i, variant, topk, pool_k, allow_contemp)
        res = []
        for j, d in zip(idx, dist):
            w = self.window(int(j), fwd=fwd)
            if w is None:
                continue
            w["dist"] = round(float(d), 4)
            res.append(w)
        return {"query": q, "results": res, "variant": variant,
                "n_windows": int(len(self.meta)), "index": self.name,
                "dist_random": round(float(np.sqrt(self.dist_random)), 3),
                "topk": int(topk or self.topk), "pool_k": int(pool_k or self.pool_k),
                "allow_contemp": bool(allow_contemp), "note": note}

    # ------------------------------------------------------------ 实时拉取
    def exclude_range(self, t0: int, t1: int, cand_idx: np.ndarray,
                      allow_contemp: bool = False) -> np.ndarray:
        """按**外部时间区间**排除候选。

        实时查询的窗口不在 meta 里，用不了 `Index.exclude_mask`（那个要 meta 行号）。
        规则与它一致：任何币种只要时间与查询窗口重叠就剔除 ——
        BTC 与 ETH 同时段几乎同一根线，那是**同一个市场事件**，不是"历史相似"。
        """
        if allow_contemp:
            return np.zeros(len(cand_idx), bool)
        m = self.meta
        c0 = m["t0"].to_numpy()[cand_idx]
        c1 = m["t1"].to_numpy()[cand_idx]
        return (c0 <= t1) & (c1 >= t0)

    def live(self, symbol: str, variant="shape", topk=None, pool_k=None,
             allow_contemp=False, fwd=0, ttl_ms=15_000, note=""):
        """⭐ 拉交易所**此刻**的最近 100 根 15m → 同一套归一化 → 检索历史相似段。

        与库内查询只差两处（都是"查询窗口不在索引里"导致的）：
          · 用 `exclude_range`（按时间区间）代替 `exclude_mask`（按 meta 行号）
          · 没有"之后 24 小时" —— `fwd_ret` 一律 None，未来还没发生
        """
        topk = self.topk if topk is None else topk
        pool_k = self.pool_k if pool_k is None else pool_k
        key = symbol.upper().strip()

        now = time.time() * 1000
        with self._lock:
            c = self._live_cache.get(key)
        hit = bool(c and (now - c["at"]) < ttl_ms)
        if hit:
            bars = c["bars"]
        else:
            bars = LF.fetch_recent(key, "15m", WIN)      # 失败会抛 LiveError
            with self._lock:
                if len(self._live_cache) > 64:
                    self._live_cache.clear()
                self._live_cache[key] = {"at": now, "bars": bars}

        ot, cl, hi, lo, qv = bars["ot"], bars["cl"], bars["h"], bars["l"], bars["qv"]
        lc = np.log(cl.astype(np.float32))
        vlog = np.log1p(np.maximum(qv.astype(np.float32), 0))
        vs, vr = BI.vectors_from_matrix(lc, vlog, int(self.ix.fp["dst"]),
                                        float(self.ix.fp["clip"]))
        q = np.asarray(vs[0] if variant != "raw" else vr[0], dtype=np.float32)

        idx, dist = self.ix.search(q, variant, topk=pool_k)
        keep = ~self.exclude_range(int(ot[0]), int(ot[-1]), idx, allow_contemp)
        idx, dist = idx[keep][:topk], dist[keep][:topk]

        amp, volp = LF.amp_and_vol(cl, hi, lo)
        qbars = [[round(float(bars["o"][k]), 10), round(float(hi[k]), 10),
                  round(float(lo[k]), 10), round(float(cl[k]), 10),
                  round(float(qv[k]), 2)] for k in range(len(ot))]

        res = []
        for j, d in zip(idx, dist):
            w = self.window(int(j), fwd=fwd)      # 结果在库里 → 它们的"之后 24h"仍然取得到
            if w is None:
                continue
            w["dist"] = round(float(d), 4)
            res.append(w)

        warn = bars.get("warnings") or []
        if warn:                                  # 数据自检没过 → 在界面上明说，别静默用
            note = (note + " ｜ " if note else "") + "⚠️ 数据自检：" + "；".join(warn)

        return {
            "query": {"idx": -1, "sym": key, "t0": int(ot[0]), "t1": int(ot[-1]),
                      "amp": amp, "vol": volp,
                      "fwd_ret": None, "fwd_max": None, "fwd_min": None,
                      "bars": qbars},
            "results": res, "variant": variant,
            "n_windows": int(len(self.meta)), "index": self.name,
            "dist_random": round(float(np.sqrt(self.dist_random)), 3),
            "topk": int(topk), "pool_k": int(pool_k),
            "allow_contemp": bool(allow_contemp), "note": note,
            "live": {"symbol": key, "source": bars["source"],
                     "interval": bars["interval"], "n_bars": int(bars["n_bars"]),
                     "fetched_at": int(bars["fetched_at"]),
                     "last_open": int(bars["last_open"]),
                     "last_closed": int(bars["last_closed"]),
                     "server_now": int(bars["server_now"]),
                     "lag_s": round((int(bars["server_now"]) - int(bars["last_closed"])) / 1000, 1),
                     "cached": hit},
        }


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None          # 由 main() 注入
    server_version = "LuoK/1.0"

    def log_message(self, fmt, *a):    # 安静一点
        pass

    # ---------------------------------------------------------------- 工具
    def _send(self, body: bytes, ctype: str, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _json(self, obj, code=200):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", code)

    def _q(self, qs, key, default=None):
        v = qs.get(key)
        return v[0] if v else default

    # ---------------------------------------------------------------- 路由
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                f = WEB / "index.html"
                if not f.exists():
                    return self._send(b"index.html missing", "text/plain; charset=utf-8", 500)
                return self._send(f.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/info":
                e = self.engine
                return self._json({
                    "index": e.name, "n_windows": int(len(e.meta)),
                    "n_coins": len(e.syms),
                    "t_min": e.t_min, "t_max": e.t_max,
                    "win": WIN, "interval": "15m",
                    "dist_random": round(float(np.sqrt(e.dist_random)), 3),
                    "topk_default": e.topk, "pool_k": e.pool_k,
                    "live_source": LF.BASE})
            if u.path == "/api/coins":
                return self._json({"coins": self.engine.syms})
            if u.path == "/api/search":
                return self._json(self._search(qs))
            if u.path == "/api/live":
                return self._json(self._live(qs))
            return self._json({"error": f"未知路径 {u.path}"}, 404)
        except LF.LiveError as e:
            # 交易所连不上 / 币不存在 —— 这是"外部依赖失败"，不是本服务的 bug
            self._json({"error": str(e), "live_failed": True,
                        "hint": f"实时拉取走 {LF.BASE}；网络不通时请改用"
                                f"「指定币 + 时间」或「最近一段」查库内数据"}, 502)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:                       # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            if u.path == "/api/shot":
                n = int(self.headers.get("Content-Length", 0) or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                return self._json(self._shot(body))
            return self._json({"error": f"未知路径 {u.path}"}, 404)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:                       # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _shot(self, b: dict):
        e = self.engine
        img_b64 = str(b.get("img_b64") or "")
        if img_b64.startswith("data:"):
            img_b64 = img_b64.split(",", 1)[1]
        if not img_b64:
            raise ValueError("没有收到图片（img_b64 为空）")
        img = base64.b64decode(img_b64)
        for k in ("box", "t0", "p_top", "p_bottom"):
            if k not in b:
                raise ValueError(f"缺少标定参数：{k}")
        box = [float(v) for v in b["box"]]
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("框选区域不对：需要 [x1,y1,x2,y2] 且 x2>x1、y2>y1")
        t0 = int(pd.Timestamp(str(b["t0"]).strip(), tz="UTC").timestamp() * 1000)
        interval = str(b.get("interval", "15m"))
        variant = b.get("variant", "shape")
        if variant not in ("shape", "raw"):
            variant = "shape"
        topk = max(1, min(int(b.get("topk", e.topk)), 120))
        pool_k = int(b.get("pool_k", max(600, topk * 20)))
        contemp = bool(b.get("contemp", False))
        fwd = FWD if b.get("after") else 0

        payload, df = e.shot(img, box, t0, interval, float(b["p_top"]),
                             float(b["p_bottom"]), variant=variant, topk=topk,
                             pool_k=pool_k, allow_contemp=contemp, fwd=fwd,
                             log_scale=bool(b.get("log_scale", False)))

        # 存档：上传图 + 提取的 OHLCV + 标定参数（复核识别误差时要用）
        try:
            outdir = SHOT_DIR / time.strftime("%Y%m%d_%H%M%S")
            outdir.mkdir(parents=True, exist_ok=True)
            ext = "png" if img[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"
            (outdir / f"shot.{ext}").write_bytes(img)
            df.to_csv(outdir / "extracted_15m.csv", index=False)
            (outdir / "calib.json").write_text(json.dumps(
                {k: b.get(k) for k in ("box", "t0", "interval", "p_top",
                                       "p_bottom", "log_scale", "variant",
                                       "topk", "contemp")},
                ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass                                    # 存档失败不影响查询本身
        return payload

    def _search(self, qs):
        e = self.engine
        variant = self._q(qs, "variant", "shape")
        if variant not in ("shape", "raw"):
            variant = "shape"
        topk = int(self._q(qs, "topk", e.topk))
        topk = max(1, min(topk, 120))
        pool_k = int(self._q(qs, "pool_k", max(600, topk * 20)))
        contemp = self._q(qs, "contemp", "0") in ("1", "true", "yes")
        fwd = FWD if self._q(qs, "after", "0") in ("1", "true", "yes") else 0

        note = ""
        if self._q(qs, "idx") is not None:
            i = int(self._q(qs, "idx"))
        elif self._q(qs, "random") is not None:
            i = e.rnd(float(self._q(qs, "amp_min", 2.0)),
                      float(self._q(qs, "amp_max", 60.0)))
            note = "随机逛一段"
        elif self._q(qs, "latest") is not None:
            i = e.latest(self._q(qs, "latest"))
            note = "这只币在库里最近的一段"
        else:
            sym = self._q(qs, "symbol")
            if not sym:
                raise ValueError("要给 symbol+time，或 idx，或 random=1")
            if not self._q(qs, "time"):
                i = e.latest(sym)
                note = "未给时间，取这只币最近的一段"
            else:
                i, exact = e.find(sym, self._q(qs, "time"))
                if not exact:
                    note = "没找到覆盖该时刻的窗口，已取最接近的一段"
        return e.payload(i, variant, topk, pool_k, contemp, fwd, note)

    def _live(self, qs):
        """实时入口：拉到什么就查什么（`after` 开关对它无意义，因为没有未来数据）。"""
        e = self.engine
        sym = self._q(qs, "symbol") or "BTCUSDT"
        variant = self._q(qs, "variant", "shape")
        if variant not in ("shape", "raw"):
            variant = "shape"
        topk = max(1, min(int(self._q(qs, "topk", e.topk)), 120))
        pool_k = int(self._q(qs, "pool_k", max(600, topk * 20)))
        contemp = self._q(qs, "contemp", "0") in ("1", "true", "yes")
        fwd = FWD if self._q(qs, "after", "0") in ("1", "true", "yes") else 0
        return e.live(sym, variant, topk, pool_k, contemp, fwd,
                      note="实时拉取：交易所此刻最近 100 根已收盘的 15 分钟 K 线")


def _probe_existing(host: str, port: int, timeout: float = 1.5):
    """端口上是否已经有一个**活的同款服务**？是则返回它的 /api/info，否则 None。

    ⚠️ **为什么必须探**：`ThreadingHTTPServer` 带 `SO_REUSEADDR`，而 **Windows 的语义是
    「允许两个进程绑同一 host:port」**（Linux 不允许）→ 双击两次启动器就会有**两个服务
    抢着响应**，表现为「同一个接口时好时坏地返回新旧两种格式」，看起来像随机 bug，极难排查。
    （2026-09-16 踩过，当时误判为"端口冲突"）
    """
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/info", timeout=timeout) as r:
            j = json.loads(r.read().decode("utf-8"))
    except Exception:                                       # noqa: BLE001
        return None
    return j if isinstance(j, dict) and "n_windows" in j and "index" in j else None


def _open_browser(url: str) -> None:
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:                                       # noqa: BLE001
        print(f"[serve] 打开浏览器失败，请手动访问 {url}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--index", default=INDEX_NAME)
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--pool-k", dest="pool_k", type=int, default=600)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/"

    # ① 端口上已有活的服务 → 直接复用，绝不起第二个（见 _probe_existing 注释）
    info = _probe_existing(args.host, args.port)
    if info is not None:
        print(f"[serve] 端口 {args.port} 上已有一个在跑的服务"
              f"（索引 {info.get('index')} · {int(info.get('n_windows', 0)):,} 窗口）"
              f"→ 直接复用，不重复启动", flush=True)
        print(f"[serve] 界面地址 → {url}", flush=True)
        if args.open:
            _open_browser(url)
        return 0

    # ② 没有 → 正常启动
    Handler.engine = Engine(args.index, pool_k=args.pool_k, topk=args.topk)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] 界面已就绪 → {url}", flush=True)
    print(f"[serve] 进程 PID {os.getpid()} · 按 Ctrl+C 停止", flush=True)
    if args.open:
        _open_browser(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] 已停止")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
