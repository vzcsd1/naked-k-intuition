#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时行情拉取 —— 给联想界面用：直接取「交易所此刻的最近 100 根 K 线」。

⚠️ 为什么用 `data-api.binance.vision` 而不是 `api.binance.com`：
    2026-09-17 在本机实测（`probe_binance.py`）：
      api.binance.com / api1 / api2 / api.binance.us  **全部超时或握手失败**
      data-api.binance.vision                        **通，1.2 秒返回真实 K 线**
    它是币安官方公开数据端点（为 data.binance.vision 归档站提供），
    **无需 API Key、返回结构与主站一致**，因此作为默认数据源。

两条必须守住的规矩：
  1. **丢掉最后一根未收盘的 K 线** —— 半根 K 线的形态是"没走完的形状"，
     和高低开收都定了的历史窗口不可比，会污染检索。
  2. **用交易所服务器时间判断"收盘了没"**，不用本机时钟（本机时钟快了/慢了都会误判）。

CLI（用于人工核对）：
    python scripts/live_feed.py BTCUSDT
    python scripts/live_feed.py ETHUSDT --interval 15m --n 5
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

BASE = "https://data-api.binance.vision"
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
_UA = {"User-Agent": "Mozilla/5.0 (luoK-retrieve)"}

_server_time = {"t": 0.0, "v": 0}      # 交易所服务器时间缓存（60 秒）


class LiveError(RuntimeError):
    """实时拉取失败（网络不通 / 币种不存在 / 返回异常）。"""


def _get_json(url: str, timeout: float = 10.0):
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # 币安的错误信息在 body 里（如 -1121 Invalid symbol）
        try:
            j = json.loads(e.read().decode("utf-8"))
            raise LiveError(f"交易所返回 {e.code}：{j.get('msg', j)}") from None
        except (ValueError, AttributeError):
            raise LiveError(f"交易所返回 HTTP {e.code}") from None
    except Exception as e:                                   # noqa: BLE001
        raise LiveError(f"连不上行情接口（{type(e).__name__}: {e}）") from None
    if isinstance(data, dict) and "code" in data and int(data.get("code", 0)) != 0:
        raise LiveError(f"交易所返回错误 {data.get('code')}：{data.get('msg')}")
    return data


def server_now_ms(timeout: float = 8.0) -> int:
    """交易所服务器当前时间（毫秒）。带 60 秒缓存；取不到就退回本机时钟。"""
    if time.time() - _server_time["t"] < 60 and _server_time["v"]:
        return int(_server_time["v"])
    try:
        j = _get_json(f"{BASE}/api/v3/time", timeout=timeout)
        v = int(j["serverTime"])
        _server_time.update(t=time.time(), v=v)
        return v
    except Exception:                                        # noqa: BLE001
        return int(time.time() * 1000)


def fetch_recent(symbol: str, interval: str = "15m", n: int = 100,
                 drop_unclosed: bool = True, timeout: float = 10.0) -> dict:
    """取最近 `n` 根**已收盘**的 K 线。

    返回 dict：
      ot  开盘时间(ms, int64)   o/h/l/cl  开高低收(float64)   qv  计价成交额(float64)
      fetched_at / last_closed / n_bars / source / interval / symbol
    """
    symbol = symbol.upper().strip()
    if interval not in INTERVAL_MS:
        raise LiveError(f"不支持的周期 {interval}")
    # 多要 2 根：一根可能是未收盘的，另一根防边界
    need = int(n) + 2
    url = (f"{BASE}/api/v3/klines?symbol={urllib.parse.quote(symbol)}"
           f"&interval={interval}&limit={min(1000, need)}")
    raw = _get_json(url, timeout=timeout)
    if not isinstance(raw, list) or not raw:
        raise LiveError(f"{symbol} 返回空数据（币种可能不存在）")

    ot = np.array([int(r[0]) for r in raw], dtype=np.int64)
    o = np.array([float(r[1]) for r in raw], dtype=np.float64)
    hi = np.array([float(r[2]) for r in raw], dtype=np.float64)
    lo = np.array([float(r[3]) for r in raw], dtype=np.float64)
    cl = np.array([float(r[4]) for r in raw], dtype=np.float64)
    vol = np.array([float(r[5]) for r in raw], dtype=np.float64)
    ct = np.array([int(r[6]) for r in raw], dtype=np.int64)
    qv = np.array([float(r[7]) for r in raw], dtype=np.float64)

    now = server_now_ms()
    if drop_unclosed:
        closed = ct < now
        if not closed.all():
            ot, o, hi, lo, cl, vol, ct, qv = (a[closed] for a in (ot, o, hi, lo, cl, vol, ct, qv))

    if len(ot) < n:
        raise LiveError(f"{symbol} 只取到 {len(ot)} 根已收盘 K 线（需要 {n} 根）")
    ot, o, hi, lo, cl, vol, ct, qv = (a[-n:] for a in (ot, o, hi, lo, cl, vol, ct, qv))

    out = {"symbol": symbol, "interval": interval, "n_bars": int(len(ot)),
           "ot": ot, "o": o, "h": hi, "l": lo, "cl": cl, "qv": qv,
           "fetched_at": int(time.time() * 1000),
           "server_now": int(now),
           "last_closed": int(ct[-1]),
           "last_open": int(ot[-1]),
           "source": BASE}
    out["warnings"] = validate(out)
    return out


def validate(bars: dict) -> list:
    """自洽性检查 —— 这类错误**不会报错，只会静默给出错数据**，所以必须查。

    最典型的是"字段顺序理解错"：币安 klines 是一个位置数组，
    把 index 5（volume，成交量）当成 index 7（quoteAssetVolume，计价成交额），
    代码照样跑、检索照样出结果，只是**成交量通道全错**。
    """
    issues = []
    o, h, l, c = bars["o"], bars["h"], bars["l"], bars["cl"]
    qv, ot = bars["qv"], bars["ot"]
    if np.any(~np.isfinite(c)) or np.any(c <= 0):
        issues.append("存在非正或非有限的收盘价")
    bad = (h < np.maximum(o, c) - 1e-9) | (l > np.minimum(o, c) + 1e-9)
    if bool(bad.any()):
        issues.append(f"{int(bad.sum())} 根不满足 low≤min(开,收)≤max(开,收)≤high")
    if bool((qv < 0).any()):
        issues.append("成交额出现负数")
    step = INTERVAL_MS.get(bars["interval"], 0)
    if step and len(ot) > 1 and bool((np.diff(ot) != step).any()):
        issues.append(f"{int((np.diff(ot) != step).sum())} 处时间不连续（有缺口）")
    return issues


def amp_and_vol(cl: np.ndarray, hi: np.ndarray, lo: np.ndarray) -> tuple[float, float]:
    """窗口总波幅% 与平均单根振幅% —— 与 build_index.py 的元数据口径一致。"""
    a = (float(np.exp(np.log(hi).max() - np.log(lo).min())) - 1.0) * 100.0
    v = ((hi - lo) / np.where(cl > 0, cl, np.nan))
    v = np.nan_to_num(v, nan=0.0, posinf=0.0).mean() * 100.0
    return round(a, 2), round(float(v), 3)


def main():
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    interval = "15m"
    n = 5
    for k, a in enumerate(sys.argv):
        if a == "--interval" and k + 1 < len(sys.argv):
            interval = sys.argv[k + 1]
        if a == "--n" and k + 1 < len(sys.argv):
            n = int(sys.argv[k + 1])
    try:
        d = fetch_recent(sym, interval, n=100)
    except LiveError as e:
        print(f"❌ {e}")
        return 1
    lag = (d["server_now"] - d["last_closed"]) / 1000
    print(f"✅ {d['symbol']} {interval} · 取到 {d['n_bars']} 根已收盘 K 线 · 源 {d['source']}")
    print(f"   最后一根已收盘 {np.datetime64(d['last_open'], 'ms')} "
          f"（距今 {lag:.0f} 秒）")
    amp, volp = amp_and_vol(d["cl"], d["h"], d["l"])
    print(f"   这 100 根：总波幅 {amp}% ｜ 单根平均振幅 {volp}%")
    print("-" * 70)
    print(f"{'开盘时间(UTC)':<20}{'开':>12}{'高':>12}{'低':>12}{'收':>12}{'成交额':>14}")
    for k in range(max(0, len(d["ot"]) - n), len(d["ot"])):
        t = str(np.datetime64(int(d["ot"][k]), "ms"))
        print(f"{t:<20}{d['o'][k]:>12.2f}{d['h'][k]:>12.2f}{d['l'][k]:>12.2f}"
              f"{d['cl'][k]:>12.2f}{d['qv'][k]:>14,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
