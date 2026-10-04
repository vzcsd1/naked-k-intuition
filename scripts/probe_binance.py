#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
探测本机能否直连币安行情接口 —— 「界面接实时拉取」的前置条件。

背景：沙箱内 api.binance.com 曾不可达，需要在**用户本机真实环境**下实测，
      而不是靠猜。本脚本把候选端点逐个试一遍，报告哪个通、耗时多少。

用法：
    python scripts/probe_binance.py
"""
import json
import ssl
import sys
import time
import urllib.error
import urllib.request

# 候选端点：主站 / 公开数据端点（无需 Key、通常不受地域限制）/ 备用站
ENDPOINTS = [
    ("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=3", "api.binance.com (主站)"),
    ("https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=3", "data-api.binance.vision (公开数据)"),
    ("https://api1.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=3", "api1.binance.com (备用1)"),
    ("https://api2.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=3", "api2.binance.com (备用2)"),
    ("https://api.binance.us/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=3", "api.binance.us (美国站，仅供参考)"),
]


def probe(url, timeout=8):
    """返回 (ok, elapsed_ms, detail)。"""
    ctx = ssl.create_default_context()
    # 部分环境证书链不全 → 允许降级重试一次（仅探测用，不用于取数）
    for verify in (True, False):
        if not verify:
            ctx = ssl._create_unverified_context()
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (probe)"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                raw = r.read().decode("utf-8", "replace")
            ms = int((time.time() - t0) * 1000)
            try:
                data = json.loads(raw)
            except Exception:
                return False, ms, "返回非 JSON：" + raw[:120]
            if isinstance(data, list) and data and isinstance(data[0], list):
                bar = data[-1]
                note = "K线数=%d 末根开盘=%s 收=%s" % (len(data), bar[0], bar[4])
                if not verify:
                    note += " [注意：跳过证书校验才通]"
                return True, ms, note
            if isinstance(data, dict) and "code" in data:
                return False, ms, "接口返回错误码 %s: %s" % (data.get("code"), data.get("msg"))
            return False, ms, "返回结构异常：" + raw[:120]
        except urllib.error.HTTPError as e:
            ms = int((time.time() - t0) * 1000)
            return False, ms, "HTTP %s %s" % (e.code, e.reason)
        except Exception as e:
            ms = int((time.time() - t0) * 1000)
            last = "%s: %s" % (type(e).__name__, e)
            continue
    return False, ms, last


def main():
    print("=" * 72)
    print("币安行情接口连通性探测（本机真实网络）")
    print("=" * 72)
    ok_list = []
    for url, label in ENDPOINTS:
        ok, ms, detail = probe(url)
        flag = "✅ 通" if ok else "❌ 不通"
        print("%s  %-34s %5d ms  %s" % (flag, label, ms, detail))
        if ok:
            ok_list.append((label, url, ms))
    print("-" * 72)
    if ok_list:
        label, url, ms = min(ok_list, key=lambda x: x[2])
        print("结论：可用端点 %d 个，最快 = %s（%d ms）" % (len(ok_list), label, ms))
        print("实时拉取建议使用：" + url.split("?")[0])
    else:
        print("结论：**全部不通** → 本机也无法直连币安 REST。")
        print("      需要替代方案：走代理 / 用 data.binance.vision 的每日归档（离线）。")
    return 0 if ok_list else 1


if __name__ == "__main__":
    sys.exit(main())
