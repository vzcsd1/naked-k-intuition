# -*- coding: utf-8 -*-
"""截图通道端到端回归：真库窗口 -> 渲染 TV 风格截图 -> POST /api/shot -> 应命中自身。

用法：
  1) 先起服务：  python scripts/serve.py           （或双击 启动界面.bat）
  2) 再跑本测：  python scripts/m1_shot_e2e_test.py
通过标准：
  · contemp=1 时 top-1 = 库内同一窗口（距离 sqrt 后 < 2.0，远小于随机标尺 ~10.8）
  · contemp=0 时结果不含与查询时段重叠的窗口（排除规则生效）
  · 对数轴开关下同样命中
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import serve  # noqa: E402  (借它的常量与引擎加载；不会起服务)

BASE = "http://127.0.0.1:8765"
WIN = 100
COL_UP, COL_DN = (38, 166, 154), (239, 83, 80)


def render_tv_shot(bars, p_top, p_bottom, W=1500, H=760, log_scale=False):
    """bars: [(o,h,l,c,qv)] 100 根 -> TV 风格 PNG 字节（实体 8px、影线 2px、间隔 14px）。"""
    im = np.full((H, W, 3), 255, np.uint8)
    ytop, ybot = 20, H - 20
    if log_scale:
        lt, lb = np.log(p_top), np.log(p_bottom)
        y_of = lambda p: int(ytop + (lt - np.log(p)) / (lt - lb) * (ybot - ytop))  # noqa: E731
    else:
        y_of = lambda p: int(ytop + (p_top - p) / (p_top - p_bottom) * (ybot - ytop))  # noqa: E731
    x0 = 20
    for k, (o, h, l, c, _qv) in enumerate(bars):
        col = COL_UP if c >= o else COL_DN
        xa = x0 + k * 14
        for x in range(xa + 3, xa + 5):                     # 影线 2px
            im[y_of(h):y_of(l) + 1, x] = col
        b1, b2 = sorted((y_of(o), y_of(c)))
        b2 = max(b2, b1 + 1)                                # 十字星也给 1px 实体
        for x in range(xa, xa + 8):                         # 实体 8px
            im[b1:b2 + 1, x] = col
    buf = Path("_shot_bytes.png")
    Image.fromarray(im).save(buf)          # 沙箱内不删文件：固定名覆盖写
    return buf.read_bytes()


def pick_window():
    m = pd.read_parquet(ROOT / "data" / "index" / serve.INDEX_NAME / "meta.parquet")
    m = m.reset_index(drop=True)
    sel = m[(m["symbol"].astype(str) == "BTCUSDT") & (m["t0"] >= pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000)]
    sel = sel[sel["amp_pct"].between(4, 9)].sort_values("t0")
    i = int(sel.index[len(sel) // 2])
    return i, m


def bars_of(i, m):
    d = pd.read_parquet(ROOT / "data" / "raw" / "15m" / "BTCUSDT.parquet",
                        columns=["open_time", "open", "high", "low", "close", "quote_volume"])
    ot = d["open_time"].to_numpy(np.int64)
    p = int(np.searchsorted(ot, int(m["t0"].iloc[i]), side="left"))
    w = d.iloc[p:p + WIN]
    assert len(w) == WIN and int(w["open_time"].iloc[0]) == int(m["t0"].iloc[i])
    return [(float(r.open), float(r.high), float(r.low), float(r.close),
             float(r.quote_volume)) for r in w.itertuples(index=False)]


def api_shot(img_bytes, t0, p_top, p_bottom, contemp, log_scale):
    import base64
    body = json.dumps(dict(
        img_b64="data:image/png;base64," + base64.b64encode(img_bytes).decode(),
        box=[0, 0, 1499, 759], t0=t0, interval="15m",
        p_top=p_top, p_bottom=p_bottom, log_scale=log_scale,
        variant="shape", topk=5, contemp=contemp)).encode()
    req = urllib.request.Request(BASE + "/api/shot", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main():
    i, m = pick_window()
    t0 = int(m["t0"].iloc[i])
    bars = bars_of(i, m)
    hi = max(b[1] for b in bars); lo = min(b[2] for b in bars)
    p_top, p_bottom = hi * 1.002, lo * 0.998
    t0_str = pd.to_datetime(t0, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")
    print(f"测试窗口: idx={i} BTCUSDT {t0_str} 波幅={float(m['amp_pct'].iloc[i]):.2f}%")

    ok = True
    for log_scale in (False, True):
        img = render_tv_shot(bars, p_top, p_bottom, log_scale=log_scale)
        j = api_shot(img, t0_str, p_top, p_bottom, contemp=True, log_scale=log_scale)
        if "error" in j:
            print(f"[FAIL log={log_scale}] 服务端报错: {j['error']}"); ok = False; continue
        r0 = j["results"][0]
        d0 = float(np.sqrt(r0["dist"]))
        hit = (r0["idx"] == i and d0 < 2.0)
        print(f"[{'PASS' if hit else 'FAIL'} log={log_scale}] top1 idx={r0['idx']} "
              f"(期望 {i}) dist={d0:.3f} 标尺={j['dist_random']} "
              f"识别根数={j['shot']['n_bars']}")
        ok &= hit

    img = render_tv_shot(bars, p_top, p_bottom)
    j = api_shot(img, t0_str, p_top, p_bottom, contemp=False, log_scale=False)
    overlap = [r for r in j["results"]
               if int(r["t0"]) <= t0 + 99 * 900_000 and int(r["t1"]) >= t0]
    no_overlap = len(overlap) == 0
    print(f"[{'PASS' if no_overlap else 'FAIL'}] contemp=0 排除规则："
          f"{len(j['results'])} 个结果里与查询时段重叠 {len(overlap)} 个")
    ok &= no_overlap

    print("=> 截图通道端到端：" + ("全部通过" if ok else "存在失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
