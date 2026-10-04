# -*- coding: utf-8 -*-
"""M1 入口 B：TradingView 截图 -> OHLCV CSV（颜色掩码，不用任何模型）。

用法（手工标定 6 个数，比全自动识别可靠）：
  python m1_screenshot_ohlcv.py IMG.png \
      --box x1,y1,x2,y2 --t0 "2024-01-01 00:00" --interval 15m \
      --p-top 65000 --p-bottom 60000 \
      [--t1 "2024-01-01 23:45"] [--out out.csv] [--selftest]

原理（10 号文档第三节）：TV 蜡烛是纯色填充。
  每根蜡烛 = 若干"有色列"；列内彩色像素最高/最低点 -> high/low；
  实体 = 该行彩色横向连续宽度 >= 55% 蜡烛宽度的行段 -> body 顶/底；
  绿色: close=open上沿 open=下沿；红色相反。
  时间轴：由 --t0 按周期递推（不 OCR 文字）。
输出 CSV 列：open_time,open,high,low,close,volume（volume=量柱像素高度，检索只用量比，够用）。
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
               "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

L_RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results")


def color_masks(arr):
    """arr HxWx3 uint8 -> (green_mask, red_mask) 布尔。对主题色不敏感的通道占优判据。"""
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    sat = arr.max(axis=2).astype(np.int16) - arr.min(axis=2).astype(np.int16)
    # TV 默认绿 #26a69a=(38,166,154)：绿蓝几乎相等，不能用 g>b 判绿；
    # 判据 = 绿显著高于红，且蓝没有显著高于绿（排除青色背景）
    green = (g > r + 18) & (g + 18 > b) & (sat > 30)
    red = (r > g + 18) & (r > b + 18) & (sat > 30)
    return green, red


def extract_image(im, box, p_top, p_bottom, t0_ms, interval_ms, t1_ms=None,
                  log_scale=False):
    """im: HxWx3 uint8 数组。box=(x1,y1,x2,y2) 像素坐标（绘图区，只框 K 线，不含量柱）。
    返回 DataFrame(open_time, open, high, low, close, volume=1, n_bars=15)。"""
    y1, y2 = sorted((int(box[1]), int(box[3])))
    x1, x2 = sorted((int(box[0]), int(box[2])))
    sub = im[y1:y2 + 1, x1:x2 + 1]
    H1 = sub.shape[0] - 1
    green, red = color_masks(sub)
    color = green | red
    if log_scale:
        lt, lb = np.log(float(p_top)), np.log(float(p_bottom))
        price_of = lambda r: np.exp(lt - np.asarray(r, np.float64) * (lt - lb) / H1)  # noqa: E731
    else:
        price_of = lambda r: p_top - np.asarray(r, np.float64) * (p_top - p_bottom) / H1  # noqa: E731

    col_has = color.any(axis=0)
    groups = []
    i = 0
    Wd = len(col_has)
    while i < Wd:
        if col_has[i]:
            j = i
            while j + 1 < Wd and col_has[j + 1]:
                j += 1
            groups.append((i, j))
            i = j + 1
        else:
            i += 1
    if len(groups) < 10:
        raise ValueError(f"只识别到 {len(groups)} 根蜡烛列组，检查框选区域是否框住 K 线绘图区")
    widths = np.array([b - a + 1 for a, b in groups])
    med_w = int(np.median(widths))
    rows = []
    for gi, (a, b) in enumerate(groups):
        cols = slice(a, b + 1)
        cmask = color[:, cols]
        gmask = green[:, cols]
        rows_col = np.where(cmask.any(axis=1))[0]
        if len(rows_col) == 0:
            continue
        hi_y, lo_y = rows_col[0], rows_col[-1]
        high = float(price_of(hi_y))
        low = float(price_of(lo_y))
        # 实体行：横向连续宽度 >= 55% 中位蜡烛宽
        w_required = max(2, int(0.55 * med_w))
        full_rows = np.where(cmask.sum(axis=1) >= w_required)[0]
        if len(full_rows):
            by1, by2 = full_rows[0], full_rows[-1]
            body_top = float(price_of(by1))
            body_bot = float(price_of(by2))
            is_green = gmask.sum() >= cmask.sum() / 2
            o, c = (body_bot, body_top) if is_green else (body_top, body_bot)
        else:
            mid = float(price_of((hi_y + lo_y) / 2))
            o = c = mid  # 十字星
        ot = t0_ms + gi * interval_ms
        rows.append((ot, o, high, low, c))
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close"])
    # 量柱区：本工具要求 --box 只框 K 线区（含量柱会把量柱误当蜡烛）。
    # v1 不提取量能 -> 填中性值 1.0（检索时 log(vol/median)=0，量能通道不起作用）
    df["volume"] = 1.0
    df["n_bars"] = 15
    if t1_ms is not None:
        expect = int((t1_ms - t0_ms) // interval_ms) + 1
        if len(df) != expect:
            print(f"[warn] 识别 {len(df)} 根，标定暗示 {expect} 根（--t0/--t1 不符），以识别为准")
    return df


def extract(img_path, box, p_top, p_bottom, t0_ms, interval_ms, t1_ms=None,
            log_scale=False):
    im = np.asarray(Image.open(img_path).convert("RGB"))
    return extract_image(im, box, p_top, p_bottom, t0_ms, interval_ms, t1_ms,
                         log_scale)


def extract_bytes(data, box, p_top, p_bottom, t0_ms, interval_ms, t1_ms=None,
                  log_scale=False):
    """服务端入口：图片字节（PNG/JPG）-> OHLCV DataFrame。"""
    import io as _io
    im = np.asarray(Image.open(_io.BytesIO(data)).convert("RGB"))
    return extract_image(im, box, p_top, p_bottom, t0_ms, interval_ms, t1_ms,
                         log_scale)


def make_selftest(path):
    """生成合成 TV 风格截图，供 --selftest 回环验证。"""
    rng = np.random.default_rng(7)
    n = 60
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) + np.abs(rng.normal(0, 0.4, n))
    lo = np.minimum(op, close) - np.abs(rng.normal(0, 0.4, n))
    W, H, cw = 960, 480, 12
    im = np.full((H, W, 3), 255, np.uint8)
    p_top, p_bottom = hi.max() + 1, lo.min() - 1
    y_of = lambda p: int((p_top - p) / (p_top - p_bottom) * (H - 1))  # noqa: E731
    for i in range(n):
        x0 = 10 + i * cw
        col = (38, 166, 154) if close[i] >= op[i] else (239, 83, 80)
        for x in range(x0 + 5, x0 + 9):  # 影线宽 4px
            im[y_of(hi[i]):y_of(lo[i]) + 1, x] = col
        by1, by2 = sorted((y_of(op[i]), y_of(close[i])))
        for x in range(x0 + 1, x0 + 11):  # 实体宽 10px
            im[by1:by2 + 1, x] = col
    Image.fromarray(im).save(path)
    return dict(n=n, p_top=float(p_top), p_bottom=float(p_bottom),
                t0="2024-01-01 00:00", interval="15m",
                truth=dict(open=op, high=hi, low=lo, close=close))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="截图路径（--selftest 时忽略）")
    ap.add_argument("--box", help="绘图区 x1,y1,x2,y2（像素，左上为 0,0）")
    ap.add_argument("--t0", help='首根开盘时间 "YYYY-MM-DD HH:MM" (UTC)')
    ap.add_argument("--t1", default=None, help="末根时间（仅用于一致性告警）")
    ap.add_argument("--interval", default="15m", choices=list(INTERVAL_MS))
    ap.add_argument("--p-top", type=float, default=None, help="绘图区顶部价格")
    ap.add_argument("--p-bottom", type=float, default=None, help="绘图区底部价格")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        from datetime import datetime, timezone
        tmp = os.path.join(L_RES, "m1_screenshot_selftest.png")
        meta = make_selftest(tmp)
        box = (0, 0, 959, 479)
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        df = extract(tmp, box, meta["p_top"], meta["p_bottom"],
                     int(t0.timestamp() * 1000), INTERVAL_MS[meta["interval"]])
        tr = meta["truth"]
        err = dict(
            open=float(np.abs(df["open"] - tr["open"]).max()),
            high=float(np.abs(df["high"] - tr["high"]).max()),
            low=float(np.abs(df["low"] - tr["low"]).max()),
            close=float(np.abs(df["close"] - tr["close"]).max()),
        )
        rng_price = meta["p_top"] - meta["p_bottom"]
        print(f"selftest: {len(df)}/{meta['n']} bars, max abs err = {err}")
        print(f"  (图高 {rng_price:.1f} 价格单位 / 479px -> 每像素 {rng_price/479:.4f})")
        return

    if not args.image or not args.box or not args.t0 or args.p_top is None or args.p_bottom is None:
        raise SystemExit("需要 image + --box + --t0 + --p-top + --p-bottom（或 --selftest）")
    box = tuple(float(v) if "." in v else int(v) for v in args.box.split(","))
    from datetime import datetime, timezone
    t0 = int(datetime.strptime(args.t0, "%Y-%m-%d %H:%M")
             .replace(tzinfo=timezone.utc).timestamp() * 1000)
    t1 = (int(datetime.strptime(args.t1, "%Y-%m-%d %H:%M")
              .replace(tzinfo=timezone.utc).timestamp() * 1000)
          if args.t1 else None)
    df = extract(args.image, box, args.p_top, args.p_bottom, t0,
                 INTERVAL_MS[args.interval], t1)
    out = args.out or os.path.splitext(args.image)[0] + "_ohlcv.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} bars -> {out}")
    print("下一步: python m1_query.py --csv " + out)


if __name__ == "__main__":
    main()
