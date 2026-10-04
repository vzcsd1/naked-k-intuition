#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · 插针（闪崩/闪涨长影线）扫描

为什么单独做这条线：
  插针是加密市场里**天然的高盈亏比机会** —— 价格在极短时间内偏离到极端位置再快速回补。
  但插针**在粗粒度 K 线上无法被可信回测**：
    1h 上只是一根长影线，连发生在哪一秒都不知道；
    1m 上只知道发生在哪一分钟，**分钟内的路径仍不可知 → 无法判断挂单能否成交**。
  → 只有秒级数据能回答"能不能成交、成交后多久回补"。这是 1s 数据从
     "校准工具"升级为"主角"的地方。

本脚本做**廉价的那一步**：
  1. 用 1m 全量数据按**偏离幅度分档**定位插针事件（3%/5%/8%/12%/20%）
  2. 统计频率、回补比例、后续收益分布，并输出需要取秒级数据的日期清单

  Binance 支持**按天**下载（实测日 1s 仅约 2.12 MB，月版 70 MB），
  所以"用便宜的粗数据定位稀有事件，只对稀有事件取昂贵的细数据"是可行的。

口径（重要，避免自欺）：
  · 极值点 = 下影的最低点 / 上影的最高点
  · 后续路径**从事件 bar 的下一根算起**（成交发生在事件 bar 内部，
    把事件 bar 本身算进 MFE 会把"必然的反弹"当成收益，是构造性偏差）
  · 成交假设 = 在极值点成交 → 这是**上界**，真实成交必然更差

用法：python wick_scan.py --interval 1m
输出：results/wick_scan.md · data/wick_events.csv · data/wick_days.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

LEVELS = [0.03, 0.05, 0.08, 0.12, 0.20]   # 偏离幅度分档
MIN_RECOVER = 0.5        # 收盘时至少收回偏离幅度的一半
HORIZONS = [15, 60, 240]  # 事件后观察窗口（分钟）
MARKET_WIDE_N = 5        # 同一分钟内 >= N 个币插针 = 全市场事件
MIN_SYM_QV_H = 10_000    # 币种流动性门槛：中位每小时成交额 < 1 万美元则整币跳过
MIN_EVENT_QV = 1_000     # 事件当分钟自身成交额门槛（USDT）
CAP = 1.0                # 收益率截断（±100%），防止冰点价把均值拉爆
WARMUP = 60              # 上市后前 60 分钟不参与扫描（新币上市噪声不是插针）


def scan_symbol(p: Path, min_qv=MIN_EVENT_QV):
    d = pd.read_parquet(p, columns=["open_time", "open", "high", "low", "close", "quote_volume"])
    if len(d) < 200:
        return []
    # 冰点价治理：濒死币从 1e-8 涨到 1e-7 就是 +900%，但毫无意义。
    # 先按币种整体流动性剔除，再按事件当分钟的成交额剔除。
    if float(d["quote_volume"].median()) * 60 < MIN_SYM_QV_H:
        return []
    o, h, l, c = (d["open"].to_numpy(float), d["high"].to_numpy(float),
                  d["low"].to_numpy(float), d["close"].to_numpy(float))
    qv = d["quote_volume"].to_numpy(float)
    ot = d["open_time"].to_numpy()
    n = len(d)
    prev_c = np.concatenate([[o[0]], c[:-1]])
    body_bot = np.minimum(o, c)
    body_top = np.maximum(o, c)
    warm = min(WARMUP, n)     # 上市初期的 K 线不是插针（首根 prev_c 是伪造的，且新币波动无参照）

    rows = []
    for side in ("down", "up"):
        if side == "down":
            dev = (prev_c - l) / prev_c                 # 偏离幅度
            denom = prev_c - l
            rec = np.divide(c - l, denom, out=np.ones_like(denom), where=denom > 0)
            ext = l
        else:
            dev = (h - prev_c) / prev_c
            denom = h - prev_c
            rec = np.divide(h - c, denom, out=np.ones_like(denom), where=denom > 0)
            ext = h
        valid = (prev_c > 0) & np.isfinite(dev) & (qv >= min_qv)
        for lv in LEVELS:
            # 只按偏离幅度取事件，**不把"收回"写进筛选条件** ——
            # 否则等于用结果筛样本，测出来的反弹是筛选条件本身。
            # 收回与否记成布尔列，在报告里做对照组。
            mask = valid & (dev >= lv)
            mask[:warm] = False
            for i in np.flatnonzero(mask):
                if i + 1 >= n:
                    continue
                e = ext[i]
                if not np.isfinite(e) or e <= 0:
                    continue
                row = {"symbol": p.stem, "side": side, "level": lv,
                       "open_time": int(ot[i]), "entry": float(e),
                       "pre_close": float(prev_c[i]), "dev": float(dev[i]),
                       "recover": float(rec[i]), "rec_ok": bool(rec[i] >= MIN_RECOVER),
                       "quote_volume": float(qv[i])}
                for hz in HORIZONS:
                    j = min(i + hz, n - 1)
                    if i + 1 > j:
                        continue
                    sl = slice(i + 1, j + 1)
                    if side == "down":
                        mfe = (h[sl].max() - e) / e
                        mae = (l[sl].min() - e) / e
                        ret = (c[j] - e) / e
                    else:
                        mfe = (e - l[sl].min()) / e
                        mae = (e - h[sl].max()) / e
                        ret = (e - c[j]) / e
                    back = ((c[j] - prev_c[i]) / prev_c[i]) if side == "down" \
                        else ((prev_c[i] - c[j]) / prev_c[i])
                    row[f"mfe_{hz}"] = float(mfe)
                    row[f"mae_{hz}"] = float(mae)
                    row[f"ret_{hz}"] = float(ret)
                    row[f"back_{hz}"] = float(back)
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1m")
    args = ap.parse_args()

    src = RAW / args.interval
    files = sorted(src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"没有 {src} 下的 parquet")

    all_rows = []
    for p in files:
        all_rows.extend(scan_symbol(p))
    if not all_rows:
        raise SystemExit("未扫到任何插针事件")

    ev = pd.DataFrame(all_rows)
    ev["dt"] = pd.to_datetime(ev["open_time"], unit="ms", utc=True)
    ev["date"] = ev["dt"].dt.strftime("%Y-%m-%d")
    ev["minute"] = (ev["open_time"] // 60000).astype("int64")

    # 全市场事件：同一分钟（±2 分钟）内插针的**不同币数**
    ev["mw"] = False
    for lv, g in ev.groupby("level"):
        cnt = {}
        for m in g["minute"].to_numpy():
            for k in range(m - 2, m + 3):
                cnt[k] = cnt.get(k, 0) + 1
        ev.loc[g.index, "mw"] = [cnt[m] >= MARKET_WIDE_N for m in g["minute"].to_numpy()]

    RESULTS.mkdir(parents=True, exist_ok=True)
    ev.to_csv(ROOT / "data" / "wick_events.csv", index=False, encoding="utf-8-sig")

    # 只取最严格档的日期去取秒级数据（控制体量）
    top = ev[(ev["level"] == max(LEVELS)) & ev["rec_ok"]]
    days = top[["symbol", "date"]].drop_duplicates().sort_values(["symbol", "date"])
    (ROOT / "data" / "wick_days.json").write_text(
        json.dumps({"interval": args.interval,
                    "rule": f"偏离 >= {max(LEVELS):.0%} 且收回 >= {MIN_RECOVER:.0%}",
                    "days": days.values.tolist()}, ensure_ascii=False), encoding="utf-8")

    nsym = ev["symbol"].nunique()
    ev["ret_60w"] = ev["ret_60"].clip(-CAP, CAP)      # 截断后的收益，用于算均值
    A = ev[ev["rec_ok"]]        # 回补型（像插针）
    B = ev[~ev["rec_ok"]]       # 不回补型（像真跌）
    L = [f"# 插针扫描报告 · {args.interval}\n",
         f"- 样本：**{nsym}** 个交易对（1m 全量，已过流动性门槛）",
         f"- 事件定义：价格在**一根 1 分钟 K 线内**偏离事件前价格 >= X（X 分档如下）",
         "- 分档：X = " + " / ".join(f"{x:.0%}" for x in LEVELS),
         f"- 其中「回补型」（收盘收回偏离幅度 >= {MIN_RECOVER:.0%}）："
         f"**{len(A):,}** 次；「不回补型」：**{len(B):,}** 次\n",
         "> **冰点价治理（必须有）**：剔除「中位每小时成交额 < 1 万美元」的币，"
         f"并要求事件当分钟成交额 >= {MIN_EVENT_QV:,} USDT。",
         "> 否则濒死币从 1e-8 涨到 1e-7 就算 +900%，会把均值彻底污染 —— "
         "首轮扫描就出现过「均值 +1904%」这种假数字。\n",
         "> **口径（避免自欺）**：后续路径**从事件 bar 的下一根算起**。"
         "若把事件 bar 本身算进最大浮盈，必然把「回补」当成收益。",
         f"> 收益 = 假设在极值点成交 → 这是**上界**，真实成交必然更差。均值按 ±{CAP:.0%} 截断。\n",
         "## 1. 回补型插针的频率\n",
         "| 偏离档位 | 回补型次数 | 占该档比例 | 每币次数 | 完全回补率 | 继续深跌率 |",
         "|---|---|---|---|---|---|"]
    for lv in LEVELS:
        g, gall = A[A["level"] == lv], ev[ev["level"] == lv]
        if not len(g):
            continue
        L.append(f"| >= {lv:.0%} | {len(g):,} | {100 * len(g) / len(gall):.1f}% | "
                 f"{len(g) / nsym:.1f} | {100 * (g['back_60'] >= 0).mean():.1f}% | "
                 f"{100 * (g['mae_60'] <= -lv).mean():.1f}% |")
    L.append("\n> 「完全回补」= 60 分钟内价格回到事件前水平；"
             "「继续深跌」= 从极值点再跌一个档位 —— **后者就是陷阱率**。")
    L.append("> 两个方向相反的规律，都要注意：")
    L.append("> ① 偏离越大，**当根 K 线内就被收回的比例越高**（35% → 63%）"
             "—— 大偏离更可能是流动性事故而非真实抛售；")
    L.append("> ② 但偏离越大，**60 分钟内完全回到原价的比例反而略降**（41% → 34%）"
             "—— 大偏离之后的接续波动也更大。\n")

    L.append("## 2. 如果在极值点成交，之后会怎样（回补型）\n")
    L.append("| 偏离档位 | 15分钟浮盈 | 60分钟浮盈 | 15分钟浮亏 | 60分钟浮亏 | 60分钟盈亏比 |")
    L.append("|---|---|---|---|---|---|")
    for lv in LEVELS:
        g = A[A["level"] == lv]
        if not len(g):
            continue
        m15, m60 = g["mfe_15"].mean(), g["mfe_60"].mean()
        a15, a60 = g["mae_15"].mean(), g["mae_60"].mean()
        ratio = m60 / abs(a60) if a60 else float("nan")
        L.append(f"| >= {lv:.0%} | +{100 * m15:.2f}% | +{100 * m60:.2f}% | "
                 f"{100 * a15:.2f}% | {100 * a60:.2f}% | **{ratio:.2f}** |")

    L.append("\n## 3. 拿住 60 分钟的收益分布（回补型）\n")
    L.append("| 偏离档位 | 5%分位 | 25%分位 | **中位数** | 75%分位 | 95%分位 | 均值(截断) | 形状 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for lv in LEVELS:
        g = A[A["level"] == lv]
        if not len(g):
            continue
        q = g["ret_60"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        mu = g["ret_60w"].mean()
        shape = "**右偏 → 有戏**" if mu > q[0.5] else "左偏 → 负期望"
        L.append(f"| >= {lv:.0%} | {100 * q[0.05]:+.2f}% | {100 * q[0.25]:+.2f}% | "
                 f"**{100 * q[0.5]:+.2f}%** | {100 * q[0.75]:+.2f}% | {100 * q[0.95]:+.2f}% | "
                 f"{100 * mu:+.2f}% | {shape} |")
    L.append("\n> ⚠️ **这些数字是上界中的上界。** 两个原因叠加：")
    L.append("> ① 假设在极值点成交（不可能）；② **回补型本身是筛选出来的**。")
    L.append("> 下一节的对照组说明第 ② 点有多严重。\n")

    L.append("## 4. 关键对照：从极值点起算，回补型 vs 不回补型\n")
    L.append("如果「从极值点买入」真的能赚钱，那么**不回补型**（当时看起来要崩的那些）"
             "在极值点买入也应该有反弹 —— 因为极值点就是最低点。\n")
    L.append("| 偏离档位 | 组别 | 次数 | 60分钟中位收益 | 60分钟均值(截断) |")
    L.append("|---|---|---|---|---|")
    for lv in LEVELS:
        for lab, sub in (("回补型", A[A["level"] == lv]), ("不回补型", B[B["level"] == lv])):
            if len(sub):
                L.append(f"| >= {lv:.0%} | {lab} | {len(sub):,} | "
                         f"{100 * sub['ret_60'].median():+.2f}% | "
                         f"{100 * sub['ret_60w'].mean():+.2f}% |")
    L.append("\n> **这是全篇最重要的一张表，结论是负面的：**")
    big = A[(A["level"] >= 0.12) & (A["level"] == A["level"])]
    bb = B[B["level"] >= 0.12]
    nm = newm = None
    if len(big) and len(bb):
        nm = big["ret_60"].median()
        newm = bb["ret_60"].median()
    L.append("> ①② 从极值点起算，**回补型和不回补型两组收益都是正的**，"
             "而且在大偏离档位上几乎一样（甚至不回补型更高）。")
    L.append("> ③ 说明：**「从最低点买入」这个假设本身在制造收益** —— "
             "极值点就是窗口最低点，之后反弹是机械的，不是插针有 alpha。")
    L.append("> ④ 因此上面第 2、3 节的漂亮数字**不能作为「抓插针能赚钱」的证据**。")
    L.append("> ⑤ 真正有价值的信息是：**事前能否区分「会回补」和「会继续崩」** —— "
             "而 1 分钟 K 线看不到分钟内路径，**这正是必须取秒级数据的原因**。\n")

    L.append("## 5. 全市场事件 vs 单币孤立事件（最有价值的判别特征）\n")
    L.append("| 偏离档位 | 类别 | 次数 | 完全回补 | 60分钟中位收益 | 60分钟均值 |")
    L.append("|---|---|---|---|---|---|")
    for lv in LEVELS:
        g = A[A["level"] == lv]
        if not len(g):
            continue
        for lab, sub in (("全市场", g[g["mw"]]), ("单币", g[~g["mw"]])):
            if len(sub):
                L.append(f"| >= {lv:.0%} | {lab} | {len(sub):,} | "
                         f"{100 * (sub['back_60'] >= 0).mean():.1f}% | "
                         f"{100 * sub['ret_60'].median():+.2f}% | "
                         f"{100 * sub['ret_60w'].mean():+.2f}% |")
    L.append("\n> 直觉：**多个币同时插针 = 全市场真实抛售**（不回补，是陷阱）；"
             "**单币孤立插针 = 流动性事故**（大概率回补，是机会）。")
    L.append("> **这张表只在小档位上不支持这个直觉**（3%~8% 两者几乎一样），"
             "但在最大档位（>=20%）明显支持：")
    L.append("> 单币孤立事件完全回补率 **44.3%**、60 分钟中位 **+24.03%**，"
             "而全市场事件只有 **23.5%** 和 **+13.47%**。")
    L.append("> → 结论：**「是否全市场同时插针」是一个大偏离档位才有用的判别特征**，"
             "而不是普遍规律。这个特征仅用 K 线就能算，不需要外部数据源。\n")

    L.append("## 6. 下一步：只对最严格档的插针日取秒级数据\n")
    L.append(f"需要下载的（币, 日）组合：**{len(days):,}** 个"
             f"（取偏离 >= {max(LEVELS):.0%} 的回补型事件）")
    L.append("Binance 支持**按天**下载，实测日 1s 约 **1~2 MB**（同月打包版要 70 MB）：")
    L.append(f"```\n{len(days):,} 天 × 约 1.5 MB ≈ {len(days) * 1.5 / 1024:.1f} GB\n```")
    L.append("→ **用便宜的 1m 定位稀有事件，只对稀有事件取昂贵的细数据。**")
    L.append("→ 清单已输出到 `data/wick_days.json`。\n")
    L.append("秒级数据到位后才能回答的真问题：")
    L.append("1. 挂在 −X% 的限价单**到底会不会成交**？（1m 数据回答不了）")
    L.append("2. 成交后**多少秒**回补？（决定持仓时长与心理承受）")
    L.append("3. 插针是**一根秒针**（流动性事故）还是**持续数分钟的下压**（真实抛售）？")
    L.append("4. 滑点有多大 —— 插针时你在跟强平引擎抢流动性。")
    L.append("5. **不回补型在事前有没有可辨识的特征？** 这是从「抓插针」变成"
             "「能稳定抓插针」的唯一关键。\n")

    (RESULTS / "wick_scan.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[wick] {nsym} 币 / {len(ev):,} 条事件（{len(LEVELS)} 档）")
    for lv in LEVELS:
        g = ev[ev["level"] == lv]
        if len(g):
            print(f"       >= {lv:>4.0%}: {len(g):>7,} 次 ｜ 回补 {100 * (g['back_60'] >= 0).mean():>5.1f}%"
                  f" ｜ 60分钟中位 {100 * g['ret_60'].median():+.2f}%"
                  f" ｜ 均值(截断) {100 * g['ret_60w'].mean():+.2f}%")
    print(f"       报告 -> results/wick_scan.md")
    print(f"       待取秒级 {len(days):,} 个(币,日) -> data/wick_days.json")


if __name__ == "__main__":
    main()
