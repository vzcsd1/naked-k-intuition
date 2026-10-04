#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · 数据体检（M0 收尾）

回答一个问题：这批历史 K 线能不能信。
检查项（每一项都对应 04 文档里的一个真实陷阱）：
  1. 时间缺口   —— Binance 维护/停摆造成的空档；回测里会被当成"连续持有"
  2. 僵尸尾部   —— 已下架交易对但数据继续发布（FTT 这类），回测会照着死盘下单
  3. 零成交量   —— 有价无市，无法成交
  4. 价格僵直   —— 连续多根 close 完全相同，说明没有真实撮合
  5. 可见区间   —— 输出 point-in-time 元数据，供回测按区间过滤

用法：python data_audit.py --interval 1h
输出：results/data_audit_1h.md（摘要）+ results/data_audit_1h.csv（明细）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
HOUR_MS = 3_600_000


def longest_run(mask: pd.Series) -> int:
    if len(mask) == 0:
        return 0
    g = (mask != mask.shift()).cumsum()
    return int(mask.groupby(g).sum().max()) if mask.any() else 0


def audit_symbol(p: Path) -> dict | None:
    d = pd.read_parquet(p, columns=["open_time", "close", "high", "low", "volume", "quote_volume"])
    if d.empty:
        return None
    ot = d["open_time"].to_numpy()
    gap = pd.Series(ot).diff().dropna()
    step = int(gap.mode().iloc[0]) if len(gap) else HOUR_MS
    expected = int((ot[-1] - ot[0]) / step) + 1
    missing = expected - len(ot)
    big_gaps = gap[gap > step]

    zero_vol = d["volume"].fillna(0) <= 0
    flat = (d["high"] == d["low"])

    # 尾部：最后一次"有量且价格有波动"的位置
    live = (~zero_vol) & (~flat)
    last_live = int(ot[live.to_numpy().nonzero()[0][-1]]) if live.any() else int(ot[0])
    zombie = int((ot > last_live).sum())

    # 最大缺口的位置（可用于判断停牌 vs 抓取错误）
    if len(big_gaps):
        j = int(big_gaps.idxmax())
        gap_from = str(pd.to_datetime(int(ot[j - 1]), unit="ms", utc=True))[:10]
        gap_to = str(pd.to_datetime(int(ot[j]), unit="ms", utc=True))[:10]
    else:
        gap_from = gap_to = ""

    # 流动性：中位数的每小时成交额（USDT）。M3 可用它做结构性排除
    liq = float(d["quote_volume"].median())

    return {
        "symbol": p.stem,
        "bars": len(d),
        "expected_bars": expected,
        "missing_bars": missing,
        "missing_pct": round(100.0 * missing / max(expected, 1), 3),
        "n_gaps": int(len(big_gaps)),
        "max_gap_hours": round(float(big_gaps.max() / HOUR_MS), 1) if len(big_gaps) else 0.0,
        "max_gap_from": gap_from,
        "max_gap_to": gap_to,
        "zero_vol_pct": round(100.0 * float(zero_vol.mean()), 2),
        "max_zero_vol_run": longest_run(zero_vol),
        "flat_bar_pct": round(100.0 * float(flat.mean()), 2),
        "zombie_bars": zombie,
        "median_quote_vol_h": round(liq, 1),
        "first_bar": str(pd.to_datetime(int(ot[0]), unit="ms", utc=True))[:19],
        "last_bar": str(pd.to_datetime(int(ot[-1]), unit="ms", utc=True))[:19],
        "last_live_bar": str(pd.to_datetime(last_live, unit="ms", utc=True))[:19],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--zombie-threshold", type=int, default=6,
                    help="僵尸期超过多少根 K 线就报警")
    args = ap.parse_args()

    src = RAW / args.interval
    files = sorted(p for p in src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"没有找到 {src} 下的 parquet")

    rows = [r for r in (audit_symbol(p) for p in files) if r]
    df = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / f"data_audit_{args.interval}.csv", index=False, encoding="utf-8-sig")

    total = int(df["bars"].sum())
    zombie = df[df["zombie_bars"] > args.zombie_threshold].sort_values(
        "zombie_bars", ascending=False)
    gaps = df[df["missing_bars"] > 0].sort_values("missing_pct", ascending=False)
    dead_vol = df[df["zero_vol_pct"] > 1.0].sort_values("zero_vol_pct", ascending=False)

    L = []
    L.append(f"# 数据体检报告 · {args.interval}\n")
    L.append(f"- 交易对：**{len(df)}** 个 ｜ 合计 K 线：**{total:,}** 根")
    L.append(f"- 覆盖区间：`{df['first_bar'].min()[:10]}` ~ `{df['last_bar'].max()[:10]}`（UTC）")
    L.append(f"- 生成时间：{pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC\n")

    L.append("## 1. 时间缺口\n")
    if gaps.empty:
        L.append("无缺口，全部交易对的 1h 序列连续。\n")
    else:
        L.append(f"{len(gaps)} 个交易对存在缺口。**这是真实数据，不是抓取错误**——"
                 "Binance 历史上有过维护停机。回测时若把缺口当连续持有，会高估持仓时长。\n")
        L.append("| 交易对 | 缺失根数 | 占比 | 缺口段数 | 最长缺口(小时) | 最长缺口区间 |")
        L.append("|---|---|---|---|---|---|")
        for _, r in gaps.head(12).iterrows():
            span = f"{r['max_gap_from']} → {r['max_gap_to']}" if r["max_gap_from"] else "-"
            L.append(f"| {r['symbol']} | {r['missing_bars']} | {r['missing_pct']}% | "
                     f"{r['n_gaps']} | {r['max_gap_hours']} | {span} |")
        L.append("")
        susp = df[df["max_gap_hours"] > 24 * 3].sort_values("max_gap_hours", ascending=False)
        if not susp.empty:
            L.append("**停牌级缺口（>3 天）**——这些不是维护窗口，是真实的交易中断。"
                     "缺口两侧的收益不能连续计算，必须切成独立区间：\n")
            L.append("| 交易对 | 停牌天数 | 区间 | 停牌后中位成交额/小时 | 性质 |")
            L.append("|---|---|---|---|---|")
            for _, r in susp.iterrows():
                if r["median_quote_vol_h"] < 100_000:
                    nature = "停牌后极低流动性，等于死盘"
                elif r["last_live_bar"][:7] < df["last_live_bar"].max()[:7]:
                    nature = "停牌/崩塌后最终下架"
                else:
                    nature = "停牌后恢复，仍在交易"
                L.append(f"| {r['symbol']} | {r['max_gap_hours'] / 24:.0f} | "
                         f"{r['max_gap_from']} → {r['max_gap_to']} | "
                         f"{r['median_quote_vol_h']:,.0f} | {nature} |")
            L.append("")

    L.append("## 2. 僵尸尾部（已下架但数据仍在发布）\n")
    if zombie.empty:
        L.append("未发现僵尸尾部。\n")
    else:
        L.append(f"**{len(zombie)} 个交易对**在最后一次真实撮合之后，数据源里仍有 K 线。")
        L.append("这些根必须从回测样本中剔除，否则等于照着死盘下单。\n")
        L.append("| 交易对 | 僵尸根数 | 最后真实成交 | 数据最后一行 |")
        L.append("|---|---|---|---|")
        for _, r in zombie.iterrows():
            L.append(f"| {r['symbol']} | {r['zombie_bars']} | {r['last_live_bar'][:10]} | "
                     f"{r['last_bar'][:10]} |")
        L.append("")

    L.append("## 3. 零成交量与价格僵直\n")
    if dead_vol.empty:
        L.append("零成交占比都很低。\n")
    else:
        L.append("| 交易对 | 零成交占比 | 最长连续零成交 | 僵直K线占比 |")
        L.append("|---|---|---|---|")
        for _, r in dead_vol.iterrows():
            L.append(f"| {r['symbol']} | {r['zero_vol_pct']}% | {r['max_zero_vol_run']} | "
                     f"{r['flat_bar_pct']}% |")
        L.append("")

    L.append("## 4. 流动性（中位数每小时成交额，USDT）\n")
    L.append("M3 的结构性排除规则需要它：成交额太低的币，滑点会吃掉全部优势，"
             "而且'形态'在无流动性时不可信。\n")
    top = df.nlargest(8, "median_quote_vol_h")[["symbol", "median_quote_vol_h"]]
    bot = df.nsmallest(12, "median_quote_vol_h")[["symbol", "median_quote_vol_h"]]
    L.append("| 最活跃 | 中位成交额/小时 | 最不活跃 | 中位成交额/小时 |")
    L.append("|---|---|---|---|")
    for k in range(8):
        a = top.iloc[k] if k < len(top) else None
        b = bot.iloc[k] if k < len(bot) else None
        L.append(f"| {a['symbol']} | {a['median_quote_vol_h']:,.0f} | "
                 f"{b['symbol']} | {b['median_quote_vol_h']:,.0f} |"
                 if a is not None and b is not None else "")
    illiq = df[df["median_quote_vol_h"] < 100_000]
    L.append(f"\n中位成交额 < 10 万美元/小时的交易对：**{len(illiq)}** 个"
             f"（{'、'.join(illiq['symbol'].tolist()[:10])}）。\n"
             "这些在 M3 里应作为**单独一组**报告，不要混进主结论。\n")

    L.append("## 5. point-in-time 可见区间（前 20 个）\n")
    L.append("回测时必须按此区间过滤 —— 否则就是上帝视角。\n")
    L.append("| 交易对 | 可见起点 | 最后真实成交 | K线数 | 中位成交额/小时 |")
    L.append("|---|---|---|---|---|")
    for _, r in df.head(20).iterrows():
        L.append(f"| {r['symbol']} | {r['first_bar'][:10]} | {r['last_live_bar'][:10]} | "
                 f"{r['bars']} | {r['median_quote_vol_h']:,.0f} |")
    L.append("")
    L.append(f"完整明细：`results/data_audit_{args.interval}.csv`\n")

    (RESULTS / f"data_audit_{args.interval}.md").write_text("\n".join(L), encoding="utf-8")

    # 供回测直接消费的可见区间表
    vis = {r["symbol"]: {"first": r["first_bar"], "last_live": r["last_live_bar"],
                         "interval": args.interval}
           for r in rows}
    (ROOT / "data" / "visibility.json").write_text(
        json.dumps(vis, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"[audit] {args.interval}: {len(df)} 个交易对 / {total:,} 根")
    print(f"        时间缺口 {len(gaps)} 个 | 僵尸尾部 {len(zombie)} 个 | "
          f"零成交超1% {len(dead_vol)} 个")
    print(f"        报告 -> results/data_audit_{args.interval}.md")
    print(f"        可见区间 -> data/visibility.json")


if __name__ == "__main__":
    main()
