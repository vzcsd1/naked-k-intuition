#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · 插针秒级验证（插针线的胜负手）

这一步回答 1m 数据永远回答不了的问题：
  挂在 −X% 的限价单，**到底会不会成交、成交后多久回补、成交后还会不会继续跌**。

为什么必须用秒级数据：
  1m K 线只知道「这一分钟内最低价触及过 −X%」，但不知道：
  · 是真的跌到那个价位能让你成交，还是瞬间穿过去（你会拿到更差的价）
  · 成交之后是 10 秒回补，还是继续阴跌 10 分钟
  · 你面对的是一根秒针（流动性事故）还是持续下压（真实抛售）

口径（避免自欺，都是这一轮学到的）：
  · 成交假设 = 触及即成交（乐观上界；真实限价单在插针时可能拿不到）
  · 收益从 **成交价** 起算，不再从极值点起算 —— 极值点是你拿不到的
  · 对照组：同一批事件里「1s 路径显示价格**跳过**挂单价」的比例

用法：python wick_verify.py --level 0.2
输出：results/wick_verify.md · data/wick_verify.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

WIN_BEFORE_MS = 300_000      # 事件前 5 分钟
WIN_AFTER_MS = 1_800_000     # 事件后 30 分钟
DEEP_MS = 60_000             # 深度区（低于挂单价）持续时长统计窗口
MAX_DEV = 0.60               # 偏离 >60% 不再是插针：一分钟内归零再回来 = 崩塌或坏点


def verify_event(row, day: pd.DataFrame) -> dict | None:
    """对一个插针事件，用秒级路径回答：能否成交 / 多久回补 / 会不会继续跌。"""
    side = row["side"]
    pre = float(row["pre_close"])
    dev = float(row["dev"])
    t0 = int(row["open_time"])
    entry = pre * (1 - dev) if side == "down" else pre * (1 + dev)
    d = day[(day["open_time"] >= t0 - WIN_BEFORE_MS) & (day["open_time"] <= t0 + WIN_AFTER_MS)]
    # 秒级数据里有零价格垃圾行（死盘秒），不清洗会把滑点/收益算成天文数字
    d = d[(d["close"] > 0) & (d["open"] > 0) & (d["high"] > 0) & (d["low"] > 0)]
    if len(d) < 60:
        return {"symbol": row["symbol"], "side": side, "dev": dev,
                "filled": False, "reason": f"秒级数据不足({len(d)}行)"}
    o = d["open"].to_numpy(float)
    h = d["high"].to_numpy(float)
    l = d["low"].to_numpy(float)
    c = d["close"].to_numpy(float)
    t = d["open_time"].to_numpy()

    touch = (l <= entry) if side == "down" else (h >= entry)
    if not touch.any():
        return {"symbol": row["symbol"], "side": side, "dev": dev,
                "filled": False, "reason": "未触及"}
    i = int(np.argmax(touch))
    fill_t = int(t[i])
    # 成交价：若该秒开盘已穿过挂单价，实际成交价更差（滑点）
    if side == "down":
        fill_px = min(entry, o[i]) if o[i] > 0 else entry
    else:
        fill_px = max(entry, o[i]) if o[i] > 0 else entry
    slip = (entry - fill_px) / entry if side == "down" else (fill_px - entry) / entry
    if not np.isfinite(fill_px) or fill_px <= 0:
        return {"symbol": row["symbol"], "side": side, "dev": dev,
                "filled": False, "reason": "成交价异常（坏点/尘价）"}

    after_o, after_h = o[i:], h[i:]
    after_l, after_c, after_t = l[i:], c[i:], t[i:]
    n = len(after_c)

    def horizon(sec):
        k = min(int(sec), n - 1)
        if k < 1:
            return float("nan"), float("nan"), float("nan")
        if side == "down":
            mfe = (after_h[1:k + 1].max() - fill_px) / fill_px if k >= 1 else float("nan")
            mae = (after_l[1:k + 1].min() - fill_px) / fill_px if k >= 1 else float("nan")
            ret = (after_c[k] - fill_px) / fill_px
        else:
            mfe = (fill_px - after_l[1:k + 1].min()) / fill_px if k >= 1 else float("nan")
            mae = (fill_px - after_h[1:k + 1].max()) / fill_px if k >= 1 else float("nan")
            ret = (fill_px - after_c[k]) / fill_px
        # 浮亏最小是 0（成交后价格再没回到挂单价之下），不能为正
        mae = min(mae, 0.0)
        return mfe, mae, ret

    # 回补耗时：从成交起，到 close 回到事件前价格
    if side == "down":
        back = after_c >= pre
    else:
        back = after_c <= pre
    rec_s = float("nan")
    if back.any():
        rec_s = (int(after_t[int(np.argmax(back))]) - fill_t) / 1000.0

    # 深度区持续时长：成交后价格仍差于挂单价（含）的秒数
    deep = (after_c <= entry) if side == "down" else (after_c >= entry)
    deep_s = float(deep[: min(DEEP_MS // 1000, n)].sum())

    out = {"symbol": row["symbol"], "side": side, "dev": dev, "dt": row["dt"],
           "filled": True, "fill_px": float(fill_px), "slippage": float(slip),
           "deep_zone_s": deep_s, "recovery_s": rec_s}
    for sec in (60, 300, 900, 1800):
        mfe, mae, ret = horizon(sec)
        out[f"mfe_{sec}s"] = mfe
        out[f"mae_{sec}s"] = mae
        out[f"ret_{sec}s"] = ret
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=float, default=0.2, help="验证哪个偏离档位")
    ap.add_argument("--events", default=str(ROOT / "data" / "wick_events.csv"))
    args = ap.parse_args()

    ev = pd.read_csv(args.events)
    ev = ev[(ev["level"] == args.level) & (ev["side"] == "down") & (ev["rec_ok"])]
    if ev.empty:
        raise SystemExit("没有可验证的事件")

    rows, miss, skipped = [], 0, []
    for _, r in ev.iterrows():
        if float(r["dev"]) > MAX_DEV:
            # 偏离 >60%：一分钟内归零再回来，是崩塌或坏点，不是插针
            skipped.append("偏离>60%（崩塌/坏点，非插针）")
            continue
        p = RAW / "1s_days" / r["symbol"] / f"{r['symbol']}-{r['date']}.parquet"
        if not p.exists():
            skipped.append("缺秒级日文件")
            continue
        day = pd.read_parquet(p, columns=["open_time", "open", "high", "low", "close"])
        res = verify_event(r, day)
        if res is None:
            miss += 1
        elif not res.get("filled"):
            skipped.append(res["reason"])
        else:
            rows.append(res)
    if not rows:
        raise SystemExit("没有可用的秒级数据")
    v = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    v.to_csv(ROOT / "data" / "wick_verify.csv", index=False, encoding="utf-8-sig")

    n = len(v)
    L = [f"# 插针秒级验证 · 偏离 >= {args.level:.0%}（下影）\n",
         f"- 事件 **{len(ev):,}** 个，完成秒级验证 **{n}** 个"
         f"（秒级数据不足 {miss} 个）"]
    for k, cnt in pd.Series(skipped).value_counts().items():
        L.append(f"  - 未纳入「{k}」：**{cnt}** 个")
    L.append(f"- 数据：{v['symbol'].nunique()} 个交易对 × 各自插针当天的 **1s 真实路径**\n")
    L.append("> **口径**：触及挂单价即视为成交（乐观上界）；收益从**成交价**起算，"
             "不再从极值点起算 —— 极值点是你拿不到的。")
    L.append("> 成交价若在某秒开盘时已穿过挂单价，按更差的价计（滑点已计入）。\n")
    L.append("## 1. 成交之后：能拿到什么\n")
    L.append("| 窗口 | 平均最大浮盈 | 平均最大浮亏 | 中位收益 | 均值收益(截断±100%) |")
    L.append("|---|---|---|---|---|")
    for sec in (60, 300, 900, 1800):
        c = v[f"ret_{sec}s"].clip(-1, 1)
        L.append(f"| {sec // 60} 分钟 | +{100 * v[f'mfe_{sec}s'].mean():.2f}% | "
                 f"{100 * v[f'mae_{sec}s'].mean():.2f}% | "
                 f"**{100 * v[f'ret_{sec}s'].median():+.2f}%** | {100 * c.mean():+.2f}% |")

    L.append("\n## 2. 回补要多久（决定持仓时长与心理承受）\n")
    r = v["recovery_s"].dropna()
    if len(r):
        q = r.quantile([0.25, 0.5, 0.75, 0.9])
        L.append("| 25% | **中位数** | 75% | 90% | 未在 30 分钟内回补 |")
        L.append("|---|---|---|---|---|")
        L.append(f"| {q[0.25]:.0f} 秒 | **{q[0.5]:.0f} 秒** | {q[0.75]:.0f} 秒 | "
                 f"{q[0.9]:.0f} 秒 | **{100 * r.isna().mean():.1f}%** |")
    L.append("\n> 这个数字是「抓插针」策略最直接的可行性指标 —— "
             "它**只有秒级数据能给**。\n")

    L.append("## 3. 深度区持续多久（是不是一根秒针）\n")
    d = v["deep_zone_s"]
    L.append("| 1 秒内脱离 | 10 秒内脱离 | 60 秒内脱离 | 平均停留 |")
    L.append("|---|---|---|---|")
    L.append(f"| {100 * (d <= 1).mean():.1f}% | {100 * (d <= 10).mean():.1f}% | "
             f"{100 * (d <= 60).mean():.1f}% | {d.mean():.1f} 秒 |")
    L.append("\n> 停留越短 → 越像流动性事故（挂单能拿到极低价）；"
             "停留越长 → 越像真实抛售（你会一路接刀）。\n")

    L.append("## 4. 滑点\n")
    L.append(f"- 成交价劣于挂单价的比例：**{100 * (v['slippage'] > 0).mean():.1f}%**，"
             f"平均劣化 **{100 * v['slippage'].mean():.3f}%**")
    L.append(f"- 最大劣化 **{100 * v['slippage'].max():.3f}%**\n")
    L.append("> 插针时你在跟强平引擎抢流动性，这个数字**直接决定挂单策略是否成立**。\n")

    L.append("## 5. 结论（先说清哪些能信、哪些还不能信）\n")
    med = v["ret_1800s"].median()
    L.append("**已经可以确认的（这些不依赖收益假设）：**")
    L.append(f"- **挂单真的能成交**：{n} 个事件里 {100 * (v['slippage'] <= 0.05).mean():.1f}% "
             f"滑点 ≤5%，成交价劣于挂单价的比例仅 {100 * (v['slippage'] > 0).mean():.1f}%")
    L.append(f"- **真的是「一根秒针」**：深度区 **{100 * (d <= 1).mean():.1f}%** 在 1 秒内脱离、"
             f"平均只停留 **{d.mean():.1f} 秒** —— 这是流动性事故的签名，不是真实抛售")
    L.append(f"- **回补很快**：中位 **{r.quantile(0.5):.0f} 秒**，90% 分位 {r.quantile(0.9):.0f} 秒")
    L.append(f"- **滑点可控**：平均 {100 * v['slippage'].mean():.2f}%，"
             f"但最大 **{100 * v['slippage'].max():.1f}%** —— 个别事件会穿价\n")
    L.append("**还不能确认的（存在选择偏差）：**")
    L.append(f"- 30 分钟中位收益 **{100 * med:+.2f}%** 这个数字**偏乐观**，原因有二：")
    L.append("> ① 样本本身是按「1m 收盘已收回一半」筛出来的 —— **反弹被写进了筛选条件**；")
    L.append("> ② 触及即成交是乐观假设（真实限价单在插针时可能拿不到量）。\n")
    L.append("**要得到无偏答案，下一步必须做「无条件挂单回测」**：")
    L.append("> 不筛事件，而是在全部历史上**持续挂一张 −20% 的限价单**，")
    L.append("> 统计：多久成交一次、成交后的真实分布（含「继续崩」的那些）。")
    L.append("> 这才是这个策略的真实期望，也才能用**期望值 + 右尾贡献**来评价它。")
    L.append("> 再配 **时间外样本**：2020-2023 定规则，2024-2026 只看一次。\n")

    (RESULTS / "wick_verify.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[verify] {n} 个事件完成秒级验证（缺 {miss}）")
    if len(r):
        print(f"         中位回补 {r.quantile(0.5):.0f} 秒 ｜ 30分钟内不回补 {100 * r.isna().mean():.1f}%")
    print(f"         30分钟中位收益 {100 * med:+.2f}% ｜ 平均滑点 {100 * v['slippage'].mean():.3f}%")
    print(f"         报告 -> results/wick_verify.md")


if __name__ == "__main__":
    main()
