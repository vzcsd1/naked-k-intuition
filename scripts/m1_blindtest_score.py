# -*- coding: utf-8 -*-
"""对照盘实验 v2 计分：answers.json + key.json -> 四通道被选频率 + 逐组揭晓页。

用法：python m1_blindtest_score.py
输入：results/m1_blindtest/answers.json（盲测页导出）
输出：results/m1_blindtest_结果.md + results/m1_blindtest/reveal.html
"""
import argparse
import base64
import json
import os
import time

import numpy as np
import pandas as pd

import m1_lib as L

OUT = os.path.join(L.DIR_RESULTS, "m1_blindtest")
CH_LABEL = {"C1_shape_detrend": "C1 形状·去趋势",
            "C2_shape_raw": "C2 形状·带背景",
            "C3_context_random": "C3 同处境随机",
            "C4_random": "C4 完全随机"}


def load_window(sym, s_ot, n=L.W):
    df = L.load_15m(sym, columns=["open_time", "open", "high", "low", "close", "volume"])
    ot = df["open_time"].to_numpy(np.int64)
    p = int(np.searchsorted(ot, int(s_ot), side="left"))
    return df.iloc[p:p + n].reset_index(drop=True)


def fwd_stats(sym, s_ot, n=L.FUTURE_BARS):
    """揭晓用：未来 n 根的 (末收益/ATR, 最大上影/ATR, 最大下探/ATR)。"""
    df = L.load_15m(sym, columns=["open_time", "high", "low", "close"])
    ot = df["open_time"].to_numpy(np.int64)
    p = int(np.searchsorted(ot, int(s_ot), side="left"))
    w = df.iloc[p:p + n + L.W]
    if len(w) < L.W + 5:
        return None
    tr = L.true_range(w["high"].to_numpy(float), w["low"].to_numpy(float),
                      w["close"].to_numpy(float))
    atr = float(tr[:L.W].mean())
    fut = w.iloc[L.W:]
    c0, cend = float(w["close"].iloc[L.W - 1]), float(fut["close"].iloc[-1])
    return dict(fwd=(cend - c0) / atr,
                max_up=(float(fut["high"].max()) - c0) / atr,
                max_dn=(float(fut["low"].min()) - c0) / atr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=os.path.join(OUT, "answers.json"))
    ap.add_argument("--key", default=os.path.join(OUT, "key.json"))
    ap.add_argument("--md-out", default=os.path.join(L.DIR_RESULTS, "m1_blindtest_结果.md"))
    args = ap.parse_args()
    with open(args.answers, encoding="utf-8") as f:
        ans = json.load(f)["answers"]
    with open(args.key, encoding="utf-8") as f:
        key = json.load(f)["groups"]

    tally = {ch: 0 for ch in CH_LABEL}
    rows = []
    for g in key:
        gi = str(g["group"])
        if gi not in ans:
            continue
        letter = ans[gi]["pick"]
        ch = g["order"]["ABCD".index(letter)]
        tally[ch] += 1
        rows.append(dict(group=g["group"], pick_letter=letter, channel=ch,
                         reason=ans[gi].get("reason", ""),
                         query_sym=g["query_sym"],
                         query_start=g["query_start"],
                         **{f"cand_{ch2}": dict(
                             sym=g["candidates"][ch2]["sym"],
                             start=g["candidates"][ch2]["start"],
                             reveal=fwd_stats(g["candidates"][ch2]["sym"],
                                              g["candidates"][ch2]["start"]))
                         for ch2 in CH_LABEL}))

    n = len(rows)
    lines = [f"# 对照盘实验 v2 · 计分\n",
             f"> 完成组数 {n}/{len(key)} ｜ 生成 {time.strftime('%Y-%m-%d %H:%M')}\n",
             "\n## 四通道被选频率\n",
             "| 通道 | 被选 | 占比 |\n|---|---|---|\n"]
    for ch, c in sorted(tally.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {CH_LABEL[ch]} | {c} | {c/max(n,1)*100:.0f}% |\n")
    lines.append("\n## 怎么读\n")
    lines.append("- **C1 > C2**：你主要看形态本身，去趋势是对的（08 的形状通道）\n"
                 "- **C2 > C1**：你在乎涨跌背景，v1 索引应把趋势块权重调高\n"
                 "- **C3 高**：处境像对你很重要 -> M4 条件检索提前\n"
                 "- **C4 >= 10 次**：判断可靠性存疑，重做实验（换 seed）\n"
                 "- 结合各组理由关键词（量能/位置/节奏）转成特征清单\n")
    lines.append("\n## 逐组\n")
    lines.append("| 组 | 你选 | 实际是 | 理由 |\n|---|---|---|---|\n")
    for r in rows:
        lines.append(f"| {r['group']} | {r['pick_letter']} | {CH_LABEL[r['channel']]} | "
                     f"{r['reason']} |\n")

    out_md = args.md_out
    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("".join(lines[:12]))
    print(f"-> {out_md}")


if __name__ == "__main__":
    main()
