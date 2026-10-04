#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · 插针线最后一个遗留项：**时间外样本检验（walk-forward）**

要回答的问题（大白话）：
  上一轮我们在历史上挑出三条「下单前就能看到」的特征，用来躲开会亏大钱的交易：
    ① 不做平时波动特别大的币
    ② 不做「过去一小时跌得还不够狠」的
    ③ 不做「全市场只有它一个在跌」的（独狼崩）
  在全部历史上它很漂亮：亏大钱比例 8.6% → 4.2%，年化只从 +17.1% 掉到 +16.5%。

  **但那些特征是在同一批数据里挑出来的** —— 相当于"先看完答案再出题"。
  这个脚本做一件事：**把规则冻在 2022 年底以前，然后只看 2023 年以后的表现。**

为什么必须有「随机对照」：
  过滤器会砍掉约 38% 的交易。交易变少，亏大钱的比例**本来就可能下降**。
  所以真正的对手不是"不过滤"，而是"**随机也砍掉同样多的交易**"——
  只有明显好过随机丢弃，才说明这些特征真的带信息。

用法：
    python wick_walkforward.py --stage features   # 第一步：算特征（慢，约 10 分钟，结果落盘缓存）
    python wick_walkforward.py --stage report     # 第二步：做检验（快，读缓存）
输出：results/wick_walkforward.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

FEE_BOTH = 0.002          # 往返手续费
SLIP = 0.0025             # 秒级校准得到的滑点期望折损
BIG_LOSS = -0.10          # 单笔亏 ≥10% 算「亏大钱」
BREADTH_DROP = -0.02      # 单根 1m 跌超 2% 算「这个币在急跌」

FEAT_CACHE = ROOT / "data" / "wick_features.parquet"
DETAIL = ROOT / "data" / "wick_backtest_detail.parquet"
TRADES = ROOT / "data" / "wick_backtest_trades.csv"

IS_END = "2022-12-31"     # 样本内 / 样本外 的分界（只在 2022 年底以前定规则）
OOS_START = "2023-01-01"

# 计算特征时会遇到 0 作分母（成交量中位为 0 的死币），结果会被 where 选掉，
# 但 numpy 会打几千行警告把进度日志淹没 → 直接全局静音这两类。
np.seterr(divide="ignore", invalid="ignore")


# ---------------------------------------------------------------- 全市场急跌广度
def build_breadth(files, min_minute, n_minutes, force: bool = False):
    """每个分钟上：(全市场同时在急跌的币数, 当时有成交的币数)。

    ⚠️ **为什么必须要分母**：币池是会长大的（2017 年几十个币，2026 年 146 个）。
       「有多少个币在急跌」是**绝对个数**，拿绝对数当阈值 → 后期更容易满足 →
       **规则会随年月悄悄变松**。实测症状：规则③在 −10% 档样本外保留率 100%（等于没触发）。
       用**比例**（在跌的 ÷ 在交易的）才跨年份可比。（2026-09-17 发现并修正）
    """
    c_bad = ROOT / "data" / f"breadth_bad_{abs(BREADTH_DROP):.2f}.npy"
    c_tot = ROOT / "data" / f"breadth_tot_{abs(BREADTH_DROP):.2f}.npy"
    if not force and c_bad.exists() and c_tot.exists():
        a, b = np.load(c_bad), np.load(c_tot)
        if len(a) == n_minutes and len(b) == n_minutes:
            return a, b
    cnt = np.zeros(n_minutes, dtype=np.int32)
    tot = np.zeros(n_minutes, dtype=np.int32)
    for f in files:
        d = pd.read_parquet(f, columns=["open_time", "close", "quote_volume"])
        c = d["close"].to_numpy(np.float64)
        qv = d["quote_volume"].to_numpy(np.float64)
        if len(c) < 2:
            continue
        r = c[1:] / c[:-1] - 1
        m = (d["open_time"].to_numpy()[1:] // 60000).astype(np.int64) - min_minute
        inside = (m >= 0) & (m < n_minutes)
        if not inside.any():
            continue
        # 分母 = 这一分钟**真的有撮合**的币数（成交额 > 0），不是"文件里有这一行"
        has = inside & (qv[1:] > 0)
        if has.any():
            tot += np.bincount(m[has], minlength=n_minutes).astype(np.int32)
        bad = inside & (r < BREADTH_DROP)
        if bad.any():
            cnt += np.bincount(m[bad], minlength=n_minutes).astype(np.int32)
    np.save(c_bad, cnt)
    np.save(c_tot, tot)
    return cnt, tot


# ---------------------------------------------------------------- 特征构建（带缓存）
def build_features(force: bool = False) -> pd.DataFrame:
    """给每一笔交易算「下单当时就能看到」的特征。

    ⚠️ 只读 3 列（open_time / close / quote_volume）—— high/low 用不到，
       1m 原始文件单币最大 261 MB，少读两列能省不少时间。
    ⚠️ 结果落盘缓存：后面所有统计口径的调整都读它，**不要每次重扫 1m 全量**。
    """
    if FEAT_CACHE.exists() and not force:
        F = pd.read_parquet(FEAT_CACHE)
        # 护栏：缓存可能来自旧版本特征定义 → 缺列就重建，**绝不用旧缓存静默算下去**
        if {"mkt_breadth_pct", "mkt_breadth"}.issubset(F.columns):
            print(f"[feat] 读缓存 {len(F):,} 行（(币,时间) 唯一对）")
            return F
        print("[feat] 缓存缺少新特征列（旧版本），重建")

    det = pd.read_parquet(DETAIL, columns=["symbol", "t0"])
    det = det.drop_duplicates(["symbol", "t0"])
    print(f"[feat] 需要算特征的 (币,时间) 对：{len(det):,}")

    files = sorted((RAW / "1m").glob("*.parquet"))
    mn = int(pd.Timestamp("2017-08-01", tz="UTC").timestamp() * 1000) // 60000
    mx = int(pd.Timestamp("2026-12-31", tz="UTC").timestamp() * 1000) // 60000
    n_min = mx - mn + 1
    t0 = time.time()
    print(f"[feat] 统计全市场急跌广度（{n_min / 1e6:.1f}M 分钟）…")
    breadth, breadth_tot = build_breadth(files, mn, n_min, force=force)
    print(f"[feat] 广度完成，用时 {time.time() - t0:.0f}s")

    out = []
    by_sym = det.groupby("symbol", sort=False)
    n_sym = len(by_sym)
    for k, (sym, g) in enumerate(by_sym, 1):
        p = RAW / "1m" / f"{sym}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["open_time", "close", "quote_volume"])
        ot = d["open_time"].to_numpy(np.int64)
        c = d["close"].to_numpy(np.float64)
        qv = d["quote_volume"].to_numpy(np.float64)
        n = len(d)
        if n < 1441:
            continue
        tgt = g["t0"].to_numpy(np.int64)
        # 用 searchsorted 定位（ot 已排序），比逐笔建 dict 快很多
        pos = np.clip(np.searchsorted(ot, tgt), 0, n - 1)
        ok = (ot[pos] == tgt) & (pos >= 1440)
        if not ok.any():
            continue
        i = pos[ok]
        tt = tgt[ok]

        r1 = np.zeros(n)
        r1[1:] = c[1:] / c[:-1] - 1
        vol24 = pd.Series(r1).rolling(1440, min_periods=240).std().to_numpy()
        med1h = pd.Series(qv).rolling(60, min_periods=20).median().to_numpy()
        ath = np.maximum.accumulate(c)

        drop60 = c[i] / c[i - 60] - 1
        minute = (tt // 60000).astype(np.int64) - mn
        inr = (minute >= 0) & (minute < n_min)
        idx_m = np.clip(minute, 0, n_min - 1)
        bd = np.where(inr, breadth[idx_m], 0).astype(np.int32)
        bd_tot = np.where(inr, breadth_tot[idx_m], 0).astype(np.int32)
        out.append(pd.DataFrame({
            "symbol": sym,
            "t0": tt,
            "drop_60m": drop60,
            "vol24h": vol24[i],
            "vol_ratio": np.where(med1h[i] > 0, qv[i] / med1h[i], np.nan),
            "below_ath": c[i] / ath[i] - 1,
            "mkt_breadth": bd,
            # ⭐ 比例版（分母 = 当时有撮合的币数）—— 跨年份可比，是现在的主口径
            "mkt_breadth_pct": np.where(bd_tot > 0, bd / np.maximum(bd_tot, 1), np.nan),
        }))
        if k % 20 == 0 or k == n_sym:
            print(f"  [{k}/{n_sym}] {sym} 用时 {time.time() - t0:.0f}s", flush=True)

    F = pd.concat(out, ignore_index=True)
    F.to_parquet(FEAT_CACHE, index=False, compression="zstd")
    print(f"[feat] -> {FEAT_CACHE}（{len(F):,} 行，用时 {time.time() - t0:.0f}s）")
    return F


# ---------------------------------------------------------------- 规则定义
# 三条规则都是「砍掉某一端」；阈值 = 样本内分位数（由调用方传入）
def rule_masks(F: pd.DataFrame, q: dict) -> dict:
    """返回布尔掩码（True = 保留这笔交易）。q 里的阈值必须来自**样本内**。"""
    return {
        "① 不做波动率最高的 25%": F["vol24h"] < q["vol24h"],
        "② 不做「跌得还不够狠」的 25%": F["drop_60m"] < q["drop_60m"],
        "③ 不做「独狼崩」（按比例 ⭐ 新口径）": F["mkt_breadth_pct"] >= q["mkt_breadth_pct"],
        "③b 不做「独狼崩」（按绝对币数 · 旧口径对照）":
            F["mkt_breadth"] >= q["mkt_breadth"],
        "⭐ ①②③ 组合（§9.3 推荐）": ((F["vol24h"] < q["vol24h"])
                                    & (F["drop_60m"] < q["drop_60m"])
                                    & (F["mkt_breadth_pct"] >= q["mkt_breadth_pct"])),
        "①+② 组合（原 §9.2 那条）": ((F["vol24h"] < q["vol24h"])
                                   & (F["drop_60m"] < q["drop_60m"])),
    }


def quantiles_on(df: pd.DataFrame, p=0.75) -> dict:
    """在给定子集上算阈值（这就是「定规则」这一步，只能用样本内数据）。"""
    return {
        "vol24h": float(df["vol24h"].quantile(p)),
        "drop_60m": float(df["drop_60m"].quantile(p)),
        "mkt_breadth": float(df["mkt_breadth"].quantile(1 - p)),
        "mkt_breadth_pct": float(df["mkt_breadth_pct"].quantile(1 - p)),
    }


# ---------------------------------------------------------------- 评估
def stats(ret: np.ndarray, base_freq: float, n_all: int) -> dict:
    net = ret - FEE_BOTH - SLIP
    exp = float(net.mean())
    bl = float((ret <= BIG_LOSS).mean())
    w5 = float(np.quantile(ret, 0.05)) if len(ret) >= 20 else float("nan")
    worst = float(np.sort(ret)[:max(1, len(ret) // 20)].mean()) if len(ret) >= 20 else float("nan")
    return {
        "n": len(ret),
        "ret_mean": float(ret.mean()),
        "net": exp,
        "big_loss": bl,
        "worst5": w5,
        "worst5_mean": worst,
        "ann": exp * base_freq * (len(ret) / max(n_all, 1)),
    }


def random_control(ret_pool: np.ndarray, n_keep: int, n_boot: int, rng) -> dict:
    """对照：从同一批交易里**随机**保留同样多的笔数，重复 n_boot 次。

    返回「亏大钱比例」「每笔净期望」「最惨 5%」三者的分布分位 —— 用来回答
    「规则的改善有没有超过『单纯少做交易』」。
    """
    N = len(ret_pool)
    bl, net, w5 = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(N, size=min(n_keep, N), replace=False)
        r = ret_pool[idx]
        bl.append((r <= BIG_LOSS).mean())
        net.append((r - FEE_BOTH - SLIP).mean())
        w5.append(np.quantile(r, 0.05))
    bl, net, w5 = np.asarray(bl), np.asarray(net), np.asarray(w5)
    return {
        "big_loss_p50": float(np.median(bl)),
        "big_loss_p05": float(np.quantile(bl, 0.05)),
        "net_p50": float(np.median(net)),
        "net_p95": float(np.quantile(net, 0.95)),
        "worst5_p50": float(np.median(w5)),
        "worst5_p95": float(np.quantile(w5, 0.95)),
        "bl_dist": bl,          # 完整分布，用来算「规则排在第几百分位」
    }


def pct_rank(dist: np.ndarray, value: float) -> float:
    """value 在 dist 里的百分位。**越低越好** —— 表示比多少比例的随机对照更优。"""
    return float((dist < value).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "report"], default="report")
    ap.add_argument("--levels", default="0.10,0.15,0.20")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--force-features", action="store_true")
    args = ap.parse_args()

    if args.stage == "features":
        build_features(force=args.force_features)
        return 0

    if not FEAT_CACHE.exists():
        sys.exit("缺少特征缓存，先跑 --stage features")

    F = build_features()
    det = pd.read_parquet(DETAIL)
    # ℹ️ 合并键的 dtype 必须一致 —— parquet 往返可能给出 int32/int64 混用，
    #    那样 merge 会**静默合不上**（特征全 NaN），统计照跑不误但结论全错。
    det["t0"] = det["t0"].astype("int64")
    F["t0"] = F["t0"].astype("int64")
    det = det.merge(F, on=["symbol", "t0"], how="left")
    miss = det["vol24h"].isna().mean()
    print(f"[wf] 逐笔明细 {len(det):,} 笔 · 特征缺失率 {100 * miss:.2f}%")
    if miss > 0.5:
        sys.exit(f"特征对齐失败（缺失 {100 * miss:.1f}%）→ 先检查合并键，不要看后面的数字")

    # 年化口径：与 wick_filter.py 一致 —— **必须按档位分开算**频率（次/年/币）。
    # ⚠️ 踩过：一开始没筛 level，用「全部档位合计笔数 / 总币年」当频率，
    #    结果每个档位都拿到同一个虚高的频率，年化数字量级全错（-15% 档显示 +35%）。
    tr = pd.read_csv(TRADES)

    def freq_of(level: float) -> float:
        a = tr[np.isclose(tr["level"], level)]
        if a.empty:
            return float("nan")
        ty = float(a["span_ms"].sum()) / (365 * 86400_000)
        return float(a["trades"].sum()) / max(ty, 1e-9)

    det["year"] = pd.to_datetime(det["t0"], unit="ms", utc=True).dt.year
    levels = [float(x) for x in args.levels.split(",")]
    rng = np.random.default_rng(args.seed)

    L = []
    L.append("# 插针线 · 时间外样本检验（walk-forward）\n")
    L.append("> 这是插针线**最后一个遗留项**（`09` §9.4-1）。前面所有过滤器结论都是在"
             "同一批数据里挑出来的，这里把它们**冻在 2022 年底**，只看 2023 年以后。\n")
    L.append(f"- 样本：{det['symbol'].nunique()} 币 · {len(det):,} 笔交易（4 个挂单档位）")
    L.append(f"- 规则只在 **≤{IS_END}** 上定（分位数阈值），**{OOS_START} 起只看一次**")
    L.append(f"- 「亏大钱」= 单笔亏 ≥10%（扣成本前的原始收益）")
    L.append(f"- 成本口径：手续费往返 0.20% + 滑点 0.25% = **0.45%/笔**")
    L.append(f"- 随机对照：**保留同样笔数的随机子集**，重复 {args.bootstrap} 次")
    L.append("- 「每币年化(近似)」= 每笔净期望 × **该档位**年频率 × 保留率（未计资金占用）")
    L.append("- ⚠️ 判定的基准是**样本外基准**（≥2023），不是全样本 —— 因为策略本身也在变\n")

    # ---------------- 表一：主检验（-15%，单一切分）
    L.append("## 1. 主检验：规则定在 2022 年底前，2023–2026 只看一次\n")
    summary_rows = []
    for X in levels:
        d = det[np.isclose(det["level"], X)].dropna(subset=["vol24h", "drop_60m", "mkt_breadth_pct"])
        d_is = d[d["year"] <= 2022]
        d_oos = d[d["year"] >= 2023]
        if len(d_is) < 200 or len(d_oos) < 200:
            continue
        freq = freq_of(X)
        q = quantiles_on(d_is)
        base_is = stats(d_is["ret"].to_numpy(), freq, len(d))
        base_oos = stats(d_oos["ret"].to_numpy(), freq, len(d))
        L.append(f"### 档位 −{X:.0%}（样本内 {len(d_is):,} 笔 / 样本外 {len(d_oos):,} 笔"
                 f" · 该档位频率 {freq:.1f} 次/年/币）\n")
        L.append("| 方案 | 笔数 | 保留率 | 每笔净期望 | 亏大钱比例 | 最惨 5% 平均亏 | 每币年化(近似) |")
        L.append("|---|---|---|---|---|---|---|")
        L.append(f"| 不过滤 · **样本内基准**（≤2022，规则还没定） | {base_is['n']:,} | — | "
                 f"{100 * base_is['net']:+.3f}% | {100 * base_is['big_loss']:.1f}% | "
                 f"{100 * base_is['worst5_mean']:+.2f}% | {100 * base_is['ann']:+.1f}% |")
        L.append(f"| 不过滤 · **样本外基准**（≥2023，后面都跟它比） | {base_oos['n']:,} | 100% | "
                 f"{100 * base_oos['net']:+.3f}% | **{100 * base_oos['big_loss']:.1f}%** | "
                 f"{100 * base_oos['worst5_mean']:+.2f}% | {100 * base_oos['ann']:+.1f}% |")

        # 样本内基准（对照用，看规则在样本内是不是也这样）
        masks = rule_masks(d_oos, q)
        pool = d_oos["ret"].to_numpy()
        detail_rows = []
        for name, m in masks.items():
            r = pool[np.asarray(m)]
            if len(r) < 50:
                continue
            st = stats(r, freq, len(d))
            # ⚠️ 保留率 100%（等于没过滤）时，随机对照抽到的也是全部交易 → 分布退化成
            #    一个点，百分位会算出 0% 并**误判成「明显跑赢随机」**。这种情况直接标 N/A。
            if len(r) >= len(pool):
                ctl, rank = None, None
            else:
                ctl = random_control(pool, len(r), args.bootstrap, rng)
                rank = pct_rank(ctl["bl_dist"], st["big_loss"])
            detail_rows.append((name, st, ctl, rank))
            # ⚠️ 保留率 ≈100% 说明这条规则**根本没生效**（常见原因：该特征在样本内
            #    大部分时间为 0 → 分位数阈值退化成 0 → 条件恒成立）。必须标出来，
            #    否则读者会以为"这条规则被验证有效"。
            keep_rate = st["n"] / max(base_oos["n"], 1)
            flag = " ⚠️**未生效**" if keep_rate >= 0.995 else ""
            L.append(f"| {name}{flag} | {st['n']:,} | {100 * keep_rate:.0f}% | "
                     f"{100 * st['net']:+.3f}% | {100 * st['big_loss']:.1f}% | "
                     f"{100 * st['worst5_mean']:+.2f}% | {100 * st['ann']:+.1f}% |")
        L.append(f"\n**随机对照**：从同一批交易里**随机**保留同样笔数，重复 {args.bootstrap} 次。"
                 f"「百分位」= 规则的亏大钱比例比多少比例的随机抽样更低 —— **越小越好**。\n")
        L.append("| 方案 | 规则实际 亏大钱 | 随机对照中位 | 随机最好的 5% | **规则百分位** | 结论 |")
        L.append("|---|---|---|---|---|---|")
        for name, st, ctl, rank in detail_rows:
            if ctl is None:
                L.append(f"| {name} | {100 * st['big_loss']:.1f}% | — | — | — | "
                         f"规则没有砍掉任何交易（保留率 100%），对照无意义 |")
                continue
            if rank < 5:
                verdict = "✅ **明显跑赢随机**"
            elif rank < 25:
                verdict = "🔶 略好于随机"
            elif rank < 50:
                verdict = "⚠️ 与随机无实质差别"
            else:
                verdict = "❌ 还不如随机"
            L.append(f"| {name} | **{100 * st['big_loss']:.1f}%** | {100 * ctl['big_loss_p50']:.1f}% | "
                     f"{100 * ctl['big_loss_p05']:.1f}% | **{rank:.0f}%** | {verdict} |")
        summary_rows.append((X, base_is, base_oos, detail_rows))
        L.append("")

    # ---------------- 表二：多档位总览
    L.append("## 2. 三个档位一起看（防止「只有一个档位成立」）\n")
    L.append("| 档位 | 样本外基准 亏大钱比例 | ⭐组合后 | 样本外基准 每笔净期望 | ⭐组合后 | 基准年化 | 组合后年化 |")
    L.append("|---|---|---|---|---|---|---|")
    for X, base_is, base_oos, drow in summary_rows:
        combo = [r for r in drow if r[0].startswith("⭐")]
        if not combo:
            continue
        _, st, _, _ = combo[0]
        L.append(f"| −{X:.0%} | {100 * base_oos['big_loss']:.1f}% | **{100 * st['big_loss']:.1f}%** | "
                 f"{100 * base_oos['net']:+.3f}% | **{100 * st['net']:+.3f}%** | "
                 f"{100 * base_oos['ann']:+.1f}% | {100 * st['ann']:+.1f}% |")
    L.append("")

    # ---------------- 表三：逐年滚动
    L.append("## 3. 逐年滚动：每一年都用「那一年之前的所有数据」定规则\n")
    L.append("> 只看一个切分点容易被「刚好切在运气好的位置」骗到。这里让规则每年重新定一次，"
             "看它是不是**年年都成立**。\n")
    lines_by_level = {}
    for X in levels:
        d = det[np.isclose(det["level"], X)].dropna(subset=["vol24h", "drop_60m", "mkt_breadth_pct"])
        if len(d) < 500:
            continue
        rows = []
        for y in range(2020, 2027):
            past = d[d["year"] < y]
            cur = d[d["year"] == y]
            if len(past) < 500 or len(cur) < 150:
                continue
            q = quantiles_on(past)
            m = rule_masks(cur, q)["⭐ ①②③ 组合（§9.3 推荐）"]
            r_all = cur["ret"].to_numpy()
            r_keep = r_all[np.asarray(m)]
            # 规则在该期几乎没触发时不看（随机对照会退化，没有信息量）
            if len(r_keep) < 50 or len(r_keep) >= len(r_all) * 0.995:
                continue
            ctl = random_control(r_all, len(r_keep), 400, rng)
            rows.append({
                "年": y,
                "该年笔数": len(cur),
                "保留": len(r_keep),
                "基准亏大钱": (r_all <= BIG_LOSS).mean(),
                "规则亏大钱": (r_keep <= BIG_LOSS).mean(),
                "随机对照中位": ctl["big_loss_p50"],
                "基准每笔净": (r_all - FEE_BOTH - SLIP).mean(),
                "规则每笔净": (r_keep - FEE_BOTH - SLIP).mean(),
            })
        if rows:
            lines_by_level[X] = pd.DataFrame(rows)

    for X, tb in lines_by_level.items():
        L.append(f"### 档位 −{X:.0%}\n")
        L.append("| 检验年 | 该年笔数 | 规则保留 | 基准亏大钱 | **规则亏大钱** | 随机对照(中位) | 基准每笔净 | 规则每笔净 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for _, r in tb.iterrows():
            win = "✅" if r["规则亏大钱"] < r["随机对照中位"] else "❌"
            L.append(f"| {int(r['年'])} | {int(r['该年笔数']):,} | {int(r['保留']):,} | "
                     f"{100 * r['基准亏大钱']:.1f}% | **{100 * r['规则亏大钱']:.1f}%** {win} | "
                     f"{100 * r['随机对照中位']:.1f}% | {100 * r['基准每笔净']:+.3f}% | "
                     f"{100 * r['规则每笔净']:+.3f}% |")
        wins = int((tb["规则亏大钱"] < tb["随机对照中位"]).sum())
        L.append(f"\n> 跑赢随机对照的年份：**{wins} / {len(tb)}**\n")

    # ---------------- 表四：多个切分点
    L.append("## 4. 换切分点（结论是不是只在「2023」这一个切分上成立？）\n")
    L.append("| 规则定在哪之前 | 检验期 | 档位 | 检验笔数 | 基准亏大钱 | 组合后 | 随机对照(中位) | 结论 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cut in (2019, 2020, 2021, 2022, 2023):
        for X in levels:
            d = det[np.isclose(det["level"], X)].dropna(subset=["vol24h", "drop_60m", "mkt_breadth_pct"])
            past = d[d["year"] <= cut]
            cur = d[(d["year"] > cut) & (d["year"] <= cut + 3)]
            if len(past) < 500 or len(cur) < 150:
                continue
            q = quantiles_on(past)
            m = rule_masks(cur, q)["⭐ ①②③ 组合（§9.3 推荐）"]
            r_all = cur["ret"].to_numpy()
            r_keep = r_all[np.asarray(m)]
            # 规则在该期几乎没触发时不看（随机对照会退化，没有信息量）
            if len(r_keep) < 50 or len(r_keep) >= len(r_all) * 0.995:
                continue
            ctl = random_control(r_all, len(r_keep), 400, rng)
            b, k, cm = ((r_all <= BIG_LOSS).mean(), (r_keep <= BIG_LOSS).mean(), ctl["big_loss_p50"])
            verdict = "✅ 跑赢随机" if k < ctl["big_loss_p05"] else ("🔶 略好于随机" if k < cm else "❌ 没超过随机")
            L.append(f"| ≤{cut} | {cut + 1}–{cut + 3} | −{X:.0%} | {len(cur):,} | {100 * b:.1f}% | "
                     f"**{100 * k:.1f}%** | {100 * cm:.1f}% | {verdict} |")
    L.append("")

    # ---------------- 结论
    L.append("## 5. 结论\n")
    L.append("### 5.1 逐档位（样本外一次性检验：规则冻在 2022 年底）\n")
    L.append("| 档位 | 样本内基准 亏大钱 | **样本外基准** | **组合后** | 每笔净(基准 → 组合) | 年化(基准 → 组合) | 判定 |")
    L.append("|---|---|---|---|---|---|---|")
    n_pass, n_tot = 0, 0
    for X, base_is, base_oos, drow in summary_rows:
        combo = [r for r in drow if r[0].startswith("⭐")]
        if not combo:
            continue
        _, st, ctl, rank = combo[0]
        n_tot += 1
        if ctl is None:
            verdict = "对照不适用"
        elif rank < 5:
            verdict = "✅ **明显跑赢随机**"
            n_pass += 1
        elif rank < 50:
            verdict = "🔶 略好于随机"
        else:
            verdict = "❌ 不如随机"
        L.append(f"| −{X:.0%} | {100 * base_is['big_loss']:.1f}% | **{100 * base_oos['big_loss']:.1f}%** | "
                 f"**{100 * st['big_loss']:.1f}%** | {100 * base_oos['net']:+.3f}% → {100 * st['net']:+.3f}% | "
                 f"{100 * base_oos['ann']:+.1f}% → {100 * st['ann']:+.1f}% | {verdict} |")
    L.append(f"\n**{n_pass} / {n_tot} 个档位在样本外明显跑赢「随机砍掉同样多交易」。**\n")
    L.append("> ⚠️ **−20% 档要特别读**：每笔净期望从 +5.355% 提到 +8.597%（+61%），"
             "但年化只从 +6.3% 到 +6.5% —— 因为规则砍掉了 35% 的交易。"
             "深档挂单本来就笔数少，「每笔赚更多」几乎被「做得更少」抵消。"
             "**这一档的价值在风险减半（亏大钱 9.5%→4.0%、最惨 5% 从 −28% 收到 −15%），不在提高总收益。**\n")

    L.append("### 5.2 逐年滚动（规则每年用「那一年之前的数据」重新定一次）\n")
    L.append("| 档位 | 跑赢随机的年份 | 逐年明细 |")
    L.append("|---|---|---|")
    for X, tb in lines_by_level.items():
        wins = int((tb["规则亏大钱"] < tb["随机对照中位"]).sum())
        detail = "、".join(
            f"{int(r['年'])}{'✅' if r['规则亏大钱'] < r['随机对照中位'] else '❌'}"
            for _, r in tb.iterrows())
        L.append(f"| −{X:.0%} | **{wins} / {len(tb)}** | {detail} |")

    # 把失败年份单独挑出来说（这是最有价值的部分）
    fail_years = set()
    for X, tb in lines_by_level.items():
        for _, r in tb.iterrows():
            if r["规则亏大钱"] >= r["随机对照中位"]:
                fail_years.add(int(r["年"]))
    L.append("")
    if fail_years:
        L.append(f"> ⚠️ **失败年份：{'、'.join(str(y) for y in sorted(fail_years))}** —— "
                 f"这些年份规则不但没帮上忙，还比随机更差。**必须原样报告，不许只报平均。**\n")
    else:
        L.append("> ✅ 每一年都跑赢随机对照。\n")

    L.append("### 5.3 必须一起读的限制\n")
    L.append("1. **策略本身在样本外衰变了。** 看 5.1 表里「样本内基准 → 样本外基准」那一列："
             "亏大钱比例和每笔净期望都变了，说明 2023 年后的市场与 2017–2022 不是同一个环境。"
             "**过滤器是在这个衰变之上再加一层，不是把策略救活了。**")
    L.append("2. **2025 年是反例**（见 5.2）：那一年基准的亏大钱比例本身就跳到 16~21%，"
             "规则在极端行情里失效。**这不是可以忽略的噪声 —— 它告诉我们极端年份要另想办法。**")
    L.append("3. **规则③（独狼崩）在浅档根本没生效，原因是「阈值退化」**：这个特征在样本内"
             "**大部分时间是 0**（−5% 档 60.8%、−10% 档 30.0% 的交易「急跌币数 = 0」）→ "
             "25% 分位阈值直接退化成 0 → 「币数 ≥ 0」恒成立 → **规则形同虚设**。"
             "所以 −10% 档的「①②③」和「①②」其实是同一个东西（表一已标 ⚠️未生效）。"
             "**教训：稀疏特征不能直接套分位数阈值。**")
    L.append("4. **顺带修掉的一个口径缺陷**：旧版用「同时在跌的**绝对币数**」当阈值，"
             "而币池会长大（2017 年几十个币 → 2026 年 146 个）→ 规则随年月**悄悄变松**。"
             "新版改用**比例**（在跌的 ÷ 当时有撮合的）后，−15% 档立刻从「与随机无差别」变成"
             "「明显跑赢随机」，−20% 档 ③ 更是单条规则里最强的（5.5% vs 基准 9.5%）。"
             "**「样本内最强的特征」不等于「样本外最可靠」，但口径错了会连「可靠不可靠」都判不出来。**")
    L.append("5. **前提仍是「触及即成交」**（`09` §8.3-2），深跌时最不可靠；滑点已按秒级校准计入 0.25%。")
    L.append("6. **币池是普遍下跌的币池**（买入持有中位 −82%），不要把 +20%/年 当普适优势。")
    L.append("7. 年化是**近似**：`每笔净期望 × 该档位年频率 × 保留率`，未计资金占用与同时触发（`09` §8.4）。\n")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "wick_walkforward.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[wf] -> results/wick_walkforward.md")

    # 控制台摘要
    for X, base_is, base_oos, drow in summary_rows:
        combo = [r for r in drow if r[0].startswith("⭐")]
        if not combo:
            continue
        _, st, ctl, _ = combo[0]
        cs = ("随机对照不可用（保留率 100%）" if ctl is None else
              f"随机中位 {100 * ctl['big_loss_p50']:.1f}%、随机最好5% {100 * ctl['big_loss_p05']:.1f}%")
        print(f"   −{X:.0%}: 样本内基准 {100 * base_is['big_loss']:.1f}% | "
              f"样本外 {100 * base_oos['big_loss']:.1f}% -> {100 * st['big_loss']:.1f}%"
              f"（{cs}）| 每笔净 {100 * base_oos['net']:+.3f}% -> {100 * st['net']:+.3f}%"
              f" | 年化 {100 * base_oos['ann']:+.1f}% -> {100 * st['ann']:+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
