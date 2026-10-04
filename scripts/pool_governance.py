#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · 币池治理（coin pool governance）

## 要解决什么（大白话）

我们的"币池"是 200 个交易对，里面混着三种**不该上场**的成员：

1. **根本不是加密风险资产的**：`USDC`/`TUSD`/`BUSD`（稳定币）、`EUR`（欧元）、`PAXG`（黄金）
   → 它们几乎不动、收益≈0。任何"挑低风险标的"的规则都会自动挑中它们，
     于是方案的"低风险"其实来自**偷偷持有现金**，不是策略本事（2026-09-14 已实测）。
2. **已经死掉的**：交易所早就停了撮合，数据却还在发布（200 币里有 68 个这类样本）。
3. **冷到没法交易的**：成交额太低 → 滑点吃掉全部优势，且"形态"在无流动性时不可信。

## ⭐ 铁律：选币不许用未来信息（point-in-time）

上一轮算出"只买 BTC/ETH/TRX/BNB 夏普 **1.084**"，但那是**上帝视角**
——我是**知道它们活到了今天**才挑的这四只。看着漂亮，但实盘那一刻你并不知道。

所以本脚本改用**当时就能算出来的东西**：
  **每天用「过去 30 天的中位日成交额」给所有币排名，只买当时排在前面的。**

并且必须和**同样数量的随机选币**对照，否则分不清两件事：
  "选币有真本事" vs "只是少买了几只、结果碰巧好"。

## 对照设计（缺一不可）

| 对照 | 干什么 | 它赢了说明什么 |
|---|---|---|
| **随机固定 k 只**（抽一次，持有到底） | 同样只买 k 只，但**不按任何条件挑** | 抽 k 只本身就更好 → 流动性选币是白搭 |
| **随机每天换 k 只** | 每天都随机挑 k 只 | 分离"集中持有"与"轮换"的差别 |

⚠️ 三者的比较**必须落在同一批决策时刻上**，否则又是口径作弊。

## 用法

```
python pool_governance.py --stage liquidity      # 从 15m 数据算流动性表（几秒）
python pool_governance.py --stage report         # 出对照结果
python pool_governance.py --stage report --liq_floor 3e6 --tag floor3M
```

产物：
  `data/pool_liquidity.parquet`（币 × 天：日成交额 / 过去 30 天与 60 天中位成交额）
  `results/pool_governance*.csv` + `results/pool_governance.md`（结账页）
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
DAY = 86_400_000

# 非加密风险资产（稳定币 / 法币 / 贵金属）——必须有理由才能增删
NON_CRYPTO = ["USDCUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT", "PAXGUSDT"]
MAJORS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "TRXUSDT"]


def sharpe(x, per_year: float) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std() * np.sqrt(per_year)) if len(x) > 1 and x.std() > 0 else np.nan


def perf(r, per_year: float, label: str) -> dict:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {"方案": label, "期数": 0}
    nav = np.cumprod(1 + r)
    dd = float((nav / np.maximum.accumulate(nav) - 1).min())
    return {"方案": label, "期数": len(r),
            "年化收益(%)": 100 * r.mean() * per_year,
            "年化波动(%)": 100 * r.std() * np.sqrt(per_year),
            "夏普": sharpe(r, per_year),
            "最大回撤(%)": 100 * dd}


def block_boot_diff(a, b, per_year: float, n_boot: int = 1000, block: int = 10, seed: int = 7):
    """夏普差 b−a 的分块自助法（项目铁律：样本高度自相关，不能按期数估标准误）。"""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    nb = int(np.ceil(n / block))
    obs = sharpe(b, per_year) - sharpe(a, per_year)
    diffs = []
    for _ in range(n_boot):
        st = rng.integers(0, n, nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in st])
        idx = idx[idx < n]
        diffs.append(sharpe(b[idx], per_year) - sharpe(a[idx], per_year))
    d = np.asarray(diffs, float)
    p = float(2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))
    return obs, float(np.nanstd(d)), p


def block_boot_mean_diff(a, b, block: int = 10, n_boot: int = 2000, seed: int = 11):
    """「b 比 a 每期多赚多少」的分块自助法 t 值。比"夏普差"更直接回答'是不是运气'。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 20:
        return np.nan, np.nan, np.nan
    d = b - a
    rng = np.random.default_rng(seed)
    n = len(d)
    nb = int(np.ceil(n / block))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, n, nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in st])
        idx = idx[idx < n]
        boots[i] = d[idx].mean()
    se = float(np.std(boots))
    return float(d.mean()), se, (float(d.mean() / se) if se > 0 else np.nan)


# ---------------------------------------------------------------- 流动性表
def build_liquidity(force: bool = False) -> Path:
    out = DATA / "pool_liquidity.parquet"
    if out.exists() and not force:
        print(f"[pool] 流动性表已存在，跳过（--force 可重建）：{out}")
        return out
    files = sorted((DATA / "raw" / "15m").glob("*.parquet"))
    t0 = time.time()
    parts = []
    for i, f in enumerate(files, 1):
        d = pd.read_parquet(f, columns=["open_time", "close", "quote_volume"])
        df = pd.DataFrame({
            "day": (d["open_time"].to_numpy() // DAY).astype(np.int64),
            "qv": d["quote_volume"].to_numpy(np.float64),
            "close": d["close"].to_numpy(np.float64),
        })
        g = df.groupby("day", sort=True).agg(dvol=("qv", "sum"), close=("close", "last"))
        g["sym"] = f.stem
        parts.append(g.reset_index())
        if i % 50 == 0 or i == len(files):
            print(f"  读取 15m {i}/{len(files)}  用时 {time.time() - t0:.0f}s", flush=True)
    a = pd.concat(parts, ignore_index=True)
    print(f"[pool] 原始日频 {len(a):,} 行 · {a['sym'].nunique()} 个币 · 用时 {time.time() - t0:.0f}s")

    # ⭐ 关键一步：把"没有数据的那些天"补成 0，而不是不补。
    #    否则已死的币会一直停在"最后一次有数据的成交额"上，永远看不出它已经死了。
    dmin, dmax = int(a["day"].min()), int(a["day"].max())
    full = np.arange(dmin, dmax + 1)
    print(f"[pool] 补齐到连续日历：{pd.to_datetime(dmin * DAY, unit='ms').date()} ~ "
          f"{pd.to_datetime(dmax * DAY, unit='ms').date()}（{len(full)} 天）")
    wide = a.pivot_table(index="day", columns="sym", values="dvol", aggfunc="sum").reindex(full)
    dead_days = int(wide.isna().sum().sum())
    wide = wide.fillna(0.0)
    print(f"[pool] 被补成 0 的 (币,天) 格：{dead_days:,} / {wide.size:,} = "
          f"{100 * dead_days / wide.size:.1f}%  ← 这些就是'当天没有撮合'的日子")

    liq30 = wide.rolling(30, min_periods=10).median()
    liq60 = wide.rolling(60, min_periods=20).median()
    mc = a.pivot_table(index="day", columns="sym", values="close", aggfunc="last").reindex(full)
    L = pd.concat([wide.stack().rename("dvol"), liq30.stack().rename("liq30"),
                   liq60.stack().rename("liq60"), mc.stack().rename("close")], axis=1)
    L = L.reset_index()
    L.columns = ["day", "sym", "dvol", "liq30", "liq60", "close"]
    L.to_parquet(out, index=False)
    print(f"[pool] -> {out}  {len(L):,} 行 · 用时 {time.time() - t0:.0f}s")
    return out


# ---------------------------------------------------------------- 主对照
def report(args):
    per_year = 365.0
    cells = pd.read_parquet(args.cells)
    per = pd.read_csv(args.periods, usecols=["g", "date", "year"])
    # ⚠️ 坑（2026-09-14 实测）：pandas 2.x 的 to_datetime 可能给 **微秒**精度
    #    （datetime64[us]），此时 `astype("int64")//1_000_000` 会算成"秒"再除以一天 → 天数变成 17。
    #    → 改成与时区/精度都无关的算法：先转成无时区，再按整日相减。
    _t = pd.to_datetime(per["date"], utc=True).dt.tz_localize(None)
    per["day"] = (_t - pd.Timestamp("1970-01-01")).dt.days.astype(np.int64)
    g2day = dict(zip(per["g"], per["day"]))
    g2year = dict(zip(per["g"], per["year"]))
    cells["day"] = cells["g"].map(g2day)
    print(f"[pool] 逐格明细 {len(cells):,} 行 · {cells['g'].nunique()} 期 · {cells['sym'].nunique()} 个币")

    L = pd.read_parquet(DATA / "pool_liquidity.parquet", columns=["day", "sym", "dvol", "liq30", "liq60"])
    d = cells.merge(L, on=["day", "sym"], how="left")
    miss = float(d["liq30"].isna().mean())
    print(f"[pool] 流动性对齐：缺 liq30 的 {100 * miss:.2f}% 行（新上市不足 10 天，或 15m 数据缺失）")
    # ⚠️ 静默失效防护（`04` §2.2②）：日期换算口径写错时不会报错，只会 100% 对不上 → 显式拦住
    if miss > 0.5:
        raise SystemExit(
            f"[pool] ✗ 对齐失败：{100 * miss:.1f}% 拿不到 liq30。\n"
            f"        cells 的 day 范围 {int(d['day'].min())}~{int(d['day'].max())}\n"
            f"        流动性表 day 范围 {int(L['day'].min())}~{int(L['day'].max())}\n"
            f"        → 检查日期换算口径（精度/时区），不要继续往下跑。")

    base = d[~d["sym"].isin(NON_CRYPTO)].copy()
    n_crypto = d["sym"].nunique()
    print(f"[pool] 剔非加密风险资产：{n_crypto} → {base['sym'].nunique()} 个币（剔掉 {len(d) - len(base):,} 行）")

    # 每个决策时刻：按**当时可见**的 liq30 排名（只用过去 30 天）
    gb = base.groupby("g", sort=True)
    base["liq_pct"] = gb["liq30"].rank(pct=True, na_option="keep")     # 1.0 = 当时成交额最高
    base["liq_rk"] = gb["liq30"].rank(ascending=False, method="first", na_option="keep")
    base["liq60_rk"] = gb["liq60"].rank(ascending=False, method="first", na_option="keep")
    base["dvol_rk"] = gb["dvol"].rank(ascending=False, method="first", na_option="keep")
    keep_g = set(gb.size()[gb.size() >= args.min_coins].index)

    # ⭐ 两个矩阵必须用**同一套期 × 币**：一个含 5 个非加密资产（旧口径），一个不含
    allk = d[d["g"].isin(keep_g)].copy()
    Rall = allk.pivot_table(index="g", columns="sym", values="R")
    base = base[base["g"].isin(keep_g)].copy()
    Rm = base.pivot_table(index="g", columns="sym", values="R")
    _ix, _cl = Rm.index, Rm.columns

    def wide(frame, col, ix=_ix, cl=_cl):
        return (frame.pivot_table(index="g", columns="sym", values=col)
                .reindex(index=ix, columns=cl).to_numpy(float))

    Lp = wide(base, "liq_pct")
    Lr = wide(base, "liq_rk")
    L60 = wide(base, "liq60_rk")
    Ld = wide(base, "dvol_rk")
    Lq = wide(base, "liq30")
    Rmat = Rm.to_numpy(float)
    # ⚠️ 含非加密资产的那张矩阵**必须保留它自己的 200 列**。
    #    上一版把它 reindex 到 base 的 195 列 → 5 个非加密资产被悄悄丢掉，
    #    于是"A0 含非加密"和"A 剔非加密"跑出一模一样的数字（假对照）。**静默失效，必须防。**
    RallM = Rall.reindex(index=_ix).to_numpy(float)
    syms = np.asarray(_cl, dtype=object)
    gidx = _ix.to_numpy()
    n_per, n_sym = Rmat.shape
    finite = np.isfinite(Rmat)
    finite_all = np.isfinite(RallM)
    gmap = {int(gg): i for i, gg in enumerate(gidx)}
    print(f"[pool] {n_per} 期 × {n_sym} 币 · 每期平均 {finite.sum(1).mean():.0f} 个币"
          f"（含非加密资产时 {finite_all.sum(1).mean():.0f} 个）")

    def series_of(mask, min_pick: int | None = None):
        """掩码 → (每期等权收益, 每期可用只数)。向量化。"""
        mp = args.min_pick if min_pick is None else min_pick
        m = np.asarray(mask, bool) & finite
        cnt = m.sum(1)
        tot = np.where(m, np.nan_to_num(Rmat, nan=0.0), 0.0).sum(1)
        s = np.where(cnt >= mp, tot / np.maximum(cnt, 1), np.nan)
        return s, cnt

    def ew_of(mat, mask, min_pick: int | None = None):
        """给定矩阵的等权组合（用于 A0 这种列集合不同的情况）"""
        mp = args.min_pick if min_pick is None else min_pick
        m = np.asarray(mask, bool)
        cnt = m.sum(1)
        tot = np.where(m, np.nan_to_num(mat, nan=0.0), 0.0).sum(1)
        return np.where(cnt >= mp, tot / np.maximum(cnt, 1), np.nan), cnt

    names, series, counts = [], [], []
    def add(label, mask):
        s, c = series_of(mask)
        names.append(label); series.append(s); counts.append(c)

    def add_raw(label, s, c):
        names.append(label); series.append(s); counts.append(c)

    add_raw("A0 全池·含非加密(旧口径)", *ew_of(RallM, finite_all))
    add("A 全池·剔非加密", finite)
    add(f"F 全池·剔冷清(liq30<{args.liq_floor:.0e})", finite & (Lq >= args.liq_floor))
    add("B1 成交额最高25%", finite & (Lp >= 0.75))
    add("B2 成交额最高50%", finite & (Lp >= 0.50))
    add("B3 成交额最低25%", finite & (Lp <= 0.25))
    add("C 前20只(每日按成交额)", finite & (Lr <= 20))
    add("Cx 前20只·剔除大盘4只", finite & (Lr <= 20) & ~np.isin(syms, MAJORS)[None, :])
    add("C4 前4只(每日按成交额)", finite & (Lr <= 4))
    add("E 固定大盘4只(上帝视角)", np.ones_like(Rmat, bool) & np.isin(syms, MAJORS)[None, :])

    cov = pd.DataFrame([{"方案": n, "每期平均持仓只数": float(np.mean(c)),
                         "非空期数": int(np.isfinite(s).sum())}
                        for n, s, c in zip(names, series, counts)])
    t1 = pd.DataFrame([perf(s, per_year, n) for n, s in zip(names, series)])
    # 基准用「A 全池·剔非加密」——它 2560 期**全都有**，不会因为别处期数少而不可比
    ref_name = "A 全池·剔非加密"
    ref = np.asarray(series[names.index(ref_name)], float)
    t1["夏普差 vs A全池"] = t1["夏普"] - t1.loc[t1["方案"] == ref_name, "夏普"].iloc[0]
    t1 = t1.merge(cov, on="方案")
    print("\n=== 表一 不同币池的收益与风险（等权、每期）===")
    print(t1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---------------- 关键对照：按流动性选 k 只 vs 随机选 k 只 ----------------
    rng = np.random.default_rng(args.seed)

    def topk_mask(rk, k, exclude_majors=False):
        m = finite & (np.asarray(rk, float) <= k)
        if exclude_majors:
            m = m & ~np.isin(syms, MAJORS)[None, :]
        return m

    def evaluate_k(k, rk=None, exclude_majors=False, n_rand=None):
        """返回 (流动性策略夏普, 随机固定k只夏普数组, 随机每天换k只夏普, 共同期数掩码)"""
        rk = Lr if rk is None else rk
        n_rand = args.n_rand if n_rand is None else n_rand
        s_liq, _ = series_of(topk_mask(rk, k, exclude_majors))
        common = np.isfinite(s_liq)
        avail = np.flatnonzero(finite.sum(1) >= k)
        # ① 随机固定 k 只（抽一次持有到底）
        sh_fixed = np.empty(n_rand)
        for j in range(n_rand):
            if exclude_majors:
                cand = np.flatnonzero(~np.isin(syms, MAJORS))
                pick = rng.choice(cand, size=k, replace=False)
            else:
                pick = rng.choice(n_sym, size=k, replace=False)
            m = np.zeros_like(finite)
            m[:, pick] = True
            s, _ = series_of(m & finite)
            sh_fixed[j] = sharpe(s[common], per_year)
        # ② 随机每天换 k 只
        m = np.zeros_like(finite)
        for i in avail:
            cand = np.flatnonzero(finite[i])
            if exclude_majors:
                cand = cand[~np.isin(syms[cand], MAJORS)]
            if len(cand) >= k:
                m[i, rng.choice(cand, size=k, replace=False)] = True
        s_rot, _ = series_of(m)
        return (sharpe(s_liq[common], per_year), sh_fixed,
                sharpe(s_rot[common], per_year), s_liq, common)

    print(f"\n=== 表二 关键对照：随机抽 k 只（{args.n_rand} 次）vs 按流动性选 k 只 ===")
    print("    （三者都在**同一批决策时刻**上比较）")
    ctrl_rows = []
    for k in (4, 20, 50):
        sh_liq, sh_fixed, sh_rot, s_liq, common = evaluate_k(k)
        pct = float(100 * np.nanmean(sh_fixed < sh_liq))
        ctrl_rows.append({
            "每次买几只": k, "共同期数": int(common.sum()),
            "随机固定k只_中位": float(np.nanmedian(sh_fixed)),
            "随机固定k只_5%": float(np.nanpercentile(sh_fixed, 5)),
            "随机固定k只_95%": float(np.nanpercentile(sh_fixed, 95)),
            "随机每天换k只": sh_rot, "按流动性前k只": sh_liq,
            "流动性策略百分位(%)": pct})
        print(f"  k={k:3d} | 随机固定 {np.nanmedian(sh_fixed):+.3f} "
              f"[5% {np.nanpercentile(sh_fixed, 5):+.3f}, 95% {np.nanpercentile(sh_fixed, 95):+.3f}]"
              f" | 随机每天换 {sh_rot:+.3f} | **按流动性 {sh_liq:+.3f}** → 百分位 {pct:.0f}%")
    ctrl = pd.DataFrame(ctrl_rows)

    # ---------------- 表五 稳健性：k 扫描 / 换指标 / 换日成交额 / 剔除大盘 / 换手率 ----------------
    print("\n=== 表五 稳健性（这个结论是不是挑参数挑出来的）===")
    rob_rows = []
    for k in (5, 10, 20, 30, 40, 60, 100):
        sh_liq, sh_fixed, sh_rot, s_liq, common = evaluate_k(k, n_rand=min(args.n_rand, 200))
        sel = finite & (Lr <= k)
        turn = float(np.mean(1 - (sel[1:] & sel[:-1]).sum(1) / np.maximum((sel[1:] | sel[:-1]).sum(1), 1)))
        rob_rows.append({"k": k, "按流动性前k只": sh_liq,
                         "随机固定k只中位": float(np.nanmedian(sh_fixed)),
                         "随机95%分位": float(np.nanpercentile(sh_fixed, 95)),
                         "百分位(%)": float(100 * np.nanmean(sh_fixed < sh_liq)),
                         "成分日均换手率(%)": 100 * turn})
        print(f"  k={k:3d} | 按流动性 {sh_liq:+.3f} | 随机中位 {np.nanmedian(sh_fixed):+.3f} / "
              f"95% {np.nanpercentile(sh_fixed, 95):+.3f} | 百分位 {100*np.nanmean(sh_fixed < sh_liq):.0f}%"
              f" | 成分换手 {100*turn:.1f}%/天")
    for nm, rk in (("改用过去60天成交额", L60), ("改用当日成交额(不取中位)", Ld)):
        sh_liq, sh_fixed, _, _, _ = evaluate_k(20, rk=rk, n_rand=min(args.n_rand, 200))
        rob_rows.append({"k": 20, "按流动性前k只": sh_liq, "随机固定k只中位": float(np.nanmedian(sh_fixed)),
                         "随机95%分位": float(np.nanpercentile(sh_fixed, 95)),
                         "百分位(%)": float(100 * np.nanmean(sh_fixed < sh_liq)),
                         "成分日均换手率(%)": np.nan, "变体": nm})
        print(f"  【{nm}】前20只 {sh_liq:+.3f} | 随机中位 {np.nanmedian(sh_fixed):+.3f}"
              f" | 百分位 {100*np.nanmean(sh_fixed < sh_liq):.0f}%")
    sh_liq, sh_fixed, _, _, _ = evaluate_k(20, rk=Lr, exclude_majors=True,
                                           n_rand=min(args.n_rand, 200))
    rob_rows.append({"k": 20, "按流动性前k只": sh_liq, "随机固定k只中位": float(np.nanmedian(sh_fixed)),
                     "随机95%分位": float(np.nanpercentile(sh_fixed, 95)),
                     "百分位(%)": float(100 * np.nanmean(sh_fixed < sh_liq)),
                     "成分日均换手率(%)": np.nan, "变体": "前20只·剔除BTC/ETH/BNB/TRX"})
    print(f"  【前20只·剔除BTC/ETH/BNB/TRX】{sh_liq:+.3f} | 随机中位 {np.nanmedian(sh_fixed):+.3f}"
          f" | 百分位 {100*np.nanmean(sh_fixed < sh_liq):.0f}%")
    rob = pd.DataFrame(rob_rows)

    # ---------------- 分年度 ----------------
    years = np.array([g2year.get(int(x), -1) for x in gidx])
    yr_rows = []
    for y in sorted({int(v) for v in years if v > 2000}):
        sel = years == y
        row = {"年": y, "期数": int(sel.sum())}
        for n, s in zip(names, series):
            vv = np.asarray(s, float)[sel]
            row[n] = 100 * np.nanmean(vv) * per_year if np.isfinite(vv).any() else np.nan
        yr_rows.append(row)
    ytab = pd.DataFrame(yr_rows)
    print("\n=== 表三 分年度年化收益(%)（每一年是不是独立成立）===")
    print(ytab.to_string(index=False, float_format=lambda x: f"{x:+.0f}"))

    print(f"\n=== 表四 显著性（分块自助法，基准 =「A 全池·剔非加密」，2560 期全有）===")
    sig = []
    for n in names:
        if n == ref_name:
            continue
        s = np.asarray(series[names.index(n)], float)
        obs, se, p = block_boot_diff(ref, s, per_year)
        md, mse, mt = block_boot_mean_diff(ref, s)
        sig.append({"方案": n, "夏普差": obs, "p值(夏普差)": p,
                    "每期多赚(%)": 100 * md, "t值(每期多赚)": mt,
                    "结论": ("显著" if (p < 0.05 or (np.isfinite(mt) and abs(mt) > 2.5)) else "不显著")})
    sdf = pd.DataFrame(sig)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------- 表六 落地清单：哪些币长期排在前列 ----------------
    sel20 = finite & (Lr <= 20)
    med_liq = np.array([
        (float(np.nanmedian(Lq[:, j][sel20[:, j]])) / 1e6 if sel20[:, j].any() else np.nan)
        for j in range(n_sym)])
    # ⚠️ 三列必须在**排序之前**一起构造好；先 sort 再按位置赋一列会整体错位（已踩过）
    share = pd.DataFrame({"币": syms, "进过前20名的期数占比": sel20.mean(0),
                          "在榜期间中位成交额(百万USDT/天)": med_liq,
                          "有数据的期数": finite.sum(0)})
    share_out = share[share["进过前20名的期数占比"] > 0] \
        .sort_values("进过前20名的期数占比", ascending=False).reset_index(drop=True)
    print(f"\n=== 表六 长期排在前 20 名的币（共 {len(share_out)} 个进过榜）===")
    print(share_out.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    longest = max((int((finite[:, j]).sum()) for j in range(n_sym)), default=0)
    print(f"  最长上市 {longest} 期（约 {longest / 365:.1f} 年）· 进过榜的币 {len(share_out)} 个"
          f" / 样本内出现过的 {int((finite.sum(0) > 0).sum())} 个")

    lastday = base.groupby("sym")["day"].max()
    gmax = int(base["day"].max())
    n_dead = int((lastday < gmax - 30).sum())
    print(f"\n[pool] 旁证：剔非加密后 {base['sym'].nunique()} 个币里，"
          f"有 {n_dead} 个在期末前 30 天以上就没数据了（= 下架/停摆）")

    suf = f"_{args.tag}" if args.tag else ""
    for nm, tb in (("pool_governance", t1), ("pool_governance_control", ctrl),
                   ("pool_governance_robust", rob), ("pool_governance_byyear", ytab),
                   ("pool_governance_sig", sdf), ("pool_governance_coverage", cov),
                   ("pool_governance_coins", share_out)):
        tb.to_csv(RESULTS / f"{nm}{suf}.csv", index=False, encoding="utf-8-sig")
    (RESULTS / f"pool_governance{suf}.json").write_text(json.dumps(
        {"cells": args.cells, "periods": args.periods, "liq_floor": args.liq_floor,
         "n_rand": args.n_rand, "n_periods": int(n_per), "n_symbols": int(n_sym),
         "n_dead": n_dead,
         "table1": t1.to_dict("records"), "control": ctrl.to_dict("records"),
         "robust": rob.to_dict("records"), "by_year": ytab.to_dict("records"),
         "significance": sdf.to_dict("records")}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[pool] -> results/pool_governance{suf}.csv")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["liquidity", "report", "all"])
    ap.add_argument("--cells", default=str(RESULTS / "vol_position_cells_15m_100_aw27_e1.parquet"))
    ap.add_argument("--periods", default=str(RESULTS / "vol_position_periods_15m_100_aw27_e1.csv"))
    ap.add_argument("--liq_floor", type=float, default=1e6,
                    help="「冷清」门槛：过去 30 天中位日成交额低于它 → 剔除（USDT/天）")
    ap.add_argument("--min_pick", type=int, default=3, help="某期子集至少要有几只才算数")
    ap.add_argument("--min_coins", type=int, default=20, help="该期基准池至少要有几只")
    ap.add_argument("--n_rand", type=int, default=300, help="随机对照抽多少次")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()
    if args.stage in ("liquidity", "all"):
        build_liquidity(args.force)
    if args.stage in ("report", "all"):
        return report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
