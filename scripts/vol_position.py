#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · M1 步骤 4：验证「检索给的波动预期 → 调仓位」有没有价值

## 为什么做这个

`results/sim_eval.md` 两轮结论：
  · ❌ 检索**不能**预测方向（8 个定义 × 4 个周期，IC 全线 ≈ 0）
  · ✅ 检索**能**预测"会不会有大波动"（IC(命中+5%) 0.24，离散度 0.976→0.838）
  · ⚠️ 而且那 0.24 几乎全来自「波幅」一个数字，不是形态

项目原则早就写着：**检索结果调【仓位】，不调【方向】**。
→ 所以现在要验证的就一句话：**"知道会不会有大波动"能不能换成钱？**

## 怎么做（大白话）

同一批交易，只在**仓位大小**上做文章：预期波动大 → 仓位给小；预期波动小 → 仓位给大。
然后看"**每单位风险的收益**"（夏普）有没有变好。

两套口径：
  · **表一 横截面再分配**：每期总仓位固定，只是在不同币之间挪（低波动预期多给）
  · **表二 时间序列总仓位**：整体仓位随"当前该有多小心"升降

## ⚠️ 关键是必须防自欺欺人

"降仓位"本身就会让风险变小 —— 这不需要任何本事。所以必须有对照：

| 方案 | 干什么 | 它赢了说明什么 |
|---|---|---|
| A 等权 / 固定 | 不调仓 | 基准 |
| B 按**当前波幅** | **不需要检索**，只看这段 K 线自己的波动 | 那检索就是白做 |
| C 按**检索的波动预期** | 用历史相似段后来的波动 | 本次要验证的 |
| D **随机置换** C 的权重 | 权重分布一模一样，只是乱分配 | 改善来自"分散"不是"预测" → 白搭 |
| E ⭐ **上帝视角** | 按**真实未来波动**调，不可能实现 | **天花板：连它都提升很小 → 整条路不值得走** |

## 口径

- 横截面 = **按天分桶**（每个币每天取最后一根窗口），每 N 天一个决策时刻
- 权重归一化到合计 = 1（**总仓位恒定 → 比较才公平**）
- 未来波动 V = 24h 区间幅度（`fwd_max − fwd_min`，%）
- 检索邻居：**时间硬隔离**（同 `sim_eval.py`）；`--amp_w` 控制波幅在检索里占多少

用法：
  python vol_position.py --index 15m_100 --every 1 --amp_w 27
输出：
  results/vol_position.md · results/vol_position.csv · results/vol_position.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sim_eval as S

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DAY_MS = 86_400_000
BAR_MS = 15 * 60_000
HZ_MS = 96 * BAR_MS              # 24h（与 meta 的 fwd_* 一致）
AMP_W = {"0": 0.0, "11": 4.0, "27": 12.0, "50": 32.0, "75": 96.0}


class AugSearch:
    """把「波幅」拼成第 65 维的**增广向量**，于是
    `距离 = 形状距离 + w×(波幅差)²` 变成一次普通矩阵乘法，不用开中间张量。
    """

    def __init__(self, idx_dir: Path, amp_z: np.ndarray, w: float, device: str):
        self.device = device
        V = np.load(idx_dir / "vectors_shape.npy", mmap_mode="r")
        n, d = V.shape
        g = torch.empty((n, d + 1), dtype=torch.float32, device=device)
        step = 1_000_000
        for s in range(0, n, step):
            e = min(s + step, n)
            g[s:e, :d] = torch.from_numpy(np.asarray(V[s:e], np.float32)).to(device)
        col = np.sqrt(w) * amp_z if w > 0 else np.zeros(n, np.float32)
        g[:, d] = torch.from_numpy(col.astype(np.float32)).to(device)
        self.V = g
        self.nrm = (g * g).sum(1)                      # (N,) 预存模长
        del g

    def vectors(self, rows: np.ndarray) -> np.ndarray:
        out = np.empty((len(rows), self.V.shape[1]), np.float32)
        step = 200_000
        for s in range(0, len(rows), step):
            e = min(s + step, len(rows))
            idx = torch.from_numpy(rows[s:e]).to(self.device)
            out[s:e] = self.V[idx].cpu().numpy()
        return out

    def topk(self, Q: np.ndarray, pool: int, chunk: int, batch: int):
        n = self.V.shape[0]
        out = np.empty((len(Q), pool), np.int64)
        for s0 in range(0, len(Q), batch):
            e0 = min(s0 + batch, len(Q))
            qt = torch.from_numpy(Q[s0:e0]).to(self.device)
            qn = (qt * qt).sum(1)
            bv = torch.full((e0 - s0, pool), float("inf"), device=self.device)
            bi = torch.zeros((e0 - s0, pool), dtype=torch.long, device=self.device)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                blk = self.V[s:e]
                D = qn[:, None] + self.nrm[s:e][None, :] - 2.0 * (qt @ blk.t())
                D.clamp_min_(0)
                k = min(pool, D.shape[1])
                cv, ci = torch.topk(D, k, dim=1, largest=False, sorted=True)
                ci = ci + s
                if s == 0:
                    bv, bi = cv, ci
                else:
                    av = torch.cat([bv, cv], 1); ai = torch.cat([bi, ci], 1)
                    _, o = torch.topk(av, pool, dim=1, largest=False, sorted=True)
                    bv = torch.gather(av, 1, o); bi = torch.gather(ai, 1, o)
                del D, cv, ci
            out[s0:e0] = bi.cpu().numpy()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="15m_100")
    ap.add_argument("--every", type=int, default=1, help="每 N 天一个决策时刻")
    ap.add_argument("--amp_w", default="27", choices=list(AMP_W), help="波幅在检索距离里的占比")
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--pool", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=300_000)
    ap.add_argument("--min_coins", type=int, default=50)
    ap.add_argument("--r_cap", type=float, default=1.0,
                    help="单期收益绝对值上限（1.0 = ±100%），剔除冰点价/换币异常")
    ap.add_argument("--v_cap", type=float, default=100.0,
                    help="未来 24h 区间幅度上限（%），同上")
    ap.add_argument("--seed", type=int, default=20260913)
    args = ap.parse_args()
    w_amp = AMP_W[args.amp_w]

    t0 = time.time()
    idx_dir = ROOT / "data" / "index" / args.index
    m = pd.read_parquet(idx_dir / "meta.parquet",
                        columns=["symbol", "t0", "t1", "amp_pct",
                                 "fwd_ret", "fwd_max", "fwd_min"])
    codes = m["symbol"].cat.codes.to_numpy()
    sym_all = np.asarray(m["symbol"].cat.categories, dtype=object)
    t0a = m["t0"].to_numpy(); t1a = m["t1"].to_numpy()
    amp = m["amp_pct"].to_numpy(np.float64)
    R = m["fwd_ret"].to_numpy(np.float64)
    V = (m["fwd_max"] - m["fwd_min"]).to_numpy(np.float64) * 100.0
    # ⚠️ 收益截断（项目纪律）：LUNAUSDT 之类出现 fwd_ret = 1.4e7%、区间幅度 4.8e7%，
    #    0.02% 的异常行会把均值彻底带偏。检索用的 V 也必须先截断。
    n_raw = len(R)
    V = np.clip(V, 0.0, args.v_cap)
    exc = R - m.groupby("t1", observed=True)["fwd_ret"].transform("mean").to_numpy()
    la = np.log(np.maximum(amp, 1e-6))
    amp_z = (la - np.nanmean(la)) / np.nanstd(la)
    print(f"[vol_pos] {args.index} · 窗口 {len(m):,} · 波幅占比 {args.amp_w}%")

    # ---------------- 横截面：按天分桶，每币每天取最后一根窗口 ----------------
    day = (t1a // DAY_MS).astype(np.int64)
    key = day * 1_000_000 + codes                       # 天 + 币
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cut = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    seg = np.r_[cut, len(ks)]
    rows_daily = np.array([order[a:b][-1] for a, b in zip(seg[:-1], seg[1:])])
    gday = day[rows_daily]
    all_days = np.unique(gday)
    sel_days = all_days[:: args.every]
    keep = np.isin(gday, sel_days)
    rows_daily = rows_daily[keep]; gday = gday[keep]

    gmap = {d: k for k, d in enumerate(sel_days)}
    gid = np.array([gmap[int(d)] for d in gday], np.int32)
    G = len(sel_days)
    # 只保留币数够的横截面
    cnt = np.bincount(gid, minlength=G)
    goodg = np.flatnonzero(cnt >= args.min_coins)
    gm2 = {g: k for k, g in enumerate(goodg)}
    selm = np.isin(gid, goodg)
    rows_daily = rows_daily[selm]
    gid = np.array([gm2[int(g)] for g in gid[selm]], np.int32)
    G = len(goodg)
    print(f"[vol_pos] 天 {len(all_days):,} → 决策时刻 {G:,}"
          f"（每 {args.every} 天）· 检索查询 {len(rows_daily):,} 条")

    # ---------------- 检索：邻居的波动预期（向量化） ----------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bs = AugSearch(idx_dir, amp_z, w_amp, dev)
    Qa = bs.vectors(rows_daily)
    if w_amp > 0:
        Qa[:, -1] = np.sqrt(w_amp) * amp_z[rows_daily]

    nb_vol = np.full(len(rows_daily), np.nan)
    nb_ret = np.full(len(rows_daily), np.nan)
    bsize = 2048
    for s in range(0, len(rows_daily), bsize):
        e = min(s + bsize, len(rows_daily))
        sub = rows_daily[s:e]
        bi = bs.topk(Qa[s:e], args.pool, args.chunk, args.batch)   # (B,pool) 按距离升序
        B, P = bi.shape
        c0 = t0a[bi]; c1 = t1a[bi]
        okm = (c1 <= t0a[sub][:, None]) | (c0 >= (t1a[sub] + HZ_MS)[:, None])
        rank = np.argsort(~okm, axis=1, kind="stable")            # 保留的排前面，且维持距离序
        sel = np.take_along_axis(bi, rank[:, : args.topk], 1)
        valid = np.take_along_axis(okm, rank[:, : args.topk], 1)
        vv = V[sel].copy(); vv[~valid] = np.nan
        rr = exc[sel].copy(); rr[~valid] = np.nan
        with np.errstate(all="ignore"):
            nb_vol[s:e] = np.nanmedian(vv, axis=1)
            nb_ret[s:e] = np.nanmean(rr, axis=1)
        if (s // bsize) % 10 == 0 or e == len(rows_daily):
            print(f"  检索 {e:,}/{len(rows_daily):,}  用时 {time.time() - t0:.0f}s")
        del bi, c0, c1, okm, rank, sel, valid, vv, rr
    bad = int(np.sum(~np.isfinite(nb_vol)))
    print(f"[vol_pos] 检索完成（邻居不足 10 的 {bad} 条）用时 {time.time() - t0:.0f}s")

    df = pd.DataFrame({
        "g": gid, "row": rows_daily, "sym": sym_all[codes[rows_daily]],
        "amp": amp[rows_daily], "nb_vol": nb_vol, "nb_ret": nb_ret,
        "R": R[rows_daily], "V": V[rows_daily],
    })
    df["RA"] = df["R"] / df["V"].replace(0, np.nan)
    cleanc = df["R"].notna() & df["V"].notna() & (df["V"] > 1e-9) & df["nb_vol"].notna()
    clean = cleanc & (df["R"].abs() <= args.r_cap) & (df["V"] <= args.v_cap)
    d = df[clean]
    print(f"[vol_pos] 有效 (币,天) 样本 {len(d):,}（截断剔除 {int(cleanc.sum()) - len(d):,}）"
          f"  平均每期 {len(d)/G:.0f} 个币")

    # ---------------- 诊断：预期波动 vs 未来真实波动 / 单位风险收益 ----------------
    diag = {
        "IC(检索预期波动 → 未来真实波动)": S.spearman(d["nb_vol"], d["V"]),
        "IC(当前波幅 → 未来真实波动)": S.spearman(d["amp"], d["V"]),
        "IC(检索预期波动 → 单位风险收益)": S.spearman(d["nb_vol"], d["RA"]),
        "IC(当前波幅 → 单位风险收益)": S.spearman(d["amp"], d["RA"]),
        "IC(检索预期收益 → 未来真实收益)": S.spearman(d["nb_ret"], d["R"]),
    }

    # 五分位：低波动预期那批，后来收益/波动各是多少
    quint = []
    for nm, col in (("当前波幅", "amp"), ("检索波动预期", "nb_vol"), ("真实未来波动", "V")):
        qq = pd.qcut(d[col].rank(method="first"), 5, labels=False)
        gg = d.groupby(qq).agg(Rm=("R", "mean"), Vm=("V", "mean"))
        for grp, row in gg.iterrows():
            quint.append({
                "按什么分组": nm, "组": int(grp) + 1,
                "平均未来收益(%)": round(float(row["Rm"]) * 100, 3),
                "平均未来波动(%)": round(float(row["Vm"]), 3),
                "单位风险收益": (round(float(row["Rm"]) * 100 / float(row["Vm"]), 4)
                                 if row["Vm"] else np.nan),
            })
    quint = pd.DataFrame(quint)

    # ---------------- 表一：横截面再分配 ----------------
    def inv(x):
        x = np.asarray(x, float)
        x = np.where(np.isfinite(x) & (x > 1e-6), x, np.nan)
        med = np.nanmedian(x)
        x = np.where(np.isfinite(x), x, med if np.isfinite(med) else 1.0)
        return 1.0 / x

    rng = np.random.default_rng(args.seed)
    schemes = ["A 等权（不调仓）", "B 按当前波幅", "C 按检索波动预期",
               "D 随机置换(对照)", "E 上帝视角(上限)"]
    port1 = {k: [] for k in schemes}
    delta = {k: [] for k in schemes}          # 相对等权的超额（检验"真本事 vs 运气"）
    period_rows = []
    tot_raw = {k: [] for k in ["A 固定仓位", "B 按当前波幅", "C 按检索波动预期",
                               "E 上帝视角(上限)"]}
    for g, sub in d.groupby("g", sort=True):
        n = len(sub)
        if n < args.min_coins:
            continue
        w = {"A 等权（不调仓）": np.ones(n),
             "B 按当前波幅": inv(sub["amp"].to_numpy()),
             "C 按检索波动预期": inv(sub["nb_vol"].to_numpy()),
             "E 上帝视角(上限)": inv(sub["V"].to_numpy())}
        w["D 随机置换(对照)"] = rng.permutation(w["C 按检索波动预期"])
        Rw = sub["R"].to_numpy()
        base = float(Rw.mean())
        for k in schemes:
            v = np.asarray(w[k], float)
            v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
            v = v / v.sum() if v.sum() > 0 else np.full(n, 1.0 / n)
            pk = float((v * Rw).sum())
            port1[k].append(pk)
            delta[k].append(pk - base)
        prow = {"g": int(g), "mkt": base}
        prow.update({f"Δ {k}": delta[k][-1] for k in schemes})
        period_rows.append(prow)
        # 表二：总仓位随时间升降（用全市场平均预期波动），**归一化仓位而不是收益**
        for k, col in (("A 固定仓位", None), ("B 按当前波幅", "amp"),
                       ("C 按检索波动预期", "nb_vol"), ("E 上帝视角(上限)", "V")):
            s = 1.0 if col is None else float(np.nanmean(inv(sub[col].to_numpy())))
            tot_raw[k].append((s, base))

    # 逐期明细 + 分年度（**铁律：样本高度自相关，不能只按期数估置信区间**）
    per = pd.DataFrame(period_rows)
    per["date"] = pd.to_datetime(sel_days[per["g"].to_numpy()] * DAY_MS, unit="ms", utc=True)
    per["year"] = per["date"].dt.year
    yrows = []
    for y, s in per.groupby("year", sort=True):
        r = {"年": int(y), "期数": len(s), "市场收益(%)": 100 * s["mkt"].mean()}
        for k in schemes:
            r[f"超额{k[0]}(%)"] = 100 * s[f"Δ {k}"].mean()
        yrows.append(r)
    ytab = pd.DataFrame(yrows)
    ywin = {k: int((ytab[f"超额{k[0]}(%)"] > 0).sum()) for k in schemes}

    def stats(series, per_year):
        r = np.asarray(series, float)
        nav = np.cumprod(1 + r)
        dd = float((nav / np.maximum.accumulate(nav) - 1).min())
        return {"平均每期收益(%)": 100 * r.mean(),
                "年化收益(%)": 100 * r.mean() * per_year,
                "年化波动(%)": 100 * r.std() * np.sqrt(per_year),
                "夏普": (float(r.mean() / r.std() * np.sqrt(per_year)) if r.std() > 0 else np.nan),
                "最大回撤(%)": 100 * dd, "胜率": float((r > 0).mean())}

    per_year1 = 365.0 / args.every
    t1out = pd.DataFrame([dict(方案=k, **stats(v, per_year1)) for k, v in port1.items()])
    t1out["相对A的夏普变化"] = t1out["夏普"] - t1out.loc[0, "夏普"]
    # 相对等权的超额：正且 t 值大 = 这个倾斜真有本事；≈0 = 只是运气/分散
    dd_ = []
    for k in schemes:
        x = np.asarray(delta[k], float)
        x = x[np.isfinite(x)]
        tstat = float(x.mean() / x.std() * np.sqrt(len(x))) if x.std() > 0 else np.nan
        # 更保守的 t：每隔 5 期取 1 期（降低自相关把 t 值撑虚）
        xs = x[::5]
        tst = float(xs.mean() / xs.std() * np.sqrt(len(xs))) if len(xs) > 10 and xs.std() > 0 else np.nan
        dd_.append({"方案": k,
                    "超额(每期,%)": 100 * x.mean(),
                    "年化超额(%)": 100 * x.mean() * per_year1,
                    "t值": tstat,
                    "t值(隔5期)": tst,
                    "正超额年数": f"{ywin[k]}/{len(ytab)}",
                    "期数": len(x)})
    t1d = pd.DataFrame(dd_)

    # 表二：总仓位随时间升降（归一化**仓位**到平均 1，收益保持原量纲）
    t2out = []
    for k, pairs in tot_raw.items():
        s = np.array([p[0] for p in pairs], float)
        mk = np.array([p[1] for p in pairs], float)
        s = s / np.nanmean(s)
        t2out.append(dict(方案=k, **stats(s * mk, per_year1)))
    t2out = pd.DataFrame(t2out)
    t2out["相对A的夏普变化"] = t2out["夏普"] - t2out.loc[0, "夏普"]

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"{args.index}_aw{args.amp_w}_e{args.every}"
    t1out.to_csv(RESULTS / f"vol_position_{tag}.csv", index=False, encoding="utf-8-sig")
    t1d.to_csv(RESULTS / f"vol_position_skill_{tag}.csv", index=False, encoding="utf-8-sig")
    quint.to_csv(RESULTS / f"vol_position_quintile_{tag}.csv", index=False, encoding="utf-8-sig")
    ytab.to_csv(RESULTS / f"vol_position_byyear_{tag}.csv", index=False, encoding="utf-8-sig")
    per.to_csv(RESULTS / f"vol_position_periods_{tag}.csv", index=False, encoding="utf-8-sig")
    # 逐格明细落盘 → 以后改统计口径只需 pandas 几秒，不用重跑 20 分钟检索
    d[["g", "row", "sym", "amp", "nb_vol", "nb_ret", "R", "V"]].to_parquet(
        RESULTS / f"vol_position_cells_{tag}.parquet", index=False)
    (RESULTS / f"vol_position_{tag}.json").write_text(json.dumps({
        "index": args.index, "every": args.every, "amp_w": args.amp_w, "topk": args.topk,
        "n_cross_section": int(G), "n_query": int(len(rows_daily)),
        "n_valid_cell": int(len(d)), "avg_coins": float(len(d) / G),
        "periods_per_year": per_year1, "diag": diag,
        "table_cross": t1out.to_dict("records"), "table_skill": t1d.to_dict("records"),
        "table_total": t2out.to_dict("records"), "quintile": quint.to_dict("records"),
        "by_year": ytab.to_dict("records"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n=== 一、预期波动到底能不能预测未来（关键诊断）===")
    for k, v in diag.items():
        print(f"  {k:<32} {v:+.4f}")
    print("\n=== 二、按预测波动分 5 组，后来实际怎样 ===")
    print(quint.pivot_table(index="组", columns="按什么分组",
                            values=["平均未来收益(%)", "平均未来波动(%)", "单位风险收益"]
                            ).to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n=== 三、表一 横截面再分配（总仓位恒定）===")
    print(t1out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== 三之二、倾斜相对等权的超额（关键：是本事还是运气）===")
    print(t1d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== 三之三、分年度：每一年这个倾斜赚不赚（防自相关把 t 值撑虚）===")
    print(ytab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== 四、表二 总仓位随时间升降（平均仓位已归一）===")
    print(t2out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n（一年 {per_year1:.0f} 期；每期 {args.every} 天）")
    print(f"[vol_pos] -> results/vol_position_{tag}.csv  用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
