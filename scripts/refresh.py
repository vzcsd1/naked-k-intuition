#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · 滚动补数据（rolling refresh）

## 要解决什么（大白话）

检索库是 2026-09-14 建的，数据停在 **2026-08-31**。今天是 09-17，库里凭空少了 16 天。
而且这个窟窿**每天自己长大一天** —— 今天差 16 天，下个月差 46 天。

界面上那个「⚡ 看此刻」只能查"此刻"，查不了"这半个月里随便哪一刻"。
所以必须把新行情**并进库**。

## 为什么是"增量"而不是"重建"

从零重建要重跑 9 年 × 20 亿根 1m 的聚合与 840 万窗口的归一化。
增量只需要重算**被改动的那批币**，其余币的分片原封不动搬过来。

⚠️ 但增量有个天然风险：**"没重算"和"算错了"在结果上长得一样**。
所以本脚本配了一个必须跑的动作 —— `verify`：拿改动前的**老窗口指纹**逐条比对，
证明"老数据一个字节都没动"。**跳过它，增量就等于没有证据。**

## 用法

```
python scripts/refresh.py --stage snapshot    # 改索引之前：存老窗口指纹（基准）
python scripts/refresh.py --stage fetch       # 下载缺口期「每日归档」1m
python scripts/refresh.py --stage merge       # 并进 data/raw/1m，产出 touched 清单
python scripts/refresh.py --stage derive      # 只重派生 touched 的 15m
python scripts/refresh.py --stage liquidity   # 重建流动性表
python scripts/refresh.py --stage index       # 增量扩建索引
python scripts/refresh.py --stage verify      # 事后体检（必须全绿）
python scripts/refresh.py --stage all         # 除 snapshot 外全跑（snapshot 通常单独先跑）
```

## 为什么走「每日归档」而不是月度归档

缺口落在**月中**，`data.binance.vision` 的月度 zip 要等下月初才生成（实测 2026-09 返回 404）。
但**每日归档是次日就有**（实测 09-01~09-16 全部 HTTP 200，各 1440 根，时间戳是**微秒**）。
项目里 `fetch_data.py fetchdays` 已经用过同一条路，这里沿用同一套路径与解析。

产物：
  `data/raw/1m_days/{币}/{币}-1m-{日期}.parquet`（每日原始，可断点续跑）
  `data/raw/1m/{币}.parquet`（合并后的全量）
  `data/touched_syms.txt`（本轮动过的币清单，供 derive / build 直接吃）
  `data/refresh_state.json`（本轮做了什么）
  `results/refresh_report.json`（verify 的结构化结果）
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
sys.path.insert(0, str(SCRIPTS))

CDN = "https://data.binance.vision"
DAY_MS = 86_400_000
MIN_MS = 60_000
COLS = ["symbol", "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "dt"]

DAYS_DIR = RAW / "1m_days"
STATE = DATA / "refresh_state.json"
TOUCHED = DATA / "touched_syms.txt"
SNAP_VEC = DATA / "_snap_index_before.parquet"
SNAP_LIQ = DATA / "_snap_liq_before.parquet"
SNAP_META = DATA / "_snap_index_before.json"

IDX_DIR = DATA / "index" / "15m_100_top20"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "lK-pangen/0.1 (research)"})
try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    SESSION.mount("https://", HTTPAdapter(
        max_retries=Retry(total=3, backoff_factor=0.6,
                          status_forcelist=[429, 500, 502, 503, 504]),
        pool_connections=32, pool_maxsize=32))
except Exception:                                                  # noqa: BLE001
    pass


def say(*a):
    print(*a, flush=True)


def ms2s(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            pass
    return {}


def save_state(**kw):
    st = load_state()
    st.update(kw)
    st["updated_at"] = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    STATE.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")


def head_of_1m(sym: str):
    """只读一列拿首末时间 —— 不必把整张表读进来。"""
    p = RAW / "1m" / f"{sym}.parquet"
    if not p.exists():
        return None
    t = pd.read_parquet(p, columns=["open_time"])["open_time"]
    return int(t.iloc[0]), int(t.iloc[-1]), len(t)


# ⚠️ 踩过的坑（2026-09-17）：`data/symbols.json` 里**没有 "1m" 这个键**
#    它只有 1h 与 1s（1m 那次下载发生在该功能落地之前，键从未写入）。
#    照着它取 1m 的首末时间会得到空列表 → max() 抛 ValueError。
#    → 真值来源改用 `data/raw/1m/{币}_run.json`：200 个小文件，秒级读完，
#      而且它正是 `fetch_data.py` 自己写的元数据，口径一致。
def sym_meta(sym: str) -> dict | None:
    p = RAW / "1m" / f"{sym}_run.json"
    if p.exists():
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            return {"symbol": sym, "first": int(j["first"]), "last": int(j["last"]),
                    "rows": int(j.get("rows", 0)), "raw": j}
        except Exception:                                          # noqa: BLE001
            pass
    h = head_of_1m(sym)
    if h is None:
        return None
    return {"symbol": sym, "first": h[0], "last": h[1], "rows": h[2], "raw": None}


def all_sym_meta() -> dict[str, dict]:
    out = {}
    for p in sorted((RAW / "1m").glob("*.parquet")):
        m = sym_meta(p.stem)
        if m:
            out[p.stem] = m
    return out


# ================================================================ stage: snapshot

def cmd_snapshot(args):
    """改索引**之前**，把老窗口的指纹存下来 —— 这是"老数据没被改动"的唯一证据。

    取样的窗口必须**后续走势已经完整**（t1 ≤ 库末尾 − 96 根 × 15 分钟），
    否则它们本来就会因为"新数据补上了未来"而合法变化，混进来会污染判断。
    """
    meta_p = IDX_DIR / "meta.parquet"
    if not meta_p.exists():
        sys.exit(f"没有索引 {meta_p}")
    m = pd.read_parquet(meta_p)
    m["symbol"] = m["symbol"].astype(str)
    n_all = len(m)
    old_last = int(m["t1"].max())

    hz_bars = 96                      # dst(32) × 3 —— 与 _build_one 里的 hz 一致
    safe_cut = old_last - hz_bars * 15 * MIN_MS
    body = m[m["t1"] <= safe_cut].reset_index(drop=True)
    say(f"[snapshot] 索引 {n_all:,} 窗口 · 末尾 {ms2s(old_last)}")
    say(f"[snapshot] 后续走势完整的窗口 {len(body):,}（t1 ≤ {ms2s(safe_cut)}）")

    rng = np.random.default_rng(args.seed)
    n_take = min(args.n_sample, len(body))
    pick = np.sort(rng.choice(len(body), size=n_take, replace=False))
    sub = body.iloc[pick].reset_index(drop=True)

    shp = np.load(IDX_DIR / "vectors_shape.npy", mmap_mode="r")
    rwp = np.load(IDX_DIR / "vectors_raw.npy", mmap_mode="r")
    if shp.shape[0] != n_all:
        sys.exit(f"✗ 索引不自洽：meta {n_all} 行 vs vectors {shp.shape[0]} 行")
    # ⚠️ body 是**按条件筛出来的**，它的行号 ≠ m 里的行号 → 必须映射回原位置
    body_idx = np.flatnonzero((m["t1"] <= safe_cut).to_numpy())
    rows = body_idx[pick].astype(np.int64)

    def sha_block(arr, idx):
        out = []
        for i in idx:
            out.append(hashlib.sha1(np.ascontiguousarray(arr[i]).tobytes()).hexdigest()[:16])
        return out

    sub = sub.copy()
    sub["row"] = rows
    sub["sha_shape"] = sha_block(shp, rows)
    sub["sha_raw"] = sha_block(rwp, rows)
    sub.to_parquet(SNAP_VEC, index=False)
    say(f"[snapshot] 抽样指纹 {len(sub):,} 条 -> {SNAP_VEC}")

    # 每币窗口数（用于事后判断"是不是只有该动的币动了"）
    per = m.groupby("symbol").size().sort_index()
    meta = {"n_windows": n_all, "old_last_t1": old_last, "safe_cut": safe_cut,
            "n_sample": int(len(sub)), "per_coin": {k: int(v) for k, v in per.items()},
            "shapes": list(shp.shape), "taken_at": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"}
    SNAP_META.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    L = pd.read_parquet(DATA / "pool_liquidity.parquet", columns=["day", "sym", "liq30", "liq60"])
    L.to_parquet(SNAP_LIQ, index=False)
    say(f"[snapshot] 流动性表快照 {len(L):,} 行 -> {SNAP_LIQ}")
    return 0


# ================================================================ stage: fetch

def _latest_archive_day(sym="BTCUSDT", max_back=8) -> str | None:
    """从今天往回找，第一个能下到的每日归档日期。

    用 GET(stream) 而不是 HEAD —— 部分 CDN 对 HEAD 的响应与 GET 不一致，
    探测手段本身不能成为"找不到归档"的假原因。
    """
    today = datetime.now(timezone.utc).date()
    for k in range(max_back):
        d = (today - timedelta(days=k)).strftime("%Y-%m-%d")
        url = f"{CDN}/data/spot/daily/klines/{sym}/1m/{sym}-1m-{d}.zip"
        try:
            r = SESSION.get(url, timeout=25, stream=True)
            code = r.status_code
            r.close()
            if code == 200:
                return d
        except Exception:                                          # noqa: BLE001
            continue
    return None


def _alive_syms(cut_ms: int) -> tuple[list[str], int]:
    metas = all_sym_meta()
    out = sorted(s for s, m in metas.items() if m["last"] >= cut_ms)
    return out, len(metas)


def cmd_fetch(args):
    # ---- 1. 缺口起止
    metas = all_sym_meta()
    if not metas:
        sys.exit("没有 data/raw/1m/*_run.json，无法确定缺口")
    last_all = max(m["last"] for m in metas.values())
    gap_start_ms = int(last_all) + MIN_MS
    gap_start_day = (gap_start_ms // DAY_MS) * DAY_MS

    latest = args.until or _latest_archive_day()
    if not latest:
        sys.exit("✗ 找不到可用的每日归档（最近 8 天全是 404）→ 该换数据源了")
    end_day = int(pd.Timestamp(latest, tz="UTC").value // 10**6)

    days = [datetime.fromtimestamp(d / 1000, timezone.utc).strftime("%Y-%m-%d")
            for d in range(int(gap_start_day), int(end_day) + 1, DAY_MS)]
    say("=" * 78)
    say(f"[fetch] 现有 1m 末尾 {ms2s(last_all)}")
    say(f"[fetch] 每日归档最新 {latest}")
    say(f"[fetch] 需要补 {len(days)} 天：{days[0]} ~ {days[-1]}")
    if not days:
        say("[fetch] 没有缺口，什么都不用做")
        save_state(days=[], fetched={})
        return 0

    cut = int(last_all) - 3 * DAY_MS          # 3 天内有数据的算"还活着"
    syms, n_all = _alive_syms(cut)
    say(f"[fetch] 候选币 {len(syms)} 个（末尾 3 天内还有数据）"
        f"；其余 {n_all - len(syms)} 个已停摆，跳过")
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        syms = [s for s in syms if s.upper() in want]
        say(f"[fetch] --symbols 过滤后 {len(syms)} 个")

    # ---- 2. 抓（(币,天) 落独立文件 → 天然断点续跑）
    DAYS_DIR.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor

    todo = [(s, d) for s in syms for d in days]
    say(f"[fetch] 待抓 {len(todo):,} 个 (币,天) ｜ 并发 {args.workers}")

    def one(t):
        sym, day = t
        p = DAYS_DIR / sym
        out = p / f"{sym}-1m-{day}.parquet"
        if out.exists() and out.stat().st_size > 0 and not args.force:
            return "skip", sym, day
        if out.exists() and out.stat().st_size == 0:
            zero[0] += 1        # ⭐ 0 字节 = 上次失败留下的空壳，不算"已有"，重下并覆盖
            #（不 unlink —— 沙箱内删不掉文件；to_parquet 会直接覆盖写）
        url = f"{CDN}/data/spot/daily/klines/{sym}/1m/{sym}-1m-{day}.zip"
        try:
            r = SESSION.get(url, timeout=90)
            if r.status_code != 200 or len(r.content) < 200:
                return ("miss" if r.status_code == 404 else "fail"), sym, day
            df = _parse_zip(r.content, sym)
            if df is None or df.empty:
                return "fail", sym, day
            p.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False, compression="zstd")
            return "ok", sym, day
        except Exception as e:                                     # noqa: BLE001
            return "fail", sym, f"{day}:{type(e).__name__}"

    t0 = time.time()
    cnt = {"ok": 0, "skip": 0, "miss": 0, "fail": 0}
    zero = [0]
    bad, missing = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for st, sym, day in ex.map(one, todo):
            cnt[st] += 1
            if st == "fail":
                bad.append(f"{sym} {day}")
            elif st == "miss":
                missing.append(f"{sym} {day}")
            if (cnt["ok"] + cnt["skip"] + cnt["miss"] + cnt["fail"]) % 400 == 0:
                say(f"    进度 {sum(cnt.values()):,}/{len(todo):,}  用时 {time.time()-t0:.0f}s")
    say(f"[fetch] 完成：新下载 {cnt['ok']} · 已有跳过 {cnt['skip']} · "
        f"归档不存在 {cnt['miss']} · 失败 {cnt['fail']}  用时 {time.time()-t0:.0f}s")
    if zero[0]:
        say(f"       ⚠️ 发现并重下 0 字节坏文件 {zero[0]} 个（上次失败残留，refresh.md §15.5）")
    if bad:
        say(f"       失败样例 {bad[:5]}")
    if missing:
        # 归档不存在 ≠ 出错：多半是该币在缺口期内被下架（后半段自然没有文件）
        mm = pd.DataFrame([{"sym": s, "day": d} for s, d in (x.split() for x in missing)])
        mm.to_csv(RESULTS / "_refresh_missing.csv", index=False, encoding="utf-8-sig")
        cnt["miss_syms"] = sorted(set(mm["sym"]))
        say(f"       归档缺失 {len(mm)} 个 (币,日) · 涉及 {mm['sym'].nunique()} 个币"
            f" · 清单 -> results/_refresh_missing.csv")
        say(f"       例：{', '.join(f'{s} 缺{n}天' for s, n in mm.groupby('sym').size().items())}"[:200])
    save_state(days=days, fetched=cnt, latest_archive_day=latest)
    return 0


def _parse_zip(content: bytes, sym: str):
    """与 fetch_data.py 完全同口径（含"微秒 → 毫秒"归一，2026-09 的文件就是微秒）。"""
    z = zipfile.ZipFile(io.BytesIO(content))
    names = [n for n in z.namelist() if n.endswith(".csv")]
    if not names:
        return None
    with z.open(names[0]) as fh:
        raw = fh.read().decode("utf-8", "replace")
    lines = [l for l in raw.splitlines() if l.strip()]
    if lines and not lines[0][:1].isdigit():
        lines = lines[1:]
    if not lines or len(lines[0].split(",")) < 11:
        return None
    names12 = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=names12,
                     on_bad_lines="skip", engine="c")
    df = df.drop(columns=["ignore"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open_time", "close"])
    if df.empty:
        return None
    if df["open_time"].iloc[0] > 1e14:                 # 微秒 → 毫秒
        df["open_time"] = df["open_time"] // 1000
        df["close_time"] = df["close_time"] // 1000
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df["trades"] = df["trades"].fillna(0).astype("int64")
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.insert(0, "symbol", sym)
    return df[COLS]


# ================================================================ stage: merge

def _pq_columns(path: Path) -> list[str]:
    """只读 parquet 的 schema（不读数据）—— 做列一致性检查用，代价接近 0。"""
    import pyarrow.parquet as pq
    return list(pq.ParquetFile(path).schema_arrow.names)


def cmd_merge(args):
    """把每日文件并进 raw/1m/{币}.parquet。去重、排序、校验后整文件重写。

    幂等：同一天并两次结果一样（drop_duplicates keep=last）。
    """
    if not DAYS_DIR.exists():
        sys.exit(f"没有 {DAYS_DIR}，先跑 --stage fetch")
    days = args.days.split(",") if args.days else None
    if days is None:
        days = load_state().get("days") or []
    if not days:
        sys.exit("不知道要并哪些天（refresh_state 里没有 days），用 --days 指定")

    syms = sorted(p.name for p in DAYS_DIR.iterdir() if p.is_dir())
    say(f"[merge] {len(syms)} 个币有每日文件 · 目标日期 {len(days)} 天")
    touched, stats, failed, nogain = [], [], [], []
    t0 = time.time()
    for i, sym in enumerate(syms, 1):
      # ⚠️ 逐币容错：沙箱的写入拦截会让**单个币**失败，不能让它拖垮其余 130 个。
      #    结构性错误（列不一致 / 排序坏掉）仍走 sys.exit（SystemExit 不被 except Exception 捕获）。
      try:
        got = [d for d in days if (DAYS_DIR / sym / f"{sym}-1m-{d}.parquet").exists()]
        if not got:
            continue
        src = RAW / "1m" / f"{sym}.parquet"
        if not src.exists():
            say(f"  ⚠️ {sym} 没有全量文件，跳过")
            continue
        new = pd.concat([pd.read_parquet(DAYS_DIR / sym / f"{sym}-1m-{d}.parquet")
                         for d in sorted(got)], ignore_index=True)
        tmp = src.with_name(src.name + ".tmp")

        # ⭐ 断点续跑 / 恢复：上次已经合并完、只差最后一步改名（进程被杀，或代码有 bug）。
        #    tmp 里是**已通过下面三道校验**的成品 → 直接沿用，省掉整轮重算（实测省 7 分钟）。
        comb_time = None
        if tmp.exists() and tmp.stat().st_mtime >= src.stat().st_mtime:
            try:
                t = pd.read_parquet(tmp, columns=["open_time"])["open_time"].to_numpy(np.int64)
                if t.size and bool(np.all(np.diff(t) > 0)):
                    comb_time = t
                    say(f"  {sym:<12} 沿用上次遗留的合并结果（断点续跑，{len(t):,} 行）")
            except Exception:                                      # noqa: BLE001
                comb_time = None

        col_old = _pq_columns(src)
        if list(new.columns) != col_old:
            sys.exit(f"✗ {sym} 列不一致：{list(new.columns)} vs {col_old}")
        old_time = pd.read_parquet(src, columns=["open_time"])["open_time"].to_numpy(np.int64)
        n_old = int(old_time.size)
        old_last = int(old_time[-1])

        if comb_time is None:
            old = pd.read_parquet(src)
            comb = pd.concat([old, new], ignore_index=True)
            comb = comb.drop_duplicates(subset="open_time", keep="last")
            comb = comb.sort_values("open_time", kind="mergesort").reset_index(drop=True)
            comb_time = comb["open_time"].to_numpy(np.int64)
            # ---- 三道校验（缺一条就停，别写出坏数据）
            if not bool(np.all(np.diff(comb_time) > 0)):
                sys.exit(f"✗ {sym} 合并后时间不升序或有重复")
            comb.to_parquet(tmp, index=False, compression="zstd")

        added = comb_time > old_last
        if not bool(added.any()):
            nogain.append(sym)
            say(f"  {sym:<12} 没有新数据（归档缺失？）")
            continue
        first_new = int(comb_time[added][0])
        n_new = int(added.sum())
        if first_new != old_last + MIN_MS:
            say(f"  ⚠️ {sym:<12} 新旧接缝不连续：老末 {ms2s(old_last)} → 新首 {ms2s(first_new)}"
                f"（缺 {(first_new - old_last) // MIN_MS - 1} 分钟）")
        n_after = int(comb_time.size)
        last_new = int(comb_time[-1])
        # 原子替换：先写临时文件再改名。中途失败时原文件保持完好（沙箱里改名可用）。
        os.replace(tmp, src)
        # ---- 元数据诚实化：{币}_run.json 的 rows/first/last 必须反映合并后的真实状态
        #      ⚠️ 但**必须保留原 fingerprint** —— 否则下次 `fetch_data.py fetch --interval 1m`
        #         会认为指纹变了、去重下全年月度 zip 并**覆盖**掉我们刚补的每日数据（数据损失）。
        rj = RAW / "1m" / f"{sym}_run.json"
        old_meta = {}
        if rj.exists():
            try:
                old_meta = json.loads(rj.read_text(encoding="utf-8"))
            except Exception:                                      # noqa: BLE001
                old_meta = {}
        old_meta.update({
            "symbol": sym, "interval": "1m", "rows": n_after,
            "first": str(int(comb_time[0])),
            "last": str(last_new),
            "refresh": {"by": "scripts/refresh.py", "days": sorted(got),
                        "added": n_new,
                        "at": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"},
        })
        rj.write_text(json.dumps(old_meta, indent=1, ensure_ascii=False), encoding="utf-8")
        touched.append(sym)
        stats.append({"sym": sym, "rows_before": n_old, "rows_after": n_after,
                      "added": n_new, "days": len(got),
                      "first_new": ms2s(first_new), "last": ms2s(last_new)})
        if i % 25 == 0 or i == len(syms):
            say(f"  进度 {i}/{len(syms)} · 已合并 {len(touched)} 个 · 用时 {time.time()-t0:.0f}s")
      except SystemExit:
        raise
      except Exception as e:                                       # noqa: BLE001
        failed.append(f"{sym}\t{type(e).__name__}\t{e}")
        say(f"  ❌ {sym:<12} 合并失败已跳过：{type(e).__name__}: {e}")
        continue

    if failed:
        (RESULTS / "_refresh_merge_failed.txt").write_text(
            "sym\ttype\tmsg\n" + "\n".join(failed) + "\n", encoding="utf-8")
        say(f"[merge] ⚠️ {len(failed)} 个币合并失败 -> results/_refresh_merge_failed.txt")
    if nogain:
        say(f"[merge] ℹ️ {len(nogain)} 个币没有新数据（该币在缺口期内已下架）："
            f"{', '.join(nogain[:10])}")

    # ---- touched 清单（供 derive / build 直接 @ 进来）
    TOUCHED.write_text(
        "# 本轮补齐缺口时被改动的币（refresh.py 自动生成，勿手改）\n"
        f"# 日期 {days[0]} ~ {days[-1]}\n" + "\n".join(touched) + "\n", encoding="utf-8")
    say(f"[merge] 合并完成：{len(touched)} 个币被追加数据 -> {TOUCHED}")
    tot_add = sum(s["added"] for s in stats)
    if stats:
        say(f"        追加合计 {tot_add:,} 根 1m · 单币中位 {int(np.median([s['added'] for s in stats])):,} 根")
        say(f"        例：{stats[0]['sym']} {stats[0]['rows_before']:,} → "
            f"{stats[0]['rows_after']:,} 根（新增 {stats[0]['added']:,}）")
    pd.DataFrame(stats).to_csv(RESULTS / "_refresh_merge.csv", index=False, encoding="utf-8-sig")

    # ---- 顺带把 data/symbols.json 的 "1m" 段补上（它原本只有 1h / 1s，见 script docstring 的坑）
    try:
        sj = DATA / "symbols.json"
        prev = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
        rows = []
        for sym, m in sorted(all_sym_meta().items()):
            rows.append({"symbol": sym, "interval": "1m",
                         "first_bar_utc": pd.Timestamp(m["first"], unit="ms", tz="UTC").isoformat(),
                         "last_bar_utc": pd.Timestamp(m["last"], unit="ms", tz="UTC").isoformat(),
                         "n_bars": int(m["rows"]),
                         "source": "monthly+refresh.py daily"})
        prev["1m"] = rows
        sj.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding="utf-8")
        say(f"[merge] data/symbols.json 的 1m 段已补齐（{len(rows)} 个币）")
    except Exception as e:                                         # noqa: BLE001
        say(f"[merge] ⚠️ symbols.json 更新失败（不致命）：{type(e).__name__}: {e}")

    save_state(touched=touched, merge_rows=int(tot_add))
    return 0


# ================================================================ stage: derive / liquidity / index

def cmd_derive(args):
    import build_index as BI

    ns = argparse.Namespace(interval=args.interval, workers=args.workers,
                            symbols=args.symbols or f"@{TOUCHED}", force=True)
    if not (args.symbols or TOUCHED.exists()):
        sys.exit(f"没有 {TOUCHED}，先跑 --stage merge（或用 --symbols 指定）")
    say(f"[derive] 强制重派生：{ns.symbols}")
    return BI.cmd_derive(ns)


def cmd_liquidity(args):
    import pool_governance as PG

    say("[liquidity] 重建流动性表（含新日期）")
    return PG.build_liquidity(force=True) and 0


def cmd_index(args):
    import build_index as BI

    ns = argparse.Namespace(interval=args.interval, win=args.win, stride=args.stride,
                            dst=args.dst, force=False,
                            force_syms=args.force_syms or f"@{TOUCHED}",
                            out_dir=args.out_dir,
                            pool_top=args.pool_top, pool_delay=1, liq_floor=0.0)
    if not (args.force_syms or TOUCHED.exists()):
        sys.exit(f"没有 {TOUCHED}，先跑 --stage merge")
    say(f"[index] 增量重建：force_syms={ns.force_syms}")
    return BI.cmd_build(ns)


# ================================================================ stage: verify

def _sha_rows(path: Path, rows):
    arr = np.load(path, mmap_mode="r")
    return [hashlib.sha1(np.ascontiguousarray(arr[i]).tobytes()).hexdigest()[:16] for i in rows]


def cmd_verify(args):
    ok_all = True
    rep = {}
    say("=" * 78)
    say("事后体检（每一项都必须通过，否则增量的结论作废）")
    say("=" * 78)

    # ---------- 1. 索引自洽：三个文件行数/顺序必须一致 ----------
    m = pd.read_parquet(IDX_DIR / "meta.parquet")
    m["symbol"] = m["symbol"].astype(str)
    shp = np.load(IDX_DIR / "vectors_shape.npy", mmap_mode="r")
    rwp = np.load(IDX_DIR / "vectors_raw.npy", mmap_mode="r")
    syms = json.loads((IDX_DIR / "symbols.json").read_text(encoding="utf-8"))
    n = len(m)
    c1 = (shp.shape[0] == n == rwp.shape[0])
    say(f"\n① 行数一致           meta {n:,} · shape {shp.shape} · raw {rwp.shape}   "
        f"{'✅' if c1 else '❌'}")
    ok_all &= c1
    rep["n_windows"] = n

    # ---------- 2. 排序与唯一：分隔不变，币内时间严格递增 ----------
    order = m["symbol"].to_numpy()
    blocks = (order[1:] != order[:-1]).sum() + 1
    c2 = int(blocks) == int(m["symbol"].nunique())
    dup = m.duplicated(subset=["symbol", "t0"]).sum()
    inc = m.groupby("symbol", sort=False)["t1"].apply(lambda s: s.is_monotonic_increasing).all()
    c3 = (dup == 0) and bool(inc)
    say(f"② 币块连续不交错     {int(blocks)} 块 / {m['symbol'].nunique()} 币   {'✅' if c2 else '❌'}")
    say(f"③ 无重复窗口+时间递增 重复 {dup} 条 · 每币 t1 递增 {bool(inc)}   {'✅' if c3 else '❌'}")
    ok_all &= c2 and c3

    # ---------- 4. 老窗口逐字节没动（核心） ----------
    if SNAP_VEC.exists():
        snap = pd.read_parquet(SNAP_VEC)
        # ⚠️ 不用 merge —— 合并结果的行序不保证与左表一致，比错位了会得出"老数据被动过"的假警报。
        #    改用 (币 → {t0: 行号}) 显式查表，顺序完全由快照决定。
        rowmap: dict[str, dict] = {}
        for sym, g in m.groupby("symbol", sort=False):
            rowmap[str(sym)] = dict(zip(g["t0"].to_numpy(), g.index.to_numpy()))
        rows_new = np.array([rowmap.get(str(s), {}).get(int(t), -1)
                             for s, t in zip(snap["symbol"], snap["t0"])], dtype=np.int64)
        miss = int((rows_new < 0).sum())
        valid = np.flatnonzero(rows_new >= 0)
        rows_new = rows_new[valid]
        sv = snap.iloc[valid].reset_index(drop=True)
        mn = m.iloc[rows_new].reset_index(drop=True)

        sh_new = _sha_rows(IDX_DIR / "vectors_shape.npy", rows_new)
        rw_new = _sha_rows(IDX_DIR / "vectors_raw.npy", rows_new)
        same_shape = float(np.mean([a == b for a, b in zip(sh_new, sv["sha_shape"])]))
        same_raw = float(np.mean([a == b for a, b in zip(rw_new, sv["sha_raw"])]))
        # 数值也逐字段比（指纹一致但字段漂移是可能的：指纹只覆盖向量，不覆盖 meta）
        fmax = 0.0
        for f in ("amp_pct", "vol_pct", "t1"):
            d = np.abs(sv[f].to_numpy(np.float64) - mn[f].to_numpy(np.float64))
            fmax = max(fmax, float(np.nanmax(d)) if len(d) and not np.all(np.isnan(d)) else 0.0)
        a3 = sv[["fwd_ret", "fwd_max", "fwd_min"]].to_numpy(np.float64)
        b3 = mn[["fwd_ret", "fwd_max", "fwd_min"]].to_numpy(np.float64)
        both = np.isfinite(a3) & np.isfinite(b3)
        fwd_diff = int((np.abs(a3[both] - b3[both]) > 1e-9).sum())
        nan_shift = int((np.isnan(a3) != np.isnan(b3)).sum())
        c4 = (miss == 0) and same_shape == 1.0 and same_raw == 1.0 \
            and fmax == 0.0 and fwd_diff == 0 and nan_shift == 0
        say(f"\n④ ⭐老窗口逐字节没动  抽检 {len(snap):,} 条")
        say(f"     找不回的老窗口   {miss}   {'✅' if miss == 0 else '❌'}")
        say(f"     向量指纹相同     shape {same_shape*100:.2f}% · raw {same_raw*100:.2f}%")
        say(f"     元数据最大偏差   {fmax:.10f} · 后续走势不一致 {fwd_diff} 条 · 缺失态变化 {nan_shift}")
        say(f"     {'✅ 老数据一个字节都没动' if c4 else '❌ 老数据被动过 —— 增量不成立'}")
        ok_all &= c4
        rep.update({"old_missing": miss, "old_same_shape": same_shape,
                    "old_same_raw": same_raw, "old_field_maxdiff": fmax,
                    "old_fwd_mismatch": fwd_diff, "old_nan_shift": nan_shift})
    else:
        say("\n④ ⚠️ 没有快照（没跑 --stage snapshot）→ 无法证明老数据没被动。")
        say("     结论：这次只能算「重建」，不能声称「增量等价」—— 缺的正是证据。")
        rep["old_unchecked"] = True

    # ---------- 5. 每币窗口数：只该增加，不该凭空变化 ----------
    if SNAP_META.exists():
        meta0 = json.loads(SNAP_META.read_text(encoding="utf-8"))
        per0 = meta0["per_coin"]
        per1 = m.groupby("symbol").size().to_dict()
        changed = {k: (per0.get(k, 0), per1.get(k, 0)) for k in set(per0) | set(per1)
                   if per0.get(k, 0) != per1.get(k, 0)}
        shrunk = {k: v for k, v in changed.items() if v[1] < v[0]}
        say(f"\n⑤ 窗口数变化         总 {meta0['n_windows']:,} → {n:,}"
            f"（+{n - meta0['n_windows']:,}）")
        say(f"     窗口数有变的币   {len(changed)} / {len(per1)}"
            f"   {'✅' if not shrunk else '❌'}   变少的币 {len(shrunk)}")
        ok_all &= (not shrunk)
        rep.update({"windows_before": meta0["n_windows"], "windows_after": n,
                    "coins_changed": len(changed), "coins_shrunk": len(shrunk)})

    # ---------- 6. 流动性表：老日期必须逐位一致 ----------
    if SNAP_LIQ.exists():
        b = pd.read_parquet(SNAP_LIQ)
        a = pd.read_parquet(DATA / "pool_liquidity.parquet",
                            columns=["day", "sym", "liq30", "liq60"])
        x = b.merge(a, on=["day", "sym"], how="left", suffixes=("_b", "_a"))
        # ⚠️ 这里量的是「**新表把旧表里原本有值的格子弄没了**」，这才是"老日期被改写"。
        #    踩过的坑（2026-09-22）：第一版写成 `x["liq30_a"].isna().sum()`，
        #    把"两边**都**是 NaN"的 1,800 行（新上市不足 30 天，liq30 本来就算不出来）
        #    也算成了失败 → 620 万行数据零偏差却报 ❌。**检查方法本身会成为假警报源。**
        nan_new = int((x["liq30_b"].notna() & x["liq30_a"].isna()).sum())
        both_nan = int((x["liq30_b"].isna() & x["liq30_a"].isna()).sum())
        d30 = float(np.nanmax(np.abs(x["liq30_b"] - x["liq30_a"])))
        d60 = float(np.nanmax(np.abs(x["liq60_b"] - x["liq60_a"])))
        nl30 = int((x["liq30_b"].isna() != x["liq30_a"].isna()).sum())
        c6 = (nan_new == 0) and d30 == 0.0 and d60 == 0.0 and nl30 == 0
        say(f"\n⑥ 流动性表老日期    比对 {len(b):,} 行（旧表范围）")
        say(f"     原本有值却变空   {nan_new}   {'✅' if nan_new == 0 else '❌'}   "
            f"（两边都算不出的格子 {both_nan:,}，属正常）")
        say(f"     liq30 最大偏差   {d30:.10f} · liq60 {d60:.6f} · 缺失态变化 {nl30} 行")
        say(f"     {'✅ 老日期的流动性指标逐位未变（增长只发生在日历末尾）' if c6 else '❌ 老日期被改写 → 币池口径变了'}")
        ok_all &= c6
        rep.update({"liq30_maxdiff": d30, "liq60_maxdiff": d60, "liq_nan_new": nan_new})

    # ---------- 7. 时间上限推进 ----------
    src_last = max(m["last"] for m in all_sym_meta().values())
    say(f"\n⑦ 数据覆盖           索引最晚 t1 = {ms2s(int(m['t1'].max()))}")
    say(f"     raw/1m 最晚     {ms2s(src_last)}")
    say(f"     索引跨度         {ms2s(int(m['t0'].min()))} ~ {ms2s(int(m['t1'].max()))}")
    rep["index_tmax"] = int(m["t1"].max())
    rep["index_tmin"] = int(m["t0"].min())
    rep["raw_1m_last"] = int(src_last)
    rep["coins"] = int(m["symbol"].nunique())

    say("\n" + "=" * 78)
    say("结论：" + ("✅ 全部通过 —— 增量扩建成立，老数据零改动。"
                  if ok_all else "❌ 有项目未通过，见上。"))
    say("=" * 78)
    rep["all_ok"] = bool(ok_all)
    (RESULTS / "refresh_report.json").write_text(
        json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if ok_all else 1


# ================================================================ stage: all

def _snapshot_stale() -> bool:
    """快照缺失，或快照比当前索引旧（上次建库之后没再拍过）→ 视为过期。"""
    if not SNAP_VEC.exists():
        return True
    meta_p = IDX_DIR / "meta.parquet"
    if not meta_p.exists():
        return True
    return SNAP_VEC.stat().st_mtime < meta_p.stat().st_mtime


def cmd_all(args):
    # ⭐（refresh.md §8① 收尾，2026-10-04）快照自动管理：
    #   「老数据零改动」的证据需要一个**当次**的基准。没有快照、或快照落后于当前索引
    #   （上一轮 --stage all 建完库之后没再拍过）→ 先自动补拍，再进流水线。
    #   手动先跑了 --stage snapshot 的（快照比索引新）不会被重复拍。
    if _snapshot_stale():
        say("[all] 快照缺失或落后于当前索引 → 自动补拍老窗口指纹（基准 = 当前库）")
        rc = cmd_snapshot(args)
        if rc:
            say("✗ 自动补拍快照失败，停止（没有基准就不许动库）")
            return rc
    for fn in (cmd_fetch, cmd_merge, cmd_derive, cmd_liquidity, cmd_index, cmd_verify):
        say("")
        rc = fn(args)
        if rc:
            say(f"✗ {fn.__name__} 返回 {rc}，停止")
            return rc
    return 0


def main():
    ap = argparse.ArgumentParser(description="滚动补数据（增量并进检索库）")
    ap.add_argument("--stage", default="all",
                    choices=["snapshot", "fetch", "merge", "derive", "liquidity",
                             "index", "verify", "all"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n_sample", type=int, default=4000, help="snapshot 抽样条数")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--until", default=None, help="每日归档的最后一天（默认自动探测）")
    ap.add_argument("--days", default=None, help="merge 时指定日期，逗号分隔")
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--force", action="store_true", help="fetch 阶段重下已有文件")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--win", type=int, default=100)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--dst", type=int, default=32)
    ap.add_argument("--pool_top", type=int, default=20)
    ap.add_argument("--force_syms", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    return {"snapshot": cmd_snapshot, "fetch": cmd_fetch, "merge": cmd_merge,
            "derive": cmd_derive, "liquidity": cmd_liquidity, "index": cmd_index,
            "verify": cmd_verify, "all": cmd_all}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
