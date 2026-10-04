#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
裸K盘感项目 · M0 point-in-time 数据管道
数据源：https://data.binance.vision/ （公开直连，免登录，含已下架交易对）

设计约束（见 04_验证纪律与工程约定.md）：
  - 断点续跑：输出存在且参数指纹一致则跳过
  - 脚本只打印摘要，完整明细落盘（控制 AI 交互成本）
  - 记录每个交易对的真实可见区间 -> point-in-time，治理幸存者偏差
  - 不产生无法清理的临时文件（os.remove 在本机可用，bash rm 不可用）

用法：
  python fetch_data.py list                     # 抓取全部交易对清单
  python fetch_data.py plan  --interval 1h      # 生成下载计划（不下载）
  python fetch_data.py fetch --interval 1h      # 执行下载 -> parquet
  python fetch_data.py fetch --interval 1s --symbols BTCUSDT --months 2024-06,2025-06
  python fetch_data.py verify --interval 1h     # 校验输出完整性
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
TMP = DATA / "_tmp"

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CDN = "https://data.binance.vision"
KLINE_PREFIX = "data/spot/monthly/klines"
S3NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

COLS = ["open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "lK-pangen/0.1 (research)"})
try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    _ad = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.6,
                                        status_forcelist=[429, 500, 502, 503, 504]),
                      pool_connections=32, pool_maxsize=32)
    SESSION.mount("https://", _ad)
except Exception:                                                  # noqa: BLE001
    pass


# ---------------------------------------------------------------- S3 列举

def s3_list(prefix: str, delimiter: str | None = None, max_pages: int = 300):
    """分页列举。返回 (keys, prefixes)。"""
    keys: list[str] = []
    prefixes: list[str] = []
    start_after = None
    for _ in range(max_pages):
        p = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            p["delimiter"] = delimiter
        if start_after:
            p["start-after"] = start_after
        r = SESSION.get(S3, params=p, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for c in root.findall(".//s3:Contents", S3NS):
            keys.append(c.find("s3:Key", S3NS).text)
        for c in root.findall(".//s3:CommonPrefixes", S3NS):
            prefixes.append(c.find("s3:Prefix", S3NS).text)
        trunc = root.find(".//s3:IsTruncated", S3NS)
        if trunc is None or trunc.text != "true":
            break
        if prefixes:
            start_after = prefixes[-1]
        elif keys:
            start_after = keys[-1]
        else:
            break
    return keys, prefixes


# ---------------------------------------------------------------- 阶段一：清单

def cmd_list(_args):
    DATA.mkdir(parents=True, exist_ok=True)
    _, prefixes = s3_list(KLINE_PREFIX + "/", delimiter="/")
    symbols = sorted(p.rsplit("/", 2)[-2] for p in prefixes if p.rstrip("/").count("/") >= 3)
    out = DATA / "symbols_all.txt"
    out.write_text("\n".join(symbols), encoding="utf-8")
    _dump_run(DATA / "list", {"stage": "list", "n_symbols": len(symbols),
                              "source": S3 + "/" + KLINE_PREFIX})
    print(f"[list] 交易对总数 {len(symbols)} -> {out}")
    print("       样例: " + ", ".join(symbols[:8]) + " ...")
    return 0


# ---------------------------------------------------------------- 阶段二：计划

def cmd_plan(args):
    syms = _resolve_symbols(args)
    interval = args.interval
    avail = {}
    for s in syms:
        keys, _ = s3_list(f"{KLINE_PREFIX}/{s}/{interval}/")
        avail[s] = sorted(_month_of(k) for k in keys if k.endswith(".zip"))
    plan = {s: m for s, m in avail.items() if m}
    out = DATA / f"plan_{interval}.json"
    out.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    empty = [s for s in syms if not avail.get(s)]
    print(f"[plan] interval={interval} 有数据的交易对 {len(plan)}/{len(syms)}")
    for s in list(plan)[:12]:
        m = plan[s]
        print(f"       {s:<12} {len(m):>3} 个月  {m[0]} ~ {m[-1]}")
    if empty:
        print(f"       无数据(跳过): {', '.join(empty[:10])}")
    print(f"       明细 -> {out}")
    return 0


def _month_of(key: str) -> str:
    # .../BTCUSDT-1h-2021-01.zip -> 2021-01
    return key.rsplit("-", 2)[-2] + "-" + key.rsplit("-", 2)[-1].replace(".zip", "")


def cmd_universe(args):
    """按实际数据自动构造交易对池（替代人工挑选，避免选择性偏差）。

    规则：
      1. 只取 USDT 计价的现货交易对
      2. 数据跨度 >= --min-months 个月才入选（剔除一闪而过的短期上市）
      3. 按数据跨度排序取前 --max-symbols 个
      4. **强制保留已死亡交易对**（最后数据月 < --alive-after，跨度 >= --min-months-dead）
         —— 它们是幸存者偏差的解药，不能被排序规则挤掉
    同时产出 plan_{interval}.json，后续 fetch 不必重新列举。
    """
    import concurrent.futures as cf

    interval = args.interval
    min_m, max_s = args.min_months, args.max_symbols
    min_md = args.min_months_dead

    all_txt = DATA / "symbols_all.txt"
    if not all_txt.exists():
        cmd_list(args)
    all_syms = [l.strip() for l in all_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    usdt = [s for s in all_syms if s.endswith("USDT") and s != "USDTUSDT"]

    def probe(s):
        keys, _ = s3_list(f"{KLINE_PREFIX}/{s}/{interval}/")
        return s, sorted(_month_of(k) for k in keys if k.endswith(".zip"))

    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        res = dict(ex.map(probe, usdt))

    # 杠杆代币（UP/DOWN/BULL/BEAR）是每日再平衡的 3 倍产品，路径依赖衰减，
    # 不是"裸 K 形态"的研究对象，且会污染形态库 —— 排除
    levy = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")

    rows = [{"symbol": s, "months": ms, "span": len(ms), "first": ms[0], "last": ms[-1]}
            for s, ms in res.items()
            if ms and not s.endswith(levy)]
    # "已死亡"以数据源实际最新月份为准，不用系统当前月（当月文件尚未生成，会全体误判）
    latest = args.alive_after or max(r["last"] for r in rows)

    def mdiff(a, b):
        ya, ma = int(a[:4]), int(a[5:7])
        yb, mb = int(b[:4]), int(b[5:7])
        return (yb - ya) * 12 + (mb - ma)

    cand = [r for r in rows if r["span"] >= min_m]
    dead = [r for r in cand if mdiff(r["last"], latest) > 2 and r["span"] >= min_md]
    alive = sorted([r for r in cand if mdiff(r["last"], latest) <= 2],
                   key=lambda r: (-r["span"], r["symbol"]))
    dead.sort(key=lambda r: (-r["span"], r["symbol"]))
    # 已死亡样本全保留，但不超过配额的 1/3，保证存活样本的覆盖面
    dead_keep = dead[: max(1, max_s // 3)]
    keep = sorted(dead_keep + alive[: max_s - len(dead_keep)], key=lambda r: r["symbol"])

    syms = [r["symbol"] for r in keep]
    (DATA / "symbols_universe.txt").write_text(
        "# 自动构造（python fetch_data.py universe）\n"
        f"# 规则：USDT 计价 · 跨度 >= {min_m} 月 · 上限 {max_s} 个"
        f" · 强制保留已死亡（跨度 >= {min_md} 月）\n"
        + "\n".join(syms) + "\n", encoding="utf-8")
    plan = {r["symbol"]: r["months"] for r in keep}
    (DATA / f"plan_{interval}.json").write_text(
        json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"[universe] {interval} ｜ USDT 候选 {len(usdt)} 个 ｜ 跨度>={min_m}月 {len(cand)} 个")
    print(f"           已死亡候选 {len(dead)} 个 ｜ 实际保留 {len(dead_keep)} 个"
          f"（上限 {max(1, max_s // 3)}，其余为存活样本）")
    print(f"           最终 {len(syms)} 个 ｜ 待下载文件 {sum(len(m) for m in plan.values()):,} 个"
          f" ｜ 数据最新月 {latest}")
    print(f"           保留的已死亡样本: {', '.join(r['symbol'] for r in dead_keep[:12])}")
    print(f"           -> data/symbols_universe.txt + data/plan_{interval}.json")
    return 0


# ---------------------------------------------------------------- 阶段三：下载

def cmd_fetch(args):
    interval = args.interval
    keep_zip = args.keep_zips if args.keep_zips is not None else (interval == "1h")
    raw_dir = RAW / interval
    raw_dir.mkdir(parents=True, exist_ok=True)
    plan_path = DATA / f"plan_{interval}.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        syms = _resolve_symbols(args)
        plan = {s: sorted(_month_of(k) for k in s3_list(f"{KLINE_PREFIX}/{s}/{interval}/")[0]
                          if k.endswith(".zip")) for s in syms}

    months_filter = set(args.months.split(",")) if args.months else None
    todo = list(plan)[: args.limit] if args.limit else list(plan)
    fingerprint = {"interval": interval, "months": sorted(months_filter) if months_filter else "all"}
    written, skipped, failed = [], [], []
    t0 = time.time()
    for i, s in enumerate(todo, 1):
        months = [m for m in plan[s] if not months_filter or m in months_filter]
        if not months:
            continue
        out_path = raw_dir / f"{s}.parquet"
        if out_path.exists() and not args.force:
            meta = _read_run(raw_dir / f"{s}_run.json")
            if meta and meta.get("fingerprint") == fingerprint:
                skipped.append(s)
                continue
        try:
            df, ok, bad = _fetch_one(s, interval, months, raw_dir, keep_zip, args.workers)
        except Exception as e:                                    # noqa: BLE001
            failed.append(f"{s}: {type(e).__name__} {e}")
            print(f"  [{i}/{len(todo)}] {s} 失败: {type(e).__name__} {e}")
            continue
        if df is None or len(df) == 0:
            failed.append(f"{s}: 无有效数据")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False, compression="zstd")
        _dump_run(raw_dir / f"{s}_run", {
            "fingerprint": fingerprint, "symbol": s, "interval": interval,
            "months_requested": len(months), "months_ok": len(ok),
            "months_failed": bad, "rows": int(len(df)),
            "first": str(df["open_time"].iloc[0]), "last": str(df["open_time"].iloc[-1]),
            "source": f"{CDN}/{KLINE_PREFIX}/{s}/{interval}/",
        })
        written.append(s)
        print(f"  [{i}/{len(todo)}] {s:<12} {len(df):>8} 根  "
              f"{str(df['open_time'].iloc[0])[:16]} ~ {str(df['open_time'].iloc[-1])[:16]}")

    _build_symbols_json(interval, raw_dir)
    dt = time.time() - t0
    print(f"[fetch] {interval} 完成: 新下载 {len(written)} / 跳过 {len(skipped)} / 失败 {len(failed)}"
          f"  用时 {dt:.0f}s")
    # 失败清单每轮都重写（成功时写入"无失败"标记）。
    # 不依赖删除：沙箱内 os.remove 会被拦截并转投回收站 → 必失败（见 04 文档 2.1）。
    fail_path = DATA / f"failed_{interval}.txt"
    fail_path.write_text(
        "\n".join(failed) if failed
        else f"# 本轮无失败  {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n",
        encoding="utf-8")
    print(f"        失败清单 -> {fail_path}")
    return 0


def _fetch_one(symbol, interval, months, raw_dir, keep_zip, workers=16):
    """并发抓取单币的所有月份。返回 (df | None, 成功月份, 失败月份)。"""
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    zip_dir = raw_dir / symbol
    if keep_zip:
        zip_dir.mkdir(parents=True, exist_ok=True)

    def one(m):
        name = f"{symbol}-{interval}-{m}.zip"
        url = f"{CDN}/{KLINE_PREFIX}/{symbol}/{interval}/{name}"
        try:
            r = SESSION.get(url, timeout=180)
            if r.status_code != 200 or len(r.content) < 200:
                return m, None, None
            df = _parse_zip(r.content)
            if df is None or len(df) == 0:
                return m, None, None
            return m, df, r.content if keep_zip else None
        except Exception:                                          # noqa: BLE001
            return m, None, None

    frames, ok, bad = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for m, df, blob in ex.map(one, months):
            if df is None:
                bad.append(m)
            else:
                frames.append(df)
                ok.append(m)
                if keep_zip and blob is not None:
                    (zip_dir / f"{symbol}-{interval}-{m}.zip").write_bytes(blob)
    if not frames:
        return None, ok, bad
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    out.insert(0, "symbol", symbol)
    return out, ok, bad


def _parse_zip(content: bytes):
    import pandas as pd

    z = zipfile.ZipFile(io.BytesIO(content))
    names = [n for n in z.namelist() if n.endswith(".csv")]
    if not names:
        return None
    with z.open(names[0]) as fh:
        raw = fh.read().decode("utf-8", "replace")
    lines = [l for l in raw.splitlines() if l.strip()]
    if lines and not lines[0][:1].isdigit():          # 新版文件带表头
        lines = lines[1:]
    if not lines:
        return None
    ncol = len(lines[0].split(","))
    if ncol < 11:                                     # 列数异常，不猜
        return None
    use = COLS[: min(ncol, 12)]
    df = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=use,
                     on_bad_lines="skip", engine="c")
    df = df[[c for c in COLS[:11] if c in df.columns]].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open_time", "close"])
    if df.empty:
        return None
    # 2025 起部分文件时间戳改为微秒，统一归一到毫秒
    if df["open_time"].iloc[0] > 1e14:
        df["open_time"] = df["open_time"] // 1000
        df["close_time"] = df["close_time"] // 1000
    df["open_time"] = df["open_time"].astype("int64")
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def cmd_fetch_days(args):
    """按（币, 日）清单下载 daily klines —— 插针研究专用。

    清单来自 `scripts/wick_scan.py` 产出的 `data/wick_days.json`。
    思路：用便宜的 1m 定位稀有事件，只对稀有事件取昂贵的细数据
    （日 1s 约 2 MB，月 1s 约 70 MB，差 35 倍）。
    """
    import concurrent.futures as cf

    df_path = Path(args.days_file)
    if not df_path.exists():
        sys.exit(f"缺少 {df_path}（先跑 scripts/wick_scan.py）")
    meta = json.loads(df_path.read_text(encoding="utf-8"))
    pairs = [(s, d) for s, d in meta.get("days", [])]
    if not pairs:
        sys.exit("清单为空")
    interval = args.interval
    out_dir = RAW / f"{interval}_days"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = {"source": "daily", "interval": interval,
          "days_file": df_path.name, "rule": meta.get("rule", "")}
    ok = skip = 0
    fail = []

    def one(sym, date):
        p = out_dir / sym
        out_path = p / f"{sym}-{date}.parquet"
        if out_path.exists() and not args.force:
            # _dump_run 落的是 "<sym>-<date>.json"（无 _run 后缀），这里必须对齐
            m = _read_run(p / f"{sym}-{date}.json")
            if m and m.get("fingerprint") == fp:
                return "skip", sym, date
        url = f"{CDN}/data/spot/daily/klines/{sym}/{interval}/{sym}-{interval}-{date}.zip"
        try:
            r = SESSION.get(url, timeout=180)
            if r.status_code != 200 or len(r.content) < 200:
                return "fail", sym, date
            d = _parse_zip(r.content)
            if d is None or d.empty:
                return "fail", sym, date
            d.insert(0, "symbol", sym)
            p.mkdir(parents=True, exist_ok=True)
            d.to_parquet(out_path, index=False, compression="zstd")
            if args.keep_zips:
                (p / f"{sym}-{interval}-{date}.zip").write_bytes(r.content)
            _dump_run(p / f"{sym}-{date}", {"fingerprint": fp, "rows": int(len(d)),
                                            "first": str(d["open_time"].iloc[0]),
                                            "last": str(d["open_time"].iloc[-1])})
            return "ok", sym, date
        except Exception as e:                                     # noqa: BLE001
            return "fail", sym, f"{date}:{type(e).__name__}"

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for st, sym, date in ex.map(lambda t: one(*t), pairs):
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                fail.append(f"{sym} {date}")

    print(f"[fetchdays] {interval} ｜ 成功 {ok} / 跳过 {skip} / 失败 {len(fail)}"
          f" ｜ 落盘 {out_dir}")
    fl = DATA / f"failed_{interval}_days.txt"
    fl.write_text("\n".join(fail) if fail
                  else f"# 本轮无失败  {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n",
                  encoding="utf-8")
    if fail:
        print(f"            失败样例: {', '.join(fail[:6])}")
    return 0


# ---------------------------------------------------------------- 阶段四：可见区间

def _build_symbols_json(interval, raw_dir):
    import pandas as pd

    rows = []
    for p in sorted(raw_dir.glob("*.parquet")):
        if p.name.endswith("_run.parquet"):
            continue
        m = _read_run(raw_dir / f"{p.stem}_run.json") or {}
        try:
            d = pd.read_parquet(p, columns=["open_time"])
        except Exception:                                          # noqa: BLE001
            continue
        first = pd.to_datetime(int(d["open_time"].min()), unit="ms", utc=True)
        last = pd.to_datetime(int(d["open_time"].max()), unit="ms", utc=True)
        rows.append({
            "symbol": p.stem, "interval": interval,
            "first_bar_utc": first.isoformat(), "last_bar_utc": last.isoformat(),
            "n_bars": int(len(d)),
            "months_ok": m.get("months_ok"), "months_failed": m.get("months_failed", []),
        })
    rows.sort(key=lambda r: r["symbol"])
    out = DATA / f"symbols.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[interval] = rows
    out.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding="utf-8")

    if rows:
        last_month = max(r["last_bar_utc"] for r in rows)[:7]
        stale = [r["symbol"] for r in rows
                 if (datetime.fromisoformat(last_month + "-01T00:00:00+00:00")
                     - datetime.fromisoformat(r["last_bar_utc"][:7] + "-01T00:00:00+00:00")).days > 60]
        print(f"[meta] symbols.json 已更新: {len(rows)} 个交易对, 最新月份 {last_month}, "
              f"疑似已下架 {len(stale)} 个")
        if stale:
            print("       疑似下架: " + ", ".join(stale[:12]))
    return rows


# ---------------------------------------------------------------- 阶段五：校验

def cmd_verify(args):
    import pandas as pd

    interval = args.interval
    raw_dir = RAW / interval
    files = sorted(p for p in raw_dir.glob("*.parquet"))
    tot = 0
    problems = []
    for p in files:
        d = pd.read_parquet(p)
        tot += len(d)
        if d["open_time"].duplicated().any():
            problems.append(f"{p.stem}: open_time 重复")
        if not d["open_time"].is_monotonic_increasing:
            problems.append(f"{p.stem}: 时间未升序")
        if (d["high"] < d["low"]).any():
            problems.append(f"{p.stem}: high<low")
        if (d["high"] < d["open"]).any() or (d["high"] < d["close"]).any():
            problems.append(f"{p.stem}: high 未覆盖 open/close")
        if (d["low"] > d["open"]).any() or (d["low"] > d["close"]).any():
            problems.append(f"{p.stem}: low 未覆盖 open/close")
    print(f"[verify] {interval}: {len(files)} 个交易对, 合计 {tot:,} 根 K 线")
    if problems:
        print(f"         发现 {len(problems)} 处问题:")
        for x in problems[:20]:
            print("         - " + x)
    else:
        print("         一致性检查全部通过")
    return 0


# ---------------------------------------------------------------- 工具

def _dump_run(base: Path, obj: dict):
    base.parent.mkdir(parents=True, exist_ok=True)
    (base.with_name(base.name + ".json")).write_text(
        json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def _read_run(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return None


def _resolve_symbols(args) -> list[str]:
    if getattr(args, "symbols", None):
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for name in ("symbols_universe.txt", "symbols_core.txt"):
        f = DATA / name
        if f.exists():
            return [l.strip().upper() for l in f.read_text(encoding="utf-8").splitlines()
                    if l.strip() and not l.startswith("#")]
    sys.exit("缺少 data/symbols_universe.txt 或 data/symbols_core.txt"
             "（先跑 `universe`，或用 --symbols 指定）")


def main():
    ap = argparse.ArgumentParser(description="裸K盘感项目数据管道")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    p = sub.add_parser("universe")
    p.set_defaults(func=cmd_universe, symbols=None)
    p.add_argument("--interval", default="1m")
    p.add_argument("--min-months", dest="min_months", type=int, default=24)
    p.add_argument("--max-symbols", dest="max_symbols", type=int, default=200)
    p.add_argument("--min-months-dead", dest="min_months_dead", type=int, default=12)
    p.add_argument("--alive-after", dest="alive_after", default=None,
                   help="最后数据月早于此值即视为已死亡，默认当前月")

    p = sub.add_parser("fetchdays")
    p.set_defaults(func=cmd_fetch_days, symbols=None)
    p.add_argument("--interval", default="1s")
    p.add_argument("--days-file", dest="days_file", default=str(DATA / "wick_days.json"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-zips", dest="keep_zips", action="store_true", default=True)

    for name, fn in (("plan", cmd_plan), ("fetch", cmd_fetch), ("verify", cmd_verify)):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
        p.add_argument("--interval", default="1h")
        p.add_argument("--symbols", default=None)
        if name != "verify":
            p.add_argument("--months", default=None, help="逗号分隔，如 2024-06,2025-06")
        if name == "fetch":
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--workers", type=int, default=16)
            p.add_argument("--force", action="store_true")
            p.add_argument("--keep-zips", dest="keep_zips", action="store_true", default=None)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
