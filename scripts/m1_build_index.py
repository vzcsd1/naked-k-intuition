# -*- coding: utf-8 -*-
"""M1 步骤2：窗口切片 + 向量化 + 落盘（memmap）+ V_full 定标。
三段式（可断点续跑，不产生待删临时文件）：
  plan     预计算每个币的窗口数、行区间，预分配 memmap
  fill     逐币向量化写入（进度 JSON，中断后重跑自动跳过已完成的币）
  finalize 按块采样定标 -> 生成 V_full(N,64)
用法: python m1_build_index.py plan|fill|finalize|all
"""
import json
import os
import sys
import time

import numpy as np

import m1_lib as L

PLAN = os.path.join(L.DIR_INDEX, "plan.json")
PROGRESS = os.path.join(L.DIR_INDEX, "fill_progress.json")
SCALES = os.path.join(L.DIR_INDEX, "blocks_scale.json")

ARRS = {  # name -> (dtype, dim)
    "v_shape": (np.float32, L.DIM_SHAPE),
    "v_raw": (np.float32, L.DIM_RAW),
    "v_vol": (np.float32, L.DIM_VOL),
    "v_rng": (np.float32, L.DIM_RNG),
    "v_trd": (np.float32, L.DIM_TRD),
    "m_sym": (np.int32, 1),
    "m_start": (np.int64, 1),
    "m_end": (np.int64, 1),
    "m_atr": (np.float32, 1),
}


def _arr_path(name):
    return os.path.join(L.DIR_INDEX, name + ".npy")


def cmd_plan():
    os.makedirs(L.DIR_INDEX, exist_ok=True)
    syms = L.list_symbols()
    plan, total, cursor = {}, 0, 0
    t0 = time.time()
    for i, sym in enumerate(syms, 1):
        df = L.load_15m(sym, columns=["open_time", "n_bars"])
        starts = L.window_starts(df["open_time"].to_numpy(np.int64),
                                 df["n_bars"].to_numpy())
        k = int(len(starts))
        plan[sym] = {"row_start": cursor, "k": k}
        cursor += k
        if i % 25 == 0:
            print(f"plan {i}/{len(syms)} cumulative_windows={cursor}", flush=True)
    total = cursor
    plan["_total"] = total
    L.write_json(PLAN, plan)
    L.write_json(os.path.join(L.DIR_INDEX, "symbols.json"),
                 sorted(s for s in plan if not s.startswith("_")))
    symbols = sorted(s for s in plan if not s.startswith("_"))
    sym_i = {s: j for j, s in enumerate(symbols)}
    for name, (dt, dim) in ARRS.items():
        shape = (total,) if dim == 1 else (total, dim)
        np.lib.format.open_memmap(_arr_path(name), mode="w+", dtype=dt, shape=shape)
    L.write_json(PROGRESS, {})
    print(f"plan done: {len(symbols)} symbols, {total} windows, {time.time()-t0:.0f}s",
          flush=True)


def cmd_fill():
    with open(PLAN, encoding="utf-8") as f:
        plan = json.load(f)
    with open(PROGRESS, encoding="utf-8") as f:
        progress = json.load(f)
    symbols = sorted(s for s in plan if not s.startswith("_"))
    sym_i = {s: j for j, s in enumerate(symbols)}
    arrays = {name: np.lib.format.open_memmap(_arr_path(name), mode="r+")
              for name in ARRS}
    t0, n_done = time.time(), 0
    for sym in symbols:
        p = plan[sym]
        if progress.get(sym) == p["k"]:
            continue
        if p["k"] == 0:
            progress[sym] = 0
            L.write_json(PROGRESS, progress)
            continue
        df = L.load_15m(sym)
        vec = L.vectorize_windows(df)
        if vec is None or len(vec["start_ot"]) != p["k"]:
            # 行数必须与 plan 一致，否则行区间错位 —— 直接失败而不是写坏
            got = 0 if vec is None else len(vec["start_ot"])
            raise RuntimeError(f"{sym}: window count mismatch plan={p['k']} got={got}")
        s = p["row_start"]
        e = s + p["k"]
        for name in ("v_shape", "v_raw", "v_vol", "v_rng", "v_trd"):
            arrays[name][s:e] = vec[name]
        arrays["m_sym"][s:e] = sym_i[sym]
        arrays["m_start"][s:e] = vec["start_ot"]
        arrays["m_end"][s:e] = vec["end_ot"]
        arrays["m_atr"][s:e] = vec["atr"]
        progress[sym] = p["k"]
        L.write_json(PROGRESS, progress)
        n_done += 1
        if n_done % 10 == 0:
            print(f"fill {len(progress)}/{len(symbols)} symbols "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"fill done: {n_done} symbols this run, "
          f"{sum(int(v) for k_, v in progress.items() if not k_.startswith('_'))} windows",
          flush=True)


def cmd_finalize():
    with open(PLAN, encoding="utf-8") as f:
        plan = json.load(f)
    with open(PROGRESS, encoding="utf-8") as f:
        progress = json.load(f)
    total = plan["_total"]
    expected = sum(v["k"] for k_, v in plan.items() if not k_.startswith("_"))
    if sum(progress.values()) != expected:
        raise RuntimeError(f"fill incomplete: {sum(progress.values())}/{expected}")
    # 各块全局尺度（等通道权重默认值；盲测前不调权重 —— 见 08/10）
    rng = np.random.default_rng(42)
    sample = rng.choice(total, size=min(200_000, total), replace=False)
    scales = {}
    for name, dim in [("v_shape", L.DIM_SHAPE), ("v_vol", L.DIM_VOL),
                      ("v_rng", L.DIM_RNG), ("v_trd", L.DIM_TRD)]:
        a = np.load(_arr_path(name), mmap_mode="r")[np.sort(sample)]
        s = float(a.std(axis=0).mean())
        if not np.isfinite(s) or s <= 0:
            raise RuntimeError(f"block {name} has zero std, cannot scale")
        scales[name] = s
    L.write_json(SCALES, scales)
    # V_full = [shape/s, vol/s, rng/s, trd/s] -> (N, 64)
    v_full = np.lib.format.open_memmap(_arr_path("v_full"), mode="w+",
                                       dtype=np.float32, shape=(total, L.DIM_FULL))
    cs = [(0, L.DIM_SHAPE, "v_shape"),
          (L.DIM_SHAPE, L.DIM_SHAPE + L.DIM_VOL, "v_vol"),
          (L.DIM_SHAPE + L.DIM_VOL, L.DIM_SHAPE + L.DIM_VOL + L.DIM_RNG, "v_rng"),
          (L.DIM_FULL - L.DIM_TRD, L.DIM_FULL, "v_trd")]
    chunk = 500_000
    for s in range(0, total, chunk):
        e = min(s + chunk, total)
        for c0, c1, name in cs:
            src = np.load(_arr_path(name), mmap_mode="r")[s:e]
            v_full[s:e, c0:c1] = np.asarray(src, dtype=np.float32) / np.float32(scales[name])
    v_full.flush()
    stats = {"total_windows": total, "block_scales": scales,
             "v_full_path": _arr_path("v_full"), "dim_full": L.DIM_FULL}
    L.write_json(os.path.join(L.DIR_INDEX, "build_stats.json"), stats)
    print("finalize done:", json.dumps(stats["block_scales"]), flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(L.DIR_INDEX, exist_ok=True)
    if mode in ("plan", "all"):
        cmd_plan()
    if mode in ("fill", "all"):
        cmd_fill()
    if mode in ("finalize", "all"):
        cmd_finalize()


if __name__ == "__main__":
    main()
