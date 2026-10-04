#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键界面冒烟：起本地服务 → 等就绪 → 跑 smoke_ui.js → **必定关服务**。

## 为什么要有这个脚本

本机规矩（用户级记忆）：**长驻服务禁止用「后台任务」方式启动**。
理由：后台任务的"完成"由子进程是否退出判定，而 HTTP 服务永不退出 →
任务卡片会永远停在"运行中"，看起来像卡死，实际上服务在正常干活。

所以这里把服务放进**本脚本自己的子进程**，跑完冒烟就地关闭 ——
整个脚本会正常退出、不留僵尸，也就没有"卡片挂着"的问题。

## 用法
    python scripts/smoke_all.py            # 端口 8765
    python scripts/smoke_all.py --port 8799
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = Path(sys.executable)
NODE_CANDIDATES = [
    Path(r"C:\Users\Administrator\.workbuddy\binaries\node\versions\22.22.2-3\node.exe"),
    Path(r"C:\Program Files\nodejs\node.exe"),
]


def port_open(port: int, host: str = "127.0.0.1", t: float = 0.6) -> bool:
    s = socket.socket()
    s.settimeout(t)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def find_node() -> Path:
    for p in NODE_CANDIDATES:
        if p.exists():
            return p
    sys.exit("找不到 node（改 NODE_CANDIDATES）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--wait", type=float, default=60.0)
    args = ap.parse_args()

    node = find_node()
    log = ROOT / "results" / "_serve_test.log"
    proc: subprocess.Popen | None = None

    if port_open(args.port):
        print(f"端口 {args.port} 上已有服务在跑 → 直接复用它（不新起、也不关它）")
    else:
        fh = open(log, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [str(PY), "-u", str(ROOT / "scripts" / "serve.py"), "--port", str(args.port)],
            cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < args.wait:
            if port_open(args.port):
                break
            if proc.poll() is not None:                     # 启动即退出 = 配置有问题
                fh.close()
                print(f"✗ 服务启动后立刻退出，日志：\n{log.read_text(encoding='utf-8')[:2000]}")
                return 1
            time.sleep(0.4)
        else:
            proc.terminate()
            fh.close()
            print(f"✗ 服务 {args.wait:.0f} 秒没起来，日志：\n"
                  f"{log.read_text(encoding='utf-8')[:2000]}")
            return 1
        print(f"服务已就绪（{time.time() - t0:.1f}s，日志 {log.name}）")

    try:
        print("-" * 74)
        rc = subprocess.run([str(node), str(ROOT / "scripts" / "smoke_ui.js")],
                            cwd=ROOT).returncode
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("-" * 74)
            print("服务已关闭（本脚本起的，用完就关 → 不留僵尸进程/卡片）")
    return rc


if __name__ == "__main__":
    sys.exit(main())
