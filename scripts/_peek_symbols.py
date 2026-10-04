#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""看一眼 data/symbols.json 的结构（键名与样例）。"""
import json
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "data" / "symbols.json"
j = json.loads(p.read_text(encoding="utf-8"))
print("顶层键:", list(j.keys()))
for k, v in j.items():
    print(f"  {k}: {len(v) if hasattr(v, '__len__') else type(v)} 条")
    if isinstance(v, list) and v:
        print("    样例:", json.dumps(v[0], ensure_ascii=False))
