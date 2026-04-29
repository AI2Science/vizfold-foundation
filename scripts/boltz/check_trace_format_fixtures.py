#!/usr/bin/env python3
"""
Offline check: VizFold-compatible Boltz trace text format (Issue #42).
Runs without Boltz/GPU — validates committed fixtures under scripts/boltz/fixtures/.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import validate_boltz_traces as vbt


def main() -> int:
    fix_dir = os.path.join(_HERE, "fixtures")
    if not os.path.isdir(fix_dir):
        print(f"[FAIL] missing fixtures dir: {fix_dir}", file=sys.stderr)
        return 1
    txts = sorted(
        f for f in os.listdir(fix_dir)
        if f.endswith(".txt") and os.path.isfile(os.path.join(fix_dir, f))
    )
    if not txts:
        print(f"[FAIL] no .txt fixtures in {fix_dir}", file=sys.stderr)
        return 1
    for name in txts:
        path = os.path.join(fix_dir, name)
        vbt.check_trace_file(path)
        nheads = vbt.count_heads_in_trace_file(path)
        print(f"[OK] {name} ({nheads} head block(s))")
    print(f"[OK] all {len(txts)} reference trace fixture(s) valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
