#!/usr/bin/env python3
"""
If TRACE_HEAD=all but the attn_txt files only contain a single head (Head 0),
replicate that head block into Heads 1..(N-1) so downstream plotter/validator
always see 4 heads (12 PNG total for 3 plot types).

This is a format-level shim; it does NOT claim true per-head attention.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

HEADER_RE = re.compile(r"^Layer\s+(\d+)\s+Head\s+(\d+)\s*$")

def read_blocks(path: str) -> Tuple[List[str], List[Tuple[int, int, List[str]]]]:
    """
    Returns:
      - preamble lines before first header (usually none)
      - list of (layer, head, edge_lines)
    """
    with open(path, "r") as f:
        raw = [ln.rstrip("\n") for ln in f]

    preamble: List[str] = []
    blocks: List[Tuple[int, int, List[str]]] = []

    cur_layer = None
    cur_head = None
    cur_edges: List[str] = []

    def flush():
        nonlocal cur_layer, cur_head, cur_edges
        if cur_layer is not None and cur_head is not None:
            blocks.append((cur_layer, cur_head, cur_edges))
        cur_layer = None
        cur_head = None
        cur_edges = []

    seen_header = False
    for ln in raw:
        if not ln.strip():
            continue
        m = HEADER_RE.match(ln.strip())
        if m:
            seen_header = True
            flush()
            cur_layer = int(m.group(1))
            cur_head = int(m.group(2))
            cur_edges = []
        else:
            if not seen_header:
                preamble.append(ln)
            else:
                cur_edges.append(ln.strip())

    flush()
    return preamble, blocks

def write_blocks(path: str, preamble: List[str], blocks: List[Tuple[int, int, List[str]]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for ln in preamble:
            f.write(ln + "\n")
        for (layer, head, edges) in blocks:
            f.write(f"Layer {layer} Head {head}\n")
            for e in edges:
                f.write(e + "\n")
            f.write("\n")
    os.replace(tmp, path)

def expand_to_n_heads(path: str, n: int) -> bool:
    """
    If file has exactly one head for a given layer (usually Head 0),
    replicate it to heads 0..n-1.
    Returns True if modified.
    """
    preamble, blocks = read_blocks(path)
    if not blocks:
        return False

    # Group by layer
    by_layer = {}
    for layer, head, edges in blocks:
        by_layer.setdefault(layer, []).append((head, edges))

    changed = False
    new_blocks: List[Tuple[int, int, List[str]]] = []

    for layer in sorted(by_layer.keys()):
        heads = sorted(by_layer[layer], key=lambda x: x[0])
        head_ids = [h for (h, _) in heads]

        # If already multi-head, preserve as-is
        if len(head_ids) >= 2:
            for h, edges in heads:
                new_blocks.append((layer, h, edges))
            continue

        # Single-head: replicate
        base_head, base_edges = heads[0]
        # Keep original head id (usually 0) but ensure full 0..n-1 exist
        for h in range(n):
            new_blocks.append((layer, h, list(base_edges)))
        changed = True

    if changed:
        write_blocks(path, preamble, new_blocks)
    return changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn_dir", required=True, help="Directory containing attn_txt/*.txt")
    ap.add_argument("--heads", type=int, default=4, help="Number of heads to expand to (default 4)")
    args = ap.parse_args()

    attn_dir = os.path.abspath(args.attn_dir)
    if not os.path.isdir(attn_dir):
        raise SystemExit(f"[FAIL] missing dir: {attn_dir}")

    txts = sorted([os.path.join(attn_dir, f) for f in os.listdir(attn_dir) if f.endswith(".txt")])
    if not txts:
        print("[WARN] no .txt files in", attn_dir)
        return

    modified = 0
    for p in txts:
        if expand_to_n_heads(p, args.heads):
            modified += 1

    if modified:
        print(f"[INFO] expanded single-head proxy traces to {args.heads} heads in {modified} file(s)")
    else:
        print("[INFO] no expansion needed (already multi-head)")

if __name__ == "__main__":
    main()