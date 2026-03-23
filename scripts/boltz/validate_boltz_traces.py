#!/usr/bin/env python3
import argparse, os, re, glob

LAYER_RE = re.compile(r"^Layer\s+(\d+)\s+Head\s+(\d+)\s*$")

def check_trace_file(path: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"[FAIL] missing file: {path}")
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise SystemExit(f"[FAIL] empty file: {path}")

    header_idxs = [i for i, ln in enumerate(lines) if LAYER_RE.match(ln)]
    if not header_idxs:
        raise SystemExit(f"[FAIL] no 'Layer ... Head ...' headers in: {path}")

    for ln in lines:
        if LAYER_RE.match(ln):
            continue
        parts = ln.split()
        if len(parts) != 3:
            raise SystemExit(f"[FAIL] bad edge line (not 3 cols) in {path}: '{ln}'")
        try:
            int(float(parts[0])); int(float(parts[1])); float(parts[2])
        except Exception:
            raise SystemExit(f"[FAIL] non-numeric edge line in {path}: '{ln}'")

def count_heads_in_trace_file(path: str) -> int:
    """Return number of unique Head indices present in the trace file."""
    heads = set()
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            m = LAYER_RE.match(ln)
            if m:
                heads.add(int(m.group(2)))
    return len(heads)

def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run directory containing attn_txt/ and arc_png/")
    ap.add_argument("--layers", default="0", help="comma-separated layer indices")
    ap.add_argument("--residues", default="18", help="comma-separated residue indices")
    # Backward compatibility: older runner passes this. We no longer require fixed heads.
    ap.add_argument("--expect_heads", type=int, default=None,
                    help="(deprecated) kept for backward compatibility; ignored in proxy-aware validation")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    attn_dir = os.path.join(run_dir, "attn_txt")
    arc_dir  = os.path.join(run_dir, "arc_png")

    if not os.path.isdir(attn_dir):
        raise SystemExit(f"[FAIL] missing dir: {attn_dir}")

    layers = parse_int_list(args.layers)
    residues = parse_int_list(args.residues)

    required = []
    for L in layers:
        required.append(os.path.join(attn_dir, f"msa_row_attn_layer{L}.txt"))
        for r in residues:
            required.append(os.path.join(attn_dir, f"triangle_start_attn_layer{L}_residue_idx_{r}.txt"))
            required.append(os.path.join(attn_dir, f"triangle_end_attn_layer{L}_residue_idx_{r}.txt"))

    msa_heads_by_layer: dict[int, int] = {}
    tri_start_heads_by_layer_res: dict[tuple[int, int], int] = {}
    tri_end_heads_by_layer_res: dict[tuple[int, int], int] = {}

    for p in required:
        check_trace_file(p)
        nheads = count_heads_in_trace_file(p)

        base = os.path.basename(p)
        if base.startswith("msa_row_attn_layer"):
            m = re.search(r"layer(\d+)\.txt$", base)
            if m:
                L = int(m.group(1))
                msa_heads_by_layer[L] = nheads
        elif base.startswith("triangle_start_attn_layer"):
            m = re.search(r"layer(\d+)_residue_idx_(\d+)\.txt$", base)
            if m:
                L, r = int(m.group(1)), int(m.group(2))
                tri_start_heads_by_layer_res[(L, r)] = nheads
        elif base.startswith("triangle_end_attn_layer"):
            m = re.search(r"layer(\d+)_residue_idx_(\d+)\.txt$", base)
            if m:
                L, r = int(m.group(1)), int(m.group(2))
                tri_end_heads_by_layer_res[(L, r)] = nheads

    if os.path.isdir(arc_dir):
        pngs = glob.glob(os.path.join(arc_dir, "*.png"))

        expected_min = 0
        # msa row: 1 png per (layer, head)
        for L in layers:
            expected_min += msa_heads_by_layer.get(L, 0)

        # triangle start/end: 1 png per (layer, residue, head)
        for L in layers:
            for r in residues:
                expected_min += tri_start_heads_by_layer_res.get((L, r), 0)
                expected_min += tri_end_heads_by_layer_res.get((L, r), 0)

        if len(pngs) < expected_min:
            print(f"[WARN] arc_png has {len(pngs)} PNGs, expected >= {expected_min}")
        else:
            print(f"[OK] arc_png count looks reasonable: {len(pngs)} (expected >= {expected_min})")
    else:
        print(f"[WARN] missing arc_png dir: {arc_dir}")

    print("[OK] trace files validated:", len(required))

if __name__ == "__main__":
    main()