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

def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run directory containing attn_txt/ and arc_png/")
    ap.add_argument("--layers", default="0", help="comma-separated layer indices")
    ap.add_argument("--residues", default="18", help="comma-separated residue indices")
    ap.add_argument("--expect_heads", type=int, default=4, help="expected heads per file")
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

    for p in required:
        check_trace_file(p)

    if os.path.isdir(arc_dir):
        pngs = glob.glob(os.path.join(arc_dir, "*.png"))
        expected_min = len(layers)*args.expect_heads + 2*len(layers)*len(residues)*args.expect_heads
        if len(pngs) < expected_min:
            print(f"[WARN] arc_png has {len(pngs)} PNGs, expected >= {expected_min}")
        else:
            print(f"[OK] arc_png count looks reasonable: {len(pngs)} (expected >= {expected_min})")
    else:
        print(f"[WARN] missing arc_png dir: {arc_dir}")

    print("[OK] trace files validated:", len(required))

if __name__ == "__main__":
    main()