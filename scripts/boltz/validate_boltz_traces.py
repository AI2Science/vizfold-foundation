#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys

LAYER_RE = re.compile(r"^Layer\s+(\d+)\s+Head\s+(\d+)\s*$")

def check_trace_file(path: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"[FAIL] missing file: {path}")
    with open(path, encoding="utf-8") as f:
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
    with open(path, encoding="utf-8") as f:
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

def _real_path(p: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(p.strip())))

def check_manifest_paths_match_run_dir(run_dir: str, man: dict) -> None:
    """Ensure manifest run_dir and outputs.* match this run tree (no stale copies)."""
    rd = _real_path(run_dir)
    m_run = _real_path(str(man["run_dir"]))
    if m_run != rd:
        raise ValueError(f"manifest run_dir {m_run!r} != --run_dir {rd!r}")
    outs = man.get("outputs") or {}
    expected_dirs = {
        "pred": os.path.join(rd, "pred"),
        "attn_txt": os.path.join(rd, "attn_txt"),
        "act_npz": os.path.join(rd, "act_npz"),
        "arc_png": os.path.join(rd, "arc_png"),
    }
    for key, exp in expected_dirs.items():
        got = _real_path(str(outs[key]))
        want = _real_path(exp)
        if got != want:
            raise ValueError(f"manifest outputs.{key} {got!r} != expected {want!r}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run directory containing attn_txt/ and arc_png/")
    ap.add_argument("--layers", default="0", help="comma-separated layer indices")
    ap.add_argument("--residues", default="18", help="comma-separated residue indices")
    # Backward compatibility: older runner passes this. We no longer require fixed heads.
    ap.add_argument("--expect_heads", type=int, default=None,
                    help="(deprecated) kept for backward compatibility; ignored in proxy-aware validation")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Require manifest.json paths to match --run_dir, component_status.json, non-empty pred/, valid act_npz if any .npz",
    )
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    attn_dir = os.path.join(run_dir, "attn_txt")
    arc_dir  = os.path.join(run_dir, "arc_png")

    if not os.path.isdir(attn_dir):
        raise SystemExit(f"[FAIL] missing dir: {attn_dir}")

    layers = parse_int_list(args.layers)
    residues = parse_int_list(args.residues)
    if not layers:
        raise SystemExit("[FAIL] --layers must list at least one layer index (got empty)")
    if not residues:
        raise SystemExit("[FAIL] --residues must list at least one residue index (got empty)")

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

    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                man = json.load(f)
            for key in ("run_dir", "outputs", "trace"):
                if key not in man:
                    raise KeyError(key)
            outs = man.get("outputs") or {}
            for sub in ("pred", "attn_txt", "act_npz", "arc_png"):
                if sub not in outs:
                    raise KeyError(f"outputs.{sub}")
            check_manifest_paths_match_run_dir(run_dir, man)
            print("[OK] manifest.json present, keys OK, paths match --run_dir")
        except Exception as e:
            msg = f"[FAIL] manifest.json invalid: {e}"
            if args.strict:
                raise SystemExit(msg)
            print(f"[WARN] {msg}")
    else:
        msg = "[FAIL] missing manifest.json"
        if args.strict:
            raise SystemExit(msg)
        print(f"[WARN] {msg}")

    pred_dir = os.path.join(run_dir, "pred")
    pred_files: list[str] = []
    if os.path.isdir(pred_dir):
        for root, _dirs, names in os.walk(pred_dir):
            for n in names:
                p = os.path.join(root, n)
                if os.path.isfile(p):
                    pred_files.append(p)
        if pred_files:
            print(f"[OK] pred/ has {len(pred_files)} file(s) under tree (Boltz structure outputs)")
        else:
            msg = "[FAIL] pred/ has no files (no structural outputs from boltz predict)"
            if args.strict:
                raise SystemExit(msg)
            print(f"[WARN] {msg}")
    else:
        msg = "[FAIL] missing pred/ directory"
        if args.strict:
            raise SystemExit(msg)
        print(f"[WARN] {msg}")

    status_path = os.path.join(attn_dir, "component_status.json")
    if os.path.isfile(status_path):
        try:
            with open(status_path, encoding="utf-8") as f:
                st = json.load(f)
            for comp in ("msa", "pairformer_boltz", "sm_boltz"):
                if comp not in st:
                    raise KeyError(comp)
                ent = st[comp]
                for k in ("available", "source", "files_written"):
                    if k not in ent:
                        raise KeyError(f"{comp}.{k}")
            print("[OK] component_status.json present and well-formed")
        except Exception as e:
            msg = f"[FAIL] component_status.json invalid: {e}"
            if args.strict:
                raise SystemExit(msg)
            print(f"[WARN] {msg}")
    else:
        msg = "[FAIL] missing attn_txt/component_status.json"
        if args.strict:
            raise SystemExit(msg)
        print(f"[WARN] {msg}")

    act_pf = os.path.join(run_dir, "act_npz", "pairformer_boltz")
    npz_paths = sorted(glob.glob(os.path.join(act_pf, "*.npz")))
    if npz_paths:
        try:
            import numpy as np
        except ImportError:
            print("[WARN] numpy not installed; skipping act_npz content checks")
        else:
            for p in npz_paths:
                try:
                    z = np.load(p)
                    if "pair_norm" not in z.files or "pair_slice" not in z.files:
                        raise ValueError(f"{os.path.basename(p)} missing pair_norm or pair_slice")
                    pn = z["pair_norm"]
                    ps = z["pair_slice"]
                    if pn.ndim != 2:
                        raise ValueError(f"pair_norm ndim={pn.ndim}, expected 2")
                    if ps.ndim != 3:
                        raise ValueError(f"pair_slice ndim={ps.ndim}, expected 3")
                    if ps.shape[-1] != 8:
                        raise ValueError(f"pair_slice last dim {ps.shape[-1]}, expected 8")
                    if pn.shape[0] != pn.shape[1]:
                        raise ValueError("pair_norm not square")
                    if ps.shape[0] != ps.shape[1]:
                        raise ValueError("pair_slice not square in N,N")
                    if pn.shape[0] != ps.shape[0]:
                        raise ValueError("pair_norm and pair_slice N dimension mismatch")
                except Exception as e:
                    msg = f"[FAIL] act_npz {os.path.basename(p)}: {e}"
                    if args.strict:
                        raise SystemExit(msg)
                    print(f"[WARN] {msg}")
                    break
            else:
                print(f"[OK] act_npz/pairformer_boltz: validated {len(npz_paths)} npz file(s)")
    else:
        print("[INFO] no act_npz/pairformer_boltz/*.npz (optional summaries absent)")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)