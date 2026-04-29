"""
make_sample_trace.py — one-time setup script for the VizFold sample trace.

Run from the repo root:
    python webui/make_sample_trace.py

Copies the 6KWC PDB and FASTA from the examples/ directory into
webui/sample_trace/6KWC/ and writes synthetic attention files so the
Streamlit app can run immediately without a real inference trace.
"""

import os
import shutil
import random
import math

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(WEBUI_DIR, "sample_trace", "6KWC")

PDB_SRC = os.path.join(REPO_ROOT, "examples", "monomer", "sample_predictions",
                        "6KWC_1_model_1_ptm_relaxed.pdb")
FASTA_SRC = os.path.join(REPO_ROOT, "examples", "monomer", "fasta_dir", "6kwc.fasta")

LAYERS = [0, 10, 20, 30, 40, 47]
N_HEADS = 8
N_RESIDUES = 190
TOP_K = 100
TRIANGLE_RESIDUES = [18, 39, 51, 79, 138, 159]

random.seed(42)


def softmax_weights(n: int, bias_center: int | None = None) -> list[float]:
    """Generate plausible attention-like weights (peaked distribution)."""
    logits = [random.gauss(0, 1) for _ in range(n)]
    if bias_center is not None:
        for i in range(n):
            dist = abs(i - bias_center)
            logits[i] += max(0, 3.0 - dist * 0.15)
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def generate_connections(
    n_residues: int,
    n_connections: int,
    bias_residue: int | None = None,
) -> list[tuple[int, int, float]]:
    row_weights = softmax_weights(n_residues, bias_residue)
    connections: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()

    while len(connections) < n_connections:
        r1 = random.choices(range(n_residues), weights=row_weights)[0]
        col_weights = softmax_weights(n_residues, r1)
        r2 = random.choices(range(n_residues), weights=col_weights)[0]
        if r1 == r2:
            continue
        key = (min(r1, r2), max(r1, r2))
        if key in seen:
            continue
        seen.add(key)
        weight = random.uniform(0.001, 0.05) + random.random() * 0.02
        connections.append((r1, r2, weight))

    connections.sort(key=lambda x: x[2], reverse=True)
    return connections


def write_heads_file(path: str, layer_idx: int, n_heads: int,
                     n_residues: int, top_k: int,
                     bias_residue: int | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for head in range(n_heads):
            conns = generate_connections(n_residues, top_k, bias_residue)
            f.write(f"Layer {layer_idx}, Head {head}\n")
            for r1, r2, w in conns:
                f.write(f"{r1} {r2} {w:.6f}\n")
    print(f"  Written: {os.path.relpath(path, REPO_ROOT)}")


def main() -> None:
    print("VizFold — setting up sample trace for 6KWC\n")

    attn_dir = os.path.join(OUT_DIR, "attention")
    os.makedirs(attn_dir, exist_ok=True)

    # --- PDB ---
    pdb_dst = os.path.join(OUT_DIR, "6KWC.pdb")
    if os.path.exists(PDB_SRC):
        shutil.copy2(PDB_SRC, pdb_dst)
        print(f"  Copied: {os.path.relpath(pdb_dst, REPO_ROOT)}")
    else:
        print(f"  [WARN] PDB source not found: {PDB_SRC}")

    # --- FASTA ---
    fasta_dst = os.path.join(OUT_DIR, "6KWC.fasta")
    if os.path.exists(FASTA_SRC):
        shutil.copy2(FASTA_SRC, fasta_dst)
        print(f"  Copied: {os.path.relpath(fasta_dst, REPO_ROOT)}")
    else:
        print(f"  [WARN] FASTA source not found: {FASTA_SRC}")

    print()

    # --- MSA row attention ---
    print("Generating MSA row attention files...")
    for layer in LAYERS:
        path = os.path.join(attn_dir, f"msa_row_attn_layer{layer}.txt")
        write_heads_file(path, layer, N_HEADS, N_RESIDUES, TOP_K)

    print()

    # --- Triangle start attention ---
    print("Generating triangle start attention files...")
    for layer in [47]:
        for res_idx in TRIANGLE_RESIDUES:
            path = os.path.join(
                attn_dir,
                f"triangle_start_attn_layer{layer}_residue_idx_{res_idx}.txt",
            )
            write_heads_file(path, layer, N_HEADS, N_RESIDUES, TOP_K,
                             bias_residue=res_idx)

    print()
    print("Done. Start the app with:")
    print("  streamlit run webui/app.py")


if __name__ == "__main__":
    main()
