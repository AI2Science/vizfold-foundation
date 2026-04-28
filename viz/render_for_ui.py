"""Generate the heatmap PNGs that ``web_interface.py`` serves.

The Flask UI in :mod:`web_interface` expects, for every layer ``N`` in
``[0, 48)`` and every attention type in ``{msa_row, triangle_start}``:

    IMAGE_DIR/heatmap_<attn_type>_layer{N}.png

This module reads the sparse attention text files written by
``save_attention_topk`` during inference (real mode), reconstructs an
N x N attention map per layer (mean over heads), and renders it with
:func:`viz.plot_heatmap`.

If no text files are present (typical on a developer machine without a
fresh model run), the script falls back to synthetic data from
:mod:`viz._fakes` so the UI still has 96 PNGs to display while extraction
catches up.

Usage
-----
::

    python -m viz.render_for_ui                                # auto
    python -m viz.render_for_ui --mode demo --n-res 96         # forced synthetic
    python -m viz.render_for_ui --mode real \\
        --attn-dir outputs/attention_files_6KWC_demo_tri_18    # real
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # never pop a window
import matplotlib.pyplot as plt
import numpy as np

from viz._fakes import fake_attention_heads
from viz.plots.heatmap import plot_heatmap

ATTN_TYPES = ("msa_row", "triangle_start")
NUM_LAYERS = 48
DEFAULT_PROT = "6KWC"
DEFAULT_TRI_IDX = 18


def _attn_text_path(
    attn_dir: str, attn_type: str, layer: int, tri_idx: int
) -> str:
    """Return the on-disk path of the sparse-attention text file."""
    if attn_type == "msa_row":
        return os.path.join(attn_dir, f"msa_row_attn_layer{layer}.txt")
    if attn_type == "triangle_start":
        return os.path.join(
            attn_dir, f"triangle_start_attn_layer{layer}_residue_idx_{tri_idx}.txt"
        )
    raise ValueError(f"Unknown attention type {attn_type!r}")


def _heatmap_image_path(image_dir: str, attn_type: str, layer: int) -> str:
    return os.path.join(image_dir, f"heatmap_{attn_type}_layer{layer}.png")


def _parse_topk_text(path: str) -> Dict[int, List[Tuple[int, int, float]]]:
    """Parse a ``save_attention_topk`` text file into per-head triples.

    File format::

        Layer L Head H
        res1 res2 weight
        res1 res2 weight
        Layer L Head H+1
        ...
    """
    heads: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    current: Optional[int] = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("layer"):
                parts = line.replace(",", "").split()
                current = int(parts[-1])
                continue
            if current is None:
                raise ValueError(f"value line before any 'Layer' header in {path!r}")
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"expected 'res1 res2 weight' in {path!r}, got {line!r}"
                )
            try:
                r1, r2, w = int(parts[0]), int(parts[1]), float(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"malformed data line in {path!r}: {line!r}"
                ) from exc
            heads[int(current)].append((r1, r2, w))
    return heads


def _matrix_from_heads(
    heads: Dict[int, List[Tuple[int, int, float]]],
    n_res: int,
) -> np.ndarray:
    """Mean-over-heads N x N reconstruction from sparse top-K triples."""
    if not heads:
        return np.zeros((n_res, n_res), dtype=np.float32)
    acc = np.zeros((len(heads), n_res, n_res), dtype=np.float32)
    skipped = 0
    for hi, (_, conns) in enumerate(sorted(heads.items())):
        for r1, r2, w in conns:
            if 0 <= r1 < n_res and 0 <= r2 < n_res:
                acc[hi, r1, r2] = w
            else:
                skipped += 1
    if skipped:
        print(
            f"[render_for_ui] warning: dropped {skipped} out-of-bounds entries "
            f"(n_res={n_res})"
        )
    return acc.mean(axis=0)


def _infer_n_res_from_files(attn_dir: str, tri_idx: int) -> Optional[int]:
    """Return max residue index + 1 across all available text files."""
    found = []
    for layer in range(NUM_LAYERS):
        for attn_type in ATTN_TYPES:
            path = _attn_text_path(attn_dir, attn_type, layer, tri_idx)
            if not os.path.exists(path):
                continue
            try:
                heads = _parse_topk_text(path)
            except Exception:
                continue
            for conns in heads.values():
                for r1, r2, _ in conns:
                    found.append(max(r1, r2))
    if not found:
        return None
    return max(found) + 1


def render_real(
    attn_dir: str,
    image_dir: str,
    *,
    n_res: Optional[int] = None,
    tri_idx: int = DEFAULT_TRI_IDX,
    protein: str = DEFAULT_PROT,
) -> List[str]:
    """Build heatmap PNGs from real ``save_attention_topk`` text files.

    Returns the list of paths actually written.
    """
    if n_res is None:
        n_res = _infer_n_res_from_files(attn_dir, tri_idx)
    if n_res is None:
        raise FileNotFoundError(
            f"No attention text files found under {attn_dir!r}; "
            "run inference with --demo_attn first or use --mode demo."
        )

    os.makedirs(image_dir, exist_ok=True)
    written: List[str] = []
    for layer in range(NUM_LAYERS):
        for attn_type in ATTN_TYPES:
            txt = _attn_text_path(attn_dir, attn_type, layer, tri_idx)
            if not os.path.exists(txt):
                continue
            heads = _parse_topk_text(txt)
            mat = _matrix_from_heads(heads, n_res=n_res)
            out = _heatmap_image_path(image_dir, attn_type, layer)
            label = "MSA Row" if attn_type == "msa_row" else "Triangle Start"
            extra = (
                f" (residue {tri_idx})" if attn_type == "triangle_start" else ""
            )
            fig = plot_heatmap(
                mat,
                title=f"{protein} {label} Attention — Layer {layer}{extra} (mean over heads)",
                colorbar_label="attention weight",
                save_path=out,
            )
            plt.close(fig)
            written.append(out)
    return written


def render_demo(
    image_dir: str,
    *,
    n_res: int = 96,
    n_heads: int = 8,
    seed: int = 0,
    protein: str = DEFAULT_PROT,
    tri_idx: int = DEFAULT_TRI_IDX,
) -> List[str]:
    """Synthetic placeholders so the UI works before a real model run.

    Each layer gets a different ``viz._fakes.fake_attention_heads`` realization
    by mixing the layer index into the seed; the matrix saved is the
    head-averaged map, matching the real-mode output shape.
    """
    os.makedirs(image_dir, exist_ok=True)
    written: List[str] = []
    for layer in range(NUM_LAYERS):
        for attn_type in ATTN_TYPES:
            heads = fake_attention_heads(
                N=n_res,
                H=n_heads,
                seed=seed + layer * 100 + (0 if attn_type == "msa_row" else 1),
            )
            mat = heads.mean(axis=0)
            out = _heatmap_image_path(image_dir, attn_type, layer)
            label = "MSA Row" if attn_type == "msa_row" else "Triangle Start"
            extra = (
                f" (residue {tri_idx})" if attn_type == "triangle_start" else ""
            )
            fig = plot_heatmap(
                mat,
                title=(
                    f"{protein} {label} Attention — Layer {layer}{extra} "
                    f"(SYNTHETIC PLACEHOLDER, mean over heads)"
                ),
                colorbar_label="attention weight",
                save_path=out,
            )
            plt.close(fig)
            written.append(out)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode",
        choices=("auto", "real", "demo"),
        default="auto",
        help="`real` reads attention text files; `demo` synthesizes; `auto` tries real then falls back.",
    )
    p.add_argument(
        "--attn-dir",
        default=f"outputs/attention_files_{DEFAULT_PROT}_demo_tri_{DEFAULT_TRI_IDX}",
        help="Directory of save_attention_topk text files.",
    )
    p.add_argument(
        "--image-dir",
        default=f"outputs/attention_images_{DEFAULT_PROT}_demo_tri_{DEFAULT_TRI_IDX}",
        help="Directory the Flask UI reads PNGs from.",
    )
    p.add_argument("--protein", default=DEFAULT_PROT)
    p.add_argument("--tri-idx", type=int, default=DEFAULT_TRI_IDX)
    p.add_argument(
        "--n-res",
        type=int,
        default=None,
        help="Sequence length. Inferred from text files in real mode; defaults to 96 in demo.",
    )
    p.add_argument("--seed", type=int, default=0, help="Demo-mode RNG seed.")
    args = p.parse_args(argv)

    image_dir = args.image_dir
    if args.mode == "real":
        written = render_real(
            args.attn_dir,
            image_dir,
            n_res=args.n_res,
            tri_idx=args.tri_idx,
            protein=args.protein,
        )
        mode_used = "real"
    elif args.mode == "demo":
        written = render_demo(
            image_dir,
            n_res=args.n_res or 96,
            seed=args.seed,
            protein=args.protein,
            tri_idx=args.tri_idx,
        )
        mode_used = "demo"
    else:
        try:
            written = render_real(
                args.attn_dir,
                image_dir,
                n_res=args.n_res,
                tri_idx=args.tri_idx,
                protein=args.protein,
            )
            mode_used = "real"
        except FileNotFoundError as e:
            print(f"[auto] {e}")
            print("[auto] falling back to demo (synthetic) mode.")
            written = render_demo(
                image_dir,
                n_res=args.n_res or 96,
                seed=args.seed,
                protein=args.protein,
                tri_idx=args.tri_idx,
            )
            mode_used = "demo"

    print(f"[{mode_used}] wrote {len(written)} PNGs into {image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
