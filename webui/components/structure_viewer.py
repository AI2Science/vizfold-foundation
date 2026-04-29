"""
3D protein structure viewer powered by py3Dmol.
Residues are colored by aggregated attention score (white → red gradient).
"""

from __future__ import annotations
from typing import List, Tuple

import os

import numpy as np
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components


def render_structure(
    pdb_path: str | None,
    connections: List[Tuple[int, int, float]],
    n_residues: int,
    height: int = 480,
    pdb_string: str | None = None,
) -> None:
    if pdb_string is not None:
        pdb_str = pdb_string
    elif pdb_path and os.path.exists(pdb_path):
        with open(pdb_path) as f:
            pdb_str = f.read()
    else:
        st.info("No structure file available.")
        return

    scores = _residue_scores(connections, n_residues)

    view = py3Dmol.view(width="100%", height=height)
    view.addModel(pdb_str, "pdb")
    view.setStyle({"cartoon": {"color": "#cccccc"}})
    view.setBackgroundColor("#0e1117")

    if scores.max() > 0:
        normed = scores / scores.max()
        for resi_0, t in enumerate(normed):
            if t > 0.02:
                color = _score_to_hex(float(t))
                view.addStyle(
                    {"resi": resi_0 + 1},
                    {"cartoon": {"color": color}},
                )

    view.zoomTo()

    # stmol is optional; fall back to raw HTML embed
    try:
        from stmol import showmol  # type: ignore
        showmol(view, height=height + 20)
    except ImportError:
        components.html(view._make_html(), height=height + 20, scrolling=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _residue_scores(
    connections: List[Tuple[int, int, float]], n_residues: int
) -> np.ndarray:
    scores = np.zeros(n_residues)
    for r1, r2, w in connections:
        if r1 < n_residues:
            scores[r1] += w
        if r2 < n_residues:
            scores[r2] += w
    return scores


def _score_to_hex(t: float) -> str:
    """Map t ∈ [0, 1] to a white→orange→red gradient."""
    if t < 0.5:
        s = t * 2
        r = int(255)
        g = int(255 * (1 - s * 0.5))
        b = int(255 * (1 - s))
    else:
        s = (t - 0.5) * 2
        r = int(255)
        g = int(255 * 0.5 * (1 - s))
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"
