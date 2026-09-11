"""
Arc diagram — residue-to-residue attention arcs drawn with Matplotlib.
Adapted from visualize_attention_arc_diagram_demo_utils.py but returns
a Figure instead of saving to disk, for use inside Streamlit.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def render_arc_diagram(
    connections: List[Tuple[int, int, float]],
    fasta_seq: str,
    highlight_residue: Optional[int] = None,
) -> None:
    if not connections or not fasta_seq:
        st.info("No arc data to display.")
        return
    fig = _build_figure(connections, fasta_seq, highlight_residue)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Internal ──────────────────────────────────────────────────────────────────

def _build_figure(
    connections: List[Tuple[int, int, float]],
    fasta_seq: str,
    highlight_residue: Optional[int],
) -> plt.Figure:
    n = len(fasta_seq)
    weights = [w for _, _, w in connections]
    w_min, w_max = min(weights), max(weights)
    w_range = (w_max - w_min) or 1.0

    fig_w = max(14, n // 7)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    for r1, r2, w in connections:
        x1, x2 = r1 + 0.5, r2 + 0.5
        height = abs(x2 - x1) / 2
        norm_w = (w - w_min) / w_range
        lw = 0.3 + norm_w * 2.0
        intensity = 0.35 + 0.65 * norm_w
        color = (0.15, 0.45 * (1 - norm_w * 0.5), intensity)

        xs = np.linspace(x1, x2, 80)
        ys = height * np.sin(np.linspace(0, np.pi, 80))
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.85, solid_capstyle="round")

    ax.set_xlim(0, n)
    ax.set_ylim(0, None)
    ax.set_xticks(np.arange(n) + 0.5)
    labels = ax.set_xticklabels(
        list(fasta_seq), fontsize=max(5, min(8, 120 // n)),
        color="#aaaaaa", ha="center",
    )

    if highlight_residue is not None and 0 <= highlight_residue < len(labels):
        labels[highlight_residue].set_color("#ff4b4b")
        labels[highlight_residue].set_fontweight("bold")

    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_ylabel("Attention strength", color="#aaaaaa", fontsize=9)
    ax.yaxis.label.set_color("#aaaaaa")

    plt.tight_layout(pad=0.5)
    return fig
