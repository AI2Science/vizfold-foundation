"""
VizFold — Streamlit UI for offline protein model internals exploration.

Run:
    streamlit run webui/app.py
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from components.structure_viewer import render_structure
from components.attention_heatmap import render_heatmap
from components.arc_diagram import render_arc_diagram

from trace_reader import TraceReader, ZarrTraceReader
from visualization_adapter import (
    flatten_attention_heads,
    build_visualization_payload,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VizFold",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #111827; }
    [data-testid="stSidebar"] .stMarkdown p { color: #9ca3af; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Zarr session-state cache ──────────────────────────────────────────────────

def _get_zarr_reader(uploaded_file) -> ZarrTraceReader:
    """Cache ZarrTraceReader in session state, keyed by file hash."""
    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    if st.session_state.get("_zarr_hash") != file_hash:
        st.session_state["_zarr_hash"] = file_hash
        st.session_state["_zarr_reader"] = ZarrTraceReader(file_bytes)
    return st.session_state["_zarr_reader"]

# ── Sidebar ───────────────────────────────────────────────────────────────────

# Shared rendering outputs — populated by whichever source branch runs
connections: list = []
fasta_seq: str = ""
n_residues: int = 0
pdb_path: str | None = None
pdb_string: str | None = None
protein: str = ""
head_label: str = ""
attn_badge: str = ""
residue_idx: int | None = None

with st.sidebar:
    st.markdown("## 🔬 VizFold")
    st.caption("Protein model internals explorer")
    st.divider()

    source = st.radio(
        "Data source",
        ["Trace directory", "Zarr archive"],
        help="Load from a local trace folder, or upload a Zarr ZipStore (.zip).",
    )
    st.divider()

    # ── Branch A: Trace directory ─────────────────────────────────────────────
    if source == "Trace directory":
        default_trace = os.path.join(os.path.dirname(__file__), "sample_trace")
        trace_dir = st.text_input(
            "Trace directory",
            value=default_trace,
            help="Root folder containing per-protein trace subdirectories.",
        )

        reader = TraceReader(trace_dir)
        proteins = reader.list_proteins()

        if not proteins:
            st.error(
                "No proteins found. Check the path, or run "
                "`python webui/make_sample_trace.py` first."
            )
            st.stop()

        protein = st.selectbox("Protein", proteins)
        fasta_seq = reader.get_fasta_sequence(protein)
        n_residues = len(fasta_seq)
        if n_residues:
            st.caption(f"{n_residues} residues")

        st.divider()
        st.markdown("**Attention**")

        attn_type = st.radio(
            "Type",
            ["msa_row", "triangle_start"],
            format_func=lambda x: "MSA Row" if x == "msa_row" else "Triangle Start",
        )

        layers = reader.list_layers(protein, attn_type)
        if not layers:
            st.warning(f"No `{attn_type}` attention files found for **{protein}**.")
            st.stop()

        layer_idx = st.select_slider("Layer", options=layers, value=layers[-1])

        if attn_type == "triangle_start":
            available_res = reader.list_triangle_residues(protein, layer_idx)
            if not available_res:
                st.warning("No triangle attention files found for this layer.")
                st.stop()
            residue_idx = st.select_slider(
                "Source residue", options=available_res, value=available_res[0]
            )

        top_k = st.slider("Top-K connections", 10, 200, 50, step=10)

        if attn_type == "triangle_start" and residue_idx is not None:
            attn_data = reader.load_triangle_attention(protein, layer_idx, residue_idx, top_k)
        else:
            attn_data = reader.load_attention(protein, attn_type, layer_idx, top_k)

        n_heads = len(attn_data)
        head_sel = st.selectbox(
            "Head",
            ["Average"] + list(range(n_heads)),
            format_func=lambda x: "All heads averaged" if x == "Average" else f"Head {x}",
        )

        if head_sel == "Average":
            connections = flatten_attention_heads(attn_data)
            head_label = "All heads averaged"
        else:
            connections = attn_data.get(int(head_sel), [])
            head_label = f"Head {head_sel}"

        pdb_path = reader.get_pdb_path(protein)
        attn_badge = (
            "MSA Row" if attn_type == "msa_row"
            else f"Triangle Start (res {residue_idx})"
        )

    # ── Branch B: Zarr archive ────────────────────────────────────────────────
    else:
        uploaded = st.file_uploader(
            "Upload Zarr archive",
            type=["zip"],
            help=(
                "Upload a Zarr ZipStore containing attention arrays. "
                "Expected array shape: [n_layers, n_heads, N, N]. "
                "Optional keys: `sequence` (amino acid string), `structure_pdb` (PDB text)."
            ),
        )

        if not uploaded:
            st.info("Upload a `.zip` Zarr archive to continue.")
            st.stop()

        with st.spinner("Opening Zarr store…"):
            zarr_reader = _get_zarr_reader(uploaded)

        all_arrays = zarr_reader.list_all_arrays()
        attn_arrays = zarr_reader.list_attention_arrays()

        with st.expander("Zarr contents", expanded=not attn_arrays):
            if all_arrays:
                for name, shape in all_arrays.items():
                    marker = " ✓" if name in attn_arrays else ""
                    st.code(f"{name}: {shape}{marker}", language=None)
            else:
                st.warning("No arrays found in this Zarr store.")

        if not attn_arrays:
            st.warning(
                "No N×N attention arrays detected (shape [..., N, N]). "
                "Select any array below — it will be treated as attention."
            )
            candidate_arrays = all_arrays
        else:
            candidate_arrays = attn_arrays

        if not candidate_arrays:
            st.error("No arrays to display.")
            st.stop()

        array_name = st.selectbox(
            "Attention array",
            list(candidate_arrays.keys()),
        )

        _n_layers = zarr_reader.n_layers(array_name)
        _n_heads = zarr_reader.n_heads(array_name)
        n_residues = zarr_reader.n_residues(array_name)
        protein = uploaded.name.rsplit(".", 1)[0]

        fasta_seq = zarr_reader.get_sequence()
        if fasta_seq and len(fasta_seq) != n_residues:
            st.caption(
                f"Note: sequence length ({len(fasta_seq)}) ≠ array dim ({n_residues}). "
                "Using placeholder residue labels."
            )
            fasta_seq = ""
        if not fasta_seq:
            fasta_seq = "X" * n_residues

        st.caption(f"{n_residues} residues · {_n_layers} layers · {_n_heads} heads")

        layer_idx = st.slider("Layer", 0, max(0, _n_layers - 1), min(_n_layers - 1, 47))

        head_sel_zarr = st.selectbox(
            "Head",
            ["Average"] + list(range(_n_heads)),
            format_func=lambda x: "All heads averaged" if x == "Average" else f"Head {x}",
        )

        top_k = st.slider("Top-K connections", 10, 200, 50, step=10)

        _head_idx = None if head_sel_zarr == "Average" else int(head_sel_zarr)
        connections = zarr_reader.load_attention(array_name, layer_idx, _head_idx, top_k)
        head_label = "All heads averaged" if head_sel_zarr == "Average" else f"Head {head_sel_zarr}"

        pdb_string = zarr_reader.get_pdb_string()
        attn_badge = f"{array_name}"

    # ── Shared display toggles ────────────────────────────────────────────────
    st.divider()
    st.markdown("**Display**")
    show_structure = st.toggle("3D Structure", value=True)
    show_heatmap = st.toggle("Attention Heatmap", value=True)
    show_arc = st.toggle("Arc Diagram", value=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(f"# {protein}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Residues", n_residues)
c2.metric("Layer", layer_idx)
c3.metric("Head", head_label)
c4.metric("Connections", len(connections))

viz_payload = build_visualization_payload(
    fasta_seq=fasta_seq,
    pdb_path=pdb_path,
    connections=connections,
    layer_idx=layer_idx,
    head_label=head_label,
)

with st.expander("Visualization Integration Output", expanded=False):
    st.write("Stored trace data has been converted into visualization-ready format.")
    st.write("Format: `(source_residue, target_residue, attention_weight)`")

    st.write("Layer:", viz_payload["layer_idx"])
    st.write("Head:", viz_payload["head_label"])
    st.write("Residues:", viz_payload["n_residues"])
    st.write("Connections:", len(viz_payload["connections"]))

    if viz_payload["connections"]:
        st.dataframe(
            [
                {
                    "source_residue": r1,
                    "target_residue": r2,
                    "attention_weight": weight,
                }
                for r1, r2, weight in viz_payload["connections"][:10]
            ],
            use_container_width=True,
        )


st.caption(f"Attention: **{attn_badge}** · top-{top_k} per head")

if not connections:
    st.warning("No connections loaded — check that the trace files exist.")

st.divider()

# ── Main layout ───────────────────────────────────────────────────────────────

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    if show_structure:
        with st.container(border=True):
            st.markdown("#### 3D Structure")
            has_structure = (pdb_path and os.path.exists(pdb_path)) or pdb_string
            if has_structure:
                render_structure(
                    pdb_path=pdb_path,
                    connections=connections,
                    n_residues=n_residues,
                    pdb_string=pdb_string,
                )
            else:
                st.info(
                    "No structure file found. For Zarr archives, include a "
                    "`structure_pdb` array with PDB text content."
                )

    if show_arc:
        with st.container(border=True):
            st.markdown("#### Arc Diagram")
            render_arc_diagram(connections, fasta_seq, highlight_residue=residue_idx)

with right_col:
    if show_heatmap:
        with st.container(border=True):
            st.markdown("#### Attention Heatmap")
            render_heatmap(connections, fasta_seq, head_label)

    with st.container(border=True):
        st.markdown("#### Residue Scores")
        if connections and fasta_seq:
            import numpy as np
            import plotly.graph_objects as go

            scores = np.zeros(n_residues)
            for r1, r2, w in connections:
                if r1 < n_residues:
                    scores[r1] += w
                if r2 < n_residues:
                    scores[r2] += w

            tick_step = max(1, n_residues // 20)
            fig = go.Figure(
                go.Bar(
                    x=list(range(n_residues)),
                    y=scores,
                    marker=dict(color=scores, colorscale="Reds", showscale=False),
                    hovertemplate=(
                        "Residue %{x} (%{customdata})<br>"
                        "Score: %{y:.4f}<extra></extra>"
                    ),
                    customdata=list(fasta_seq),
                )
            )
            fig.update_layout(
                xaxis=dict(
                    title="Residue index",
                    tickvals=list(range(0, n_residues, tick_step)),
                    ticktext=[fasta_seq[i] for i in range(0, n_residues, tick_step)],
                ),
                yaxis_title="Aggregated attention",
                margin=dict(l=40, r=10, t=10, b=50),
                height=240,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load attention data to see per-residue scores.")
