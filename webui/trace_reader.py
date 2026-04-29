"""
TraceReader — loads offline VizFold inference traces from disk.

Expected directory layout:

    {trace_root}/
    └── {PROTEIN_ID}/
        ├── *.pdb
        ├── *.fasta
        └── attention/
            ├── msa_row_attn_layer{N}.txt
            └── triangle_start_attn_layer{N}_residue_idx_{R}.txt

Attention files follow the format used by the existing OpenFold viz pipeline:
    Layer {N}, Head {H}
    res1 res2 weight
    ...
"""

import logging
import os
import glob
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


AttentionMap = Dict[int, List[Tuple[int, int, float]]]


class TraceReader:
    def __init__(self, trace_root: str):
        self.root = os.path.expanduser(trace_root)

    # ── Discovery ────────────────────────────────────────────────────────────

    def list_proteins(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )

    def get_pdb_path(self, protein: str) -> Optional[str]:
        hits = glob.glob(os.path.join(self.root, protein, "*.pdb"))
        return hits[0] if hits else None

    def get_fasta_sequence(self, protein: str) -> str:
        hits = glob.glob(os.path.join(self.root, protein, "*.fasta"))
        if not hits:
            return ""
        with open(hits[0]) as f:
            lines = f.readlines()
        return "".join(l.strip() for l in lines if not l.startswith(">"))

    def list_layers(self, protein: str, attention_type: str) -> List[int]:
        attn_dir = os.path.join(self.root, protein, "attention")
        if not os.path.isdir(attn_dir):
            return []
        if attention_type == "msa_row":
            pat = re.compile(r"msa_row_attn_layer(\d+)\.txt$")
        else:
            pat = re.compile(r"triangle_start_attn_layer(\d+)_residue_idx_\d+\.txt$")
        layers: set = set()
        for fname in os.listdir(attn_dir):
            m = pat.match(fname)
            if m:
                layers.add(int(m.group(1)))
        return sorted(layers)

    def list_triangle_residues(self, protein: str, layer_idx: int) -> List[int]:
        attn_dir = os.path.join(self.root, protein, "attention")
        if not os.path.isdir(attn_dir):
            return []
        pat = re.compile(
            rf"triangle_start_attn_layer{layer_idx}_residue_idx_(\d+)\.txt$"
        )
        residues: set = set()
        for fname in os.listdir(attn_dir):
            m = pat.match(fname)
            if m:
                residues.add(int(m.group(1)))
        return sorted(residues)

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_attention(
        self,
        protein: str,
        attention_type: str,
        layer_idx: int,
        top_k: Optional[int] = None,
    ) -> AttentionMap:
        if attention_type != "msa_row":
            return {}
        path = os.path.join(
            self.root, protein, "attention",
            f"msa_row_attn_layer{layer_idx}.txt",
        )
        if not os.path.exists(path):
            return {}
        try:
            return self._parse_heads_file(path, top_k)
        except Exception as exc:
            logger.warning("Failed to load attention from %s: %s", path, exc)
            return {}

    def load_triangle_attention(
        self,
        protein: str,
        layer_idx: int,
        residue_idx: int,
        top_k: Optional[int] = None,
    ) -> AttentionMap:
        path = os.path.join(
            self.root, protein, "attention",
            f"triangle_start_attn_layer{layer_idx}_residue_idx_{residue_idx}.txt",
        )
        if not os.path.exists(path):
            return {}
        try:
            return self._parse_heads_file(path, top_k)
        except Exception as exc:
            logger.warning("Failed to load triangle attention from %s: %s", path, exc)
            return {}

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_heads_file(path: str, top_k: Optional[int]) -> AttentionMap:
        # the file looks like this (this is the format the OpenFold pipeline saves):
        #   Layer <N>, Head <H>
        #   <res1> <res2> <weight>
        #   ...
        # we re-sort even though the file should already be sorted, just to be safe
        heads: AttentionMap = {}
        current: Optional[int] = None
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("layer"):
                    parts = line.replace(",", "").split()
                    current = int(parts[-1])
                    heads[current] = []
                elif current is not None:
                    try:
                        r1, r2, w = map(float, line.split())
                        heads[current].append((int(r1), int(r2), w))
                    except ValueError:
                        logger.warning("Skipping malformed line %d in %s: %r", lineno, path, line)
        for h in heads:
            heads[h].sort(key=lambda x: x[2], reverse=True)
            if top_k is not None:
                heads[h] = heads[h][:top_k]
        return heads


# ── Zarr reader ───────────────────────────────────────────────────────────────

Connections = List[Tuple[int, int, float]]


class ZarrTraceReader:
    """
    Reads attention data from a Zarr ZipStore archive (.zip).

    Auto-detects attention arrays by shape: any array with ndim >= 2
    whose last two dimensions are equal (i.e. N×N residue–residue matrices).

    Supported shapes:
        4D [n_layers, n_heads, N, N]  — standard; layer + head indexable
        3D [n_heads, N, N]            — single-layer store
        2D [N, N]                     — single head + single layer

    Optional metadata arrays (auto-detected by name):
        sequence / seq / fasta        — amino acid string (bytes or str array)
        structure_pdb / pdb           — PDB file content (bytes or str array)
    """

    def __init__(self, file_bytes: bytes) -> None:
        import zarr
        import zarr.storage

        self._tmp_path = tempfile.mktemp(suffix=".zip")
        with open(self._tmp_path, "wb") as f:
            f.write(file_bytes)

        self._store = zarr.storage.ZipStore(self._tmp_path, mode="r")
        self._root = zarr.open_group(self._store, mode="r")
        self._arrays: Dict[str, object] = {}
        self._collect_arrays()

    def _collect_arrays(self) -> None:
        import zarr
        for name, item in self._root.members(max_depth=None):
            if isinstance(item, zarr.Array):
                self._arrays[name] = item

    # ── Discovery ─────────────────────────────────────────────────────────────

    def list_all_arrays(self) -> Dict[str, tuple]:
        return {name: arr.shape for name, arr in self._arrays.items()}  # type: ignore[union-attr]

    def list_attention_arrays(self) -> Dict[str, tuple]:
        """Arrays whose last two dims are equal and ≥ 10 (likely N×N attention)."""
        out = {}
        for name, arr in self._arrays.items():
            shape = arr.shape  # type: ignore[union-attr]
            if len(shape) >= 2 and shape[-1] == shape[-2] and shape[-1] >= 10:
                out[name] = shape
        return out

    def n_layers(self, array_name: str) -> int:
        shape = self._arrays[array_name].shape  # type: ignore[union-attr]
        return shape[0] if len(shape) >= 4 else 1

    def n_heads(self, array_name: str) -> int:
        shape = self._arrays[array_name].shape  # type: ignore[union-attr]
        if len(shape) >= 4:
            return shape[1]
        if len(shape) == 3:
            return shape[0]
        return 1

    def n_residues(self, array_name: str) -> int:
        return self._arrays[array_name].shape[-1]  # type: ignore[union-attr]

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_attention(
        self,
        array_name: str,
        layer_idx: int,
        head_idx: Optional[int],
        top_k: int = 50,
    ) -> Connections:
        arr = self._arrays[array_name]
        shape = arr.shape  # type: ignore[union-attr]

        max_layer = self.n_layers(array_name) - 1
        # zarr doesn't throw an error if you go out of bounds, it just wraps around
        # which gives you wrong data silently - so we check manually here
        if layer_idx > max_layer:
            raise ValueError(
                f"layer_idx {layer_idx} exceeds max layer {max_layer} "
                f"for array '{array_name}' with shape {shape}"
            )

        if len(shape) == 4:
            # [n_layers, n_heads, N, N]
            layer_data = np.array(arr[layer_idx])  # type: ignore[index]
            matrix = layer_data.mean(axis=0) if head_idx is None else layer_data[head_idx]
        elif len(shape) == 3:
            # [n_heads, N, N]
            data = np.array(arr[:])  # type: ignore[index]
            matrix = data.mean(axis=0) if head_idx is None else data[head_idx]
        else:
            # [N, N]
            matrix = np.array(arr[:])  # type: ignore[index]

        return _dense_to_topk_connections(matrix.astype(float), top_k)

    def get_sequence(self) -> str:
        for key in ("sequence", "seq", "fasta", "metadata/sequence"):
            if key in self._arrays:
                raw = np.array(self._arrays[key][()])
                if raw.dtype.kind in ("S", "U", "O"):
                    val = raw.flat[0]
                    return val.decode() if isinstance(val, bytes) else str(val)
        return ""

    def get_pdb_string(self) -> Optional[str]:
        for key in ("structure_pdb", "pdb", "structure", "structure/pdb"):
            if key in self._arrays:
                raw = np.array(self._arrays[key][()])
                if raw.dtype.kind in ("S", "U", "O"):
                    val = raw.flat[0]
                    text = val.decode() if isinstance(val, bytes) else str(val)
                    if "ATOM" in text or "HETATM" in text:
                        return text
        return None

    def __del__(self) -> None:
        try:
            self._store.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            os.unlink(self._tmp_path)
        except Exception:
            pass


# ── Shared helper ─────────────────────────────────────────────────────────────

def _dense_to_topk_connections(matrix: np.ndarray, top_k: int) -> Connections:
    """Convert a dense N×N attention matrix to a sorted top-k connections list."""
    n = matrix.shape[0]
    # only look at the upper triangle so we don't count each pair twice
    # (i->j and j->i would both show up otherwise which would be wrong)
    # average both directions since attention isn't always perfectly symmetric
    rows, cols = np.triu_indices(n, k=1)
    weights = (matrix[rows, cols] + matrix[cols, rows]) / 2.0
    idx = np.argsort(weights)[::-1][:top_k]
    return [(int(rows[i]), int(cols[i]), float(weights[i])) for i in idx]
