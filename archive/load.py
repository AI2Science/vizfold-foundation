"""
VizFold Archive Utilities - Load

Methods related to loading, parsing, and orchestration.
"""

import pickle
import re

import numpy as np
import zarr

from core import _validate_layer_index, tensor_to_numpy, validate_archive
from store import (
    store_attention,
    store_metadata,
    store_pair_representation,
    store_single_representation,
    store_structure_coordinates,
)


# ============================================================
# METHOD 7
# ============================================================

def load_attention_head(path, attention_type, layer_index, head_index):
    """
    Load a single attention head from the archive.

    Selective loading is important for visualization because
    attention tensors can be very large.

    This function retrieves only the requested head for a specific
    attention type and layer.

    Archive location (VizFold 1.0):
        attention/{attention_type}/layer_{layer_index:02d}

    Expected output shape:
        (tokens, tokens)

    Parameters
    ----------
    path : str
        Root path to the archive.

    attention_type : str
        Type of attention mechanism (e.g., "triangle_start", "triangle_end", "pairwise").

    layer_index : int
        Transformer layer index (0-indexed).

    head_index : int
        Index of the attention head.

    Returns
    -------
    numpy.ndarray
        Attention matrix for the specified head.

    Examples
    --------
    >>> attn = load_attention_head("trace.zarr", "triangle_start", 0, 2)
    >>> attn.shape
    (128, 128)
    """
    root = zarr.open(path, mode='r')

    if "attention" not in root:
        raise KeyError(
            f"No 'attention' group found in archive at '{path}'."
        )

    attention_group = root["attention"]

    if attention_type not in attention_group:
        raise KeyError(
            f"Attention type '{attention_type}' not found in archive at '{path}'. "
            f"Available types: {list(attention_group.keys())}"
        )

    type_group = attention_group[attention_type]
    layer_name = f"layer_{layer_index:02d}"

    if layer_name not in type_group:
        raise KeyError(
            f"Layer {layer_index} ('{layer_name}') not found for attention type "
            f"'{attention_type}' in archive at '{path}'. "
            f"Available layers: {list(type_group.keys())}"
        )

    attention = type_group[layer_name]

    if attention.ndim != 3:
        raise ValueError(
            f"Expected stored attention to be 3D (num_heads, tokens, tokens), "
            f"got {attention.ndim}D with shape {attention.shape}"
        )

    num_heads = attention.shape[0]

    if head_index < 0 or head_index >= num_heads:
        raise IndexError(
            f"head_index {head_index} is out of range for layer {layer_index} "
            f"which has {num_heads} attention head(s)."
        )

    return np.asarray(attention[head_index])


def ingest_attention_txt(archive_path, txt_file, layer_index, num_tokens,
                         attention_type="pairwise", overwrite=False):
    """
    Parse a VizFold attention text file and store it under attention/{attention_type}/layer_{layer_index:02d}.

    Expected input format:
        Layer <idx>, Head <idx>
        <res_i> <res_j> <score>
        ...

    Parameters
    ----------
    archive_path : str
        Root path to the Zarr archive.

    txt_file : str
        Path to the attention text file.

    layer_index : int
        Transformer layer index.

    num_tokens : int
        Number of tokens/residues in the sequence.

    attention_type : str, optional
        Type of attention (e.g., "triangle_start", "triangle_end", "pairwise").
        Default is "pairwise".

    overwrite : bool, optional
        Whether to overwrite existing data. Default is False.

    Returns
    -------
    dict
        Summary of ingestion including layer_index, num_heads, num_tokens, and source_file.
    """
    if num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")

    header_pattern = re.compile(r"^Layer\s+(\d+),\s+Head\s+(\d+)$", re.IGNORECASE)
    heads = {}
    current_head = None

    with open(txt_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            header_match = header_pattern.match(line)
            if header_match:
                file_layer_idx = int(header_match.group(1))
                if file_layer_idx != layer_index:
                    raise ValueError(
                        f"Layer mismatch: function arg layer_index={layer_index}, "
                        f"file header layer={file_layer_idx}"
                    )
                current_head = int(header_match.group(2))
                heads.setdefault(current_head, [])
                continue

            if current_head is None:
                raise ValueError("Found attention row before any layer/head header")

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Malformed attention row: '{line}'")

            res_i = int(float(parts[0]))
            res_j = int(float(parts[1]))
            score = float(parts[2])

            if not (0 <= res_i < num_tokens and 0 <= res_j < num_tokens):
                raise ValueError(
                    f"Residue index out of bounds for num_tokens={num_tokens}: "
                    f"({res_i}, {res_j})"
                )

            heads[current_head].append((res_i, res_j, score))

    if not heads:
        raise ValueError(f"No attention data found in '{txt_file}'")

    num_heads = max(heads.keys()) + 1
    attention = np.zeros((num_heads, num_tokens, num_tokens), dtype=np.float32)

    for head_idx, entries in heads.items():
        for res_i, res_j, score in entries:
            attention[head_idx, res_i, res_j] = score

    store_attention(archive_path, attention_type, layer_index, attention, overwrite=overwrite)
    return {
        "layer_index": layer_index,
        "attention_type": attention_type,
        "num_heads": num_heads,
        "num_tokens": num_tokens,
        "source_file": txt_file,
    }


def _extract_best_matching_array(container, key_token_patterns):
    """
    Find the best array-like match in a nested container using tokenized key patterns.

    Matching is stricter than raw substring search:
    - Key paths are tokenized on non-alphanumeric boundaries.
    - A pattern matches only if all pattern tokens are present in the path tokens.
    - The best match prefers more specific patterns and shorter paths.

    Parameters
    ----------
    container : dict | list | tuple
        Nested object to search.

    key_token_patterns : list[list[str]]
        Ordered token patterns to match (e.g., [["final", "atom", "positions"]]).

    Returns
    -------
    dict | None
        {
            "array": numpy.ndarray,
            "matched_key": str,
            "pattern": list[str],
            "shape": tuple
        }
        Returns None if no matching array is found.
    """
    normalized_patterns = [
        [token.lower() for token in pattern if token]
        for pattern in key_token_patterns
        if pattern
    ]

    best = None

    def _tokenize(path):
        return [token for token in re.split(r"[^a-z0-9]+", path.lower()) if token]

    def _consider(path, value):
        nonlocal best

        try:
            array = tensor_to_numpy(value)
        except Exception:
            return

        if not isinstance(array, np.ndarray):
            return

        path_tokens = _tokenize(path)
        if not path_tokens:
            return

        path_token_set = set(path_tokens)

        for pattern in normalized_patterns:
            if all(token in path_token_set for token in pattern):
                # Higher score means a more specific and tighter key match.
                score = len(pattern) * 100 - len(path_tokens)
                if best is None or score > best["score"]:
                    best = {
                        "array": array,
                        "matched_key": path,
                        "pattern": pattern,
                        "shape": tuple(array.shape),
                        "score": score,
                    }

    def _walk(obj, path_prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{path_prefix}/{key}" if path_prefix else str(key)
                _consider(path, value)
                _walk(value, path)
        elif isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                path = f"{path_prefix}/{idx}" if path_prefix else str(idx)
                _consider(path, value)
                _walk(value, path)

    _walk(container)

    if best is None:
        return None

    best.pop("score", None)
    return best


def ingest_output_pkl(archive_path, pkl_file, overwrite=False):
    """
    Load a VizFold/OpenFold output .pkl and route known arrays into the archive.

    Current routing behavior:
    - final_atom_positions (N, 37, 3) -> structure/atom_positions using CA atom (index 1)
    - final_atom_mask (if found) -> structure/atom_mask using CA atom (index 1)
    - ptm (if found) -> structure/ptm
    - pair representation (if found) -> representations/pair/layer_00
    """
    with open(pkl_file, "rb") as f:
        output_dict = pickle.load(f)

    if not isinstance(output_dict, dict):
        raise ValueError("Expected pickle file to contain a dictionary output")

    summary = {
        "source_file": pkl_file,
        "stored": [],
        "skipped": [],
        "key_matches": {},
    }

    final_positions_match = _extract_best_matching_array(
        output_dict,
        [["final", "atom", "positions"], ["final", "positions"]],
    )
    final_atom_mask_match = _extract_best_matching_array(
        output_dict,
        [["final", "atom", "mask"], ["atom", "mask"]],
    )
    ptm_match = _extract_best_matching_array(output_dict, [["ptm"], ["predicted", "tm"]])
    pair_match = _extract_best_matching_array(
        output_dict,
        [["pair", "representation"], ["pair", "activations"], ["pair"]],
    )

    summary["key_matches"]["final_positions"] = {
        "pattern": ["final", "atom", "positions"],
        "matched_key": None if final_positions_match is None else final_positions_match["matched_key"],
        "shape": None if final_positions_match is None else final_positions_match["shape"],
    }
    summary["key_matches"]["final_atom_mask"] = {
        "pattern": ["final", "atom", "mask"],
        "matched_key": None if final_atom_mask_match is None else final_atom_mask_match["matched_key"],
        "shape": None if final_atom_mask_match is None else final_atom_mask_match["shape"],
    }
    summary["key_matches"]["ptm"] = {
        "pattern": ["ptm"],
        "matched_key": None if ptm_match is None else ptm_match["matched_key"],
        "shape": None if ptm_match is None else ptm_match["shape"],
    }
    summary["key_matches"]["pair_representation"] = {
        "pattern": ["pair"],
        "matched_key": None if pair_match is None else pair_match["matched_key"],
        "shape": None if pair_match is None else pair_match["shape"],
    }

    final_positions = None if final_positions_match is None else final_positions_match["array"]
    final_atom_mask = None if final_atom_mask_match is None else final_atom_mask_match["array"]
    ptm = None if ptm_match is None else ptm_match["array"]

    if final_positions is not None:
        if final_positions.ndim == 3 and final_positions.shape[-1] == 3:
            atom_positions = (
                final_positions[:, 1, :]
                if final_positions.shape[1] > 1
                else final_positions[:, 0, :]
            )
            atom_mask = None
            if final_atom_mask is not None:
                final_atom_mask = tensor_to_numpy(final_atom_mask)
                if final_atom_mask.ndim == 2:
                    atom_mask = final_atom_mask[:, 1] if final_atom_mask.shape[1] > 1 else final_atom_mask[:, 0]
                elif final_atom_mask.ndim == 1:
                    atom_mask = final_atom_mask

            store_structure_coordinates(
                archive_path,
                atom_positions,
                atom_mask=atom_mask,
                ptm=ptm,
                overwrite=overwrite,
            )
            summary["stored"].append("structure/atom_positions")
            if atom_mask is not None:
                summary["stored"].append("structure/atom_mask")
            if ptm is not None:
                summary["stored"].append("structure/ptm")
        else:
            summary["skipped"].append("final_atom_positions (unexpected shape)")
    else:
        summary["skipped"].append("final_atom_positions (not found)")

    pair_array = None if pair_match is None else pair_match["array"]
    if pair_array is not None and isinstance(pair_array, np.ndarray):
        if pair_array.ndim == 3 and pair_array.shape[0] == pair_array.shape[1]:
            store_pair_representation(archive_path, 0, pair_array, overwrite=overwrite)
            summary["stored"].append("representations/pair/layer_00")
        else:
            summary["skipped"].append("representations/pair/layer_00 (unexpected shape)")
    else:
        summary["skipped"].append("representations/pair/layer_00 (not found)")

    return summary


# ============================================================
# METHOD 10
# ============================================================

def _load_dataset_as_python_value(dataset):
    """Load a Zarr dataset and normalize 0-D values to Python scalars."""
    value = np.asarray(dataset)
    if value.shape == ():
        return value.item()
    return value


def load_metadata(path):
    """
    Load metadata from a VizFold archive.

    This is the read-side counterpart to store_metadata(). It returns the
    metadata group contents as a plain dictionary so callers can inspect the
    run context before loading heavier arrays.

    Parameters
    ----------
    path : str
        Root path to the Zarr archive.

    Returns
    -------
    dict
        Dictionary containing required and optional metadata fields.
    """
    root = zarr.open(path, mode="r")

    if "metadata" not in root:
        raise KeyError(f"No 'metadata' group found in archive at '{path}'.")

    metadata_group = root["metadata"]
    required_fields = (
        "model_version",
        "config_version",
        "sequence",
        "num_residues",
        "num_recycles",
    )
    missing = [field for field in required_fields if field not in metadata_group]
    if missing:
        raise KeyError(
            f"Missing required metadata field(s) in archive at '{path}': {missing}"
        )

    metadata = {}
    for field in required_fields:
        metadata[field] = _load_dataset_as_python_value(metadata_group[field])

    for field in ("recycle_info", "residue_index", "representation_names"):
        if field in metadata_group:
            metadata[field] = _load_dataset_as_python_value(metadata_group[field])

    return metadata


# ============================================================
# METHOD 11
# ============================================================

def load_single_representation(path, layer_index):
    """
    Load a per-layer single representation from the archive.

    Parameters
    ----------
    path : str
        Root path to the Zarr archive.

    layer_index : int
        Index of the transformer layer (0-indexed).

    Returns
    -------
    numpy.ndarray
        The stored single representation array.
    """
    _validate_layer_index(layer_index)
    root = zarr.open(path, mode="r")

    layer_name = f"layer_{layer_index:02d}"
    dataset_path = f"representations/single/{layer_name}"

    if "representations" not in root or "single" not in root["representations"]:
        raise KeyError(
            f"No 'representations/single' group found in archive at '{path}'."
        )

    single_group = root["representations/single"]
    if layer_name not in single_group:
        raise KeyError(
            f"Layer '{layer_name}' not found at '{dataset_path}' in archive at '{path}'."
        )

    return np.asarray(single_group[layer_name])


# ============================================================
# METHOD 12
# ============================================================

def load_pair_representation(path, layer_index):
    """
    Load a per-layer pair representation from the archive.

    Parameters
    ----------
    path : str
        Root path to the Zarr archive.

    layer_index : int
        Index of the transformer layer (0-indexed).

    Returns
    -------
    numpy.ndarray
        The stored pair representation array.
    """
    _validate_layer_index(layer_index)
    root = zarr.open(path, mode="r")

    layer_name = f"layer_{layer_index:02d}"
    dataset_path = f"representations/pair/{layer_name}"

    if "representations" not in root or "pair" not in root["representations"]:
        raise KeyError(
            f"No 'representations/pair' group found in archive at '{path}'."
        )

    pair_group = root["representations/pair"]
    if layer_name not in pair_group:
        raise KeyError(
            f"Layer '{layer_name}' not found at '{dataset_path}' in archive at '{path}'."
        )

    return np.asarray(pair_group[layer_name])


# ============================================================
# METHOD 13
# ============================================================

class ArchiveOrchestrator:
    """
    Thin helper that sequences archive writes and records what happened.

    The class intentionally stays lightweight: it does not replace the core
    store_* functions, it only coordinates them and captures a run log.
    """

    def __init__(self, archive_path):
        self.archive_path = archive_path.rstrip("/")
        self.events = []

    def _record(self, action, target, **details):
        event = {"action": action, "target": target}
        if details:
            event.update(details)
        self.events.append(event)
        return event

    def add_metadata(self, *args, **kwargs):
        store_metadata(self.archive_path, *args, **kwargs)
        return self._record("store", "metadata")

    def add_single_layer(self, layer_index, single_array, overwrite=False):
        store_single_representation(
            self.archive_path,
            layer_index,
            single_array,
            overwrite=overwrite,
        )
        return self._record("store", f"representations/single/layer_{layer_index:02d}")

    def add_pair_layer(self, layer_index, pair_array, overwrite=False):
        store_pair_representation(
            self.archive_path,
            layer_index,
            pair_array,
            overwrite=overwrite,
        )
        return self._record("store", f"representations/pair/layer_{layer_index:02d}")

    def add_attention(self, attention_type, layer_index, attention_array, overwrite=False):
        store_attention(
            self.archive_path,
            attention_type,
            layer_index,
            attention_array,
            overwrite=overwrite,
        )
        return self._record(
            "store",
            f"attention/{attention_type}/layer_{layer_index:02d}",
        )

    def add_structure(self, atom_positions, atom_mask=None, ptm=None, overwrite=False):
        store_structure_coordinates(
            self.archive_path,
            atom_positions,
            atom_mask=atom_mask,
            ptm=ptm,
            overwrite=overwrite,
        )
        return self._record("store", "structure")

    def validate(self, validator=validate_archive, *args, **kwargs):
        result = validator(self.archive_path, *args, **kwargs)
        self._record("validate", "archive", result=result)
        return result

    def summary(self):
        return {
            "archive_path": self.archive_path,
            "events": list(self.events),
        }