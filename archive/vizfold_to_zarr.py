"""
VizFold → Zarr Archive Utilities
=================================

This module contains independent utilities for converting VizFold inference
outputs into a standardized Zarr archive format.  These functions are designed
to support lightweight, offline visualization of model internals such as:

- Per-residue (single) representations per Evoformer layer
- Pairwise representations per Evoformer layer
- Triangle-start attention weights per layer
- Predicted protein structure (atom positions, atom mask, pTM score)
- Run-level metadata (sequence, recycle info, model/config versions)

All methods are intentionally modular so they can be implemented independently
by different contributors.

Archive layout target:

run.vizfold.zarr/
│
├── metadata/
│   ├── model_version        # scalar string
│   ├── config_version       # scalar string
│   ├── sequence             # scalar string
│   ├── num_residues         # scalar int
│   ├── num_recycles         # scalar int
│   ├── recycle_info         # 1-D array, shape (num_recycles,)
│   ├── residue_index        # 1-D array, shape (num_residues,)
│   └── representation_names # 1-D string array
│
├── representations/
│   ├── single/              # group; per-residue representations
│   │   ├── layer_00         # 2-D array, shape (num_residues, single_dim)
│   │   ├── layer_01
│   │   └── ...
│   └── pair/                # group; pairwise representations
│       ├── layer_00         # 3-D array, shape (num_residues, num_residues, pair_dim)
│       ├── layer_01
│       └── ...
│
├── attention/
│   └── triangle_start/      # group; triangle-start attention per layer
│       ├── layer_00         # 4-D array, shape (num_heads, num_residues, num_residues, num_residues)
│       ├── layer_01         #   or 3-D (num_residues, num_residues, num_heads) depending on hook
│       └── ...
│
└── structure/
    ├── atom_positions        # 2-D array, shape (num_atoms, 3)  [num_atoms = num_residues × 37]
    ├── atom_mask             # 1-D array, shape (num_atoms,)
    └── ptm                  # scalar float
"""

import os
import shutil

import numpy as np
import zarr

# ============================================================
# HELPER — layer key formatting
# ============================================================

def _layer_key(layer_index: int) -> str:
    """Return the zero-padded layer key string, e.g. 2 -> 'layer_02'."""
    return f"layer_{layer_index:02d}"


# ============================================================
# METHOD 1
# ============================================================

def tensor_to_numpy(tensor):
    """
    Convert a VizFold tensor into a NumPy array.

    VizFold outputs may come from:
    - PyTorch tensors  (detach → cpu → numpy)
    - NumPy arrays     (returned as-is)

    Parameters
    ----------
    tensor : torch.Tensor | numpy.ndarray

    Returns
    -------
    numpy.ndarray
        A CPU-based NumPy representation of the input.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "detach") and hasattr(tensor, "cpu"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


# ============================================================
# METHOD 2
# ============================================================

def open_archive(archive_path: str, overwrite: bool = False):
    """
    Open (or create) a VizFold Zarr archive and initialise the required
    top-level group structure.

    Creates the following empty groups if they do not already exist:
        metadata/
        representations/single/
        representations/pair/
        attention/triangle_start/
        structure/

    Parameters
    ----------
    archive_path : str
        File-system path for the Zarr DirectoryStore
        (e.g. ``'run.vizfold.zarr'``).

    overwrite : bool
        If True, delete and recreate the store.
        If False, open for appending; raise if the path exists as a
        non-Zarr directory.

    Returns
    -------
    zarr.Group
        The root group of the opened archive, ready for writing.
    """
    if overwrite and os.path.exists(archive_path):
        shutil.rmtree(archive_path)

    root = zarr.open_group(archive_path, mode="a")

    root.require_group("metadata")
    representations = root.require_group("representations")
    representations.require_group("single")
    representations.require_group("pair")
    attention = root.require_group("attention")
    attention.require_group("triangle_start")
    root.require_group("structure")

    return root


# ============================================================
# METHOD 3
# ============================================================

def store_metadata(
    archive_path: str,
    model_version: str,
    config_version: str,
    sequence: str,
    num_residues: int,
    num_recycles: int,
    recycle_info,
    residue_index,
    representation_names,
):
    """
    Write all run-level metadata into the ``metadata/`` group.

    Archive layout written by this function:

        metadata/model_version        <- scalar string
        metadata/config_version       <- scalar string
        metadata/sequence             <- scalar string
        metadata/num_residues         <- scalar int
        metadata/num_recycles         <- scalar int
        metadata/recycle_info         <- 1-D float array, shape (num_recycles,)
        metadata/residue_index        <- 1-D int array,   shape (num_residues,)
        metadata/representation_names <- 1-D object array of strings

    Scalar strings and ints are stored as length-1 Zarr arrays so that the
    archive remains self-describing without external config files.

    Parameters
    ----------
    archive_path : str
        Root path to the Zarr archive.

    model_version : str
        Human-readable model version tag (e.g. ``'openfold-v2.2.0'``).

    config_version : str
        Identifier for the model config used (e.g. ``'model_1_ptm'``).

    sequence : str
        Amino-acid sequence in one-letter code (length == num_residues).

    num_residues : int
        Number of residues in the input sequence.

    num_recycles : int
        Number of recycling iterations performed.

    recycle_info : array-like, shape (num_recycles,)
        Per-recycle diagnostic values (e.g. pLDDT or RMSD between recycles).

    residue_index : array-like, shape (num_residues,)
        Original residue numbering from the input (accommodates gaps).

    representation_names : array-like of str
        Labels for each representation layer stored under
        ``representations/``.

    Returns
    -------
    None
    """
    recycle_info = tensor_to_numpy(recycle_info)
    residue_index = tensor_to_numpy(residue_index)

    if len(sequence) != num_residues:
        raise ValueError(
            f"sequence length ({len(sequence)}) != num_residues ({num_residues})"
        )
    if recycle_info.shape != (num_recycles,):
        raise ValueError(
            f"recycle_info shape {recycle_info.shape} != ({num_recycles},)"
        )
    if residue_index.shape != (num_residues,):
        raise ValueError(
            f"residue_index shape {residue_index.shape} != ({num_residues},)"
        )

    root = zarr.open_group(archive_path, mode="a")
    meta = root.require_group("metadata")

    scalars = {
        "model_version": np.array(model_version),
        "config_version": np.array(config_version),
        "sequence": np.array(sequence),
        "num_residues": np.array(num_residues, dtype=np.int64),
        "num_recycles": np.array(num_recycles, dtype=np.int64),
    }
    for key, value in scalars.items():
        if key in meta:
            del meta[key]
        meta[key] = value

    arrays = {
        "recycle_info": np.asarray(recycle_info),
        "residue_index": np.asarray(residue_index, dtype=np.int64),
        "representation_names": np.asarray(representation_names),
    }
    for key, value in arrays.items():
        if key in meta:
            del meta[key]
        meta[key] = value


# ============================================================
# METHOD 4
# ============================================================

def store_single_representation(
    archive_path: str,
    layer_index: int,
    single_array,
    chunks=None,
    overwrite: bool = True,
):
    """
    Store a per-residue (single) representation for one Evoformer layer.

    Archive location:
        representations/single/layer_<XX>

    Expected array shape:
        (num_residues, single_dim)

    Each layer's output is stored as a separate Zarr array under the
    ``representations/single/`` group.  The layer key is zero-padded to
    two digits (``layer_00``, ``layer_01``, …) so lexicographic and
    numeric ordering agree.

    Parameters
    ----------
    archive_path : str
        Root path to the Zarr archive.

    layer_index : int
        Zero-based Evoformer layer index.

    single_array : numpy.ndarray | torch.Tensor
        Per-residue representation, shape (num_residues, single_dim).

    chunks : tuple, optional
        Zarr chunk shape.  Defaults to one residue per chunk:
        ``(1, single_dim)``.

    overwrite : bool
        Replace existing data for this layer if True.

    Returns
    -------
    None
    """
    single_array = tensor_to_numpy(single_array)

    # Validate shape
    if single_array.ndim != 2:
        raise ValueError(
            f"Expected (num_residues, single_dim), got {single_array.shape}"
        )

    num_residues, single_dim = single_array.shape

    root = zarr.open_group(archive_path, mode="a")
    group = root["representations"]["single"]

    layer_key = _layer_key(layer_index)

    if layer_key in group:
        if not overwrite:
            raise FileExistsError(f"{layer_key} already exists")
        del group[layer_key]

    # Default chunking: per-residue
    if chunks is None:
        chunks = (1, single_dim)

    group.create_dataset(
        layer_key,
        data=single_array,
        chunks=chunks,
    )


# ============================================================
# METHOD 5
# ============================================================

def store_pair_representation(
    archive_path: str,
    layer_index: int,
    pair_array,
    chunks=None,
    overwrite: bool = True,
):
    """
    Store a pairwise representation for one Evoformer layer.

    Archive location:
        representations/pair/layer_<XX>

    Expected array shape:
        (num_residues, num_residues, pair_dim)

    The first two dimensions must be equal (square residue × residue matrix).
    Each layer is stored as a separate array; no cross-layer data is mixed.

    Parameters
    ----------
    archive_path : str
        Root path to the Zarr archive.

    layer_index : int
        Zero-based Evoformer layer index.

    pair_array : numpy.ndarray | torch.Tensor
        Pairwise representation, shape (num_residues, num_residues, pair_dim).

    chunks : tuple, optional
        Zarr chunk shape.  Defaults to one residue-row per chunk:
        ``(1, num_residues, pair_dim)``.

    overwrite : bool
        Replace existing data for this layer if True.

    Returns
    -------
    None
    """
    pair_array = tensor_to_numpy(pair_array)

    # Validate shape
    if pair_array.ndim != 3:
        raise ValueError(
            f"Expected (num_residues, num_residues, pair_dim), got {pair_array.shape}"
        )

    n_i, n_j, pair_dim = pair_array.shape

    if n_i != n_j:
        raise ValueError(
            f"Pair representation must be square, got {pair_array.shape}"
        )

    root = zarr.open_group(archive_path, mode="a")
    group = root["representations"]["pair"]

    layer_key = _layer_key(layer_index)

    if layer_key in group:
        if not overwrite:
            raise FileExistsError(f"{layer_key} already exists")
        del group[layer_key]

    # Default chunking: one row of pair matrix
    if chunks is None:
        chunks = (1, n_i, pair_dim)

    group.create_dataset(
        layer_key,
        data=pair_array,
        chunks=chunks,
    )

# ============================================================
# METHOD 6
# ============================================================

def store_triangle_attention(
    archive_path: str,
    layer_index: int,
    attention_array,
    chunks=None,
    overwrite: bool = True,
):
    """
    Store triangle-start attention weights for one layer.

    Archive location:
        attention/triangle_start/layer_<XX>

    Expected array shape (as captured from the forward hook):
        (num_residues, num_residues, num_heads)

    The triangle attention mechanism in VizFold/OpenFold operates on pair
    representations.  Each element [i, j, h] is the attention weight that
    residue-pair (i, j) received from head h during the triangular update
    starting at node i.

    Recommended chunking:
        (num_residues, num_residues, 1)
    so that a single attention head can be read without loading all heads.

    Parameters
    ----------
    archive_path : str
        Root path to the Zarr archive.

    layer_index : int
        Zero-based layer index for the triangle attention block.

    attention_array : numpy.ndarray | torch.Tensor
        Attention weights, shape (num_residues, num_residues, num_heads).

    chunks : tuple, optional
        Zarr chunk shape.  Defaults to ``(num_residues, num_residues, 1)``.

    overwrite : bool
        Replace existing data for this layer if True.

    Returns
    -------
    None
    """
    

    attention_array = tensor_to_numpy(attention_array)
    
    if attention_array.ndim != 3:
        raise ValueError(f"Expected (num_residues, num_residues, num_heads), got {attention_array.shape}")
    n_i, n_j, num_heads = attention_array.shape
    
    if n_i != n_j:
        raise ValueError(f"Triangle attention must be square in first two dims, got {attention_array.shape}")
    
    path_info = archive_path
    root = zarr.open_group(path_info, mode="a")
    group = root["attention"]["triangle_start"]
    layer_key = _layer_key(layer_index)
    
    if layer_key in group:
        if not overwrite:
            raise FileExistsError("{} already exists".format(layer_key))
        del group[layer_key]
    
    if chunks is None:
        chunks = (n_i, n_j, 1)
    
    group.create_dataset(layer_key, data=attention_array, chunks=chunks)


# ============================================================
# METHOD 7
# ============================================================

def store_structure(
    archive_path: str,
    atom_positions,
    atom_mask,
    ptm: float,
    overwrite: bool = True,
):
    """
    Store predicted structure outputs into the ``structure/`` group.

    Archive layout written by this function:

        structure/atom_positions   <- 2-D float array, shape (num_atoms, 3)
        structure/atom_mask        <- 1-D float array, shape (num_atoms,)
        structure/ptm              <- scalar float

    In OpenFold/VizFold ``num_atoms = num_residues × 37`` because the model
    predicts coordinates for all 37 heavy-atom positions per residue
    (backbone + up to 14 side-chain atoms, padded to 37).
    ``atom_mask`` is 1.0 where an atom is present and 0.0 where padded.

    Parameters
    ----------
    archive_path : str
        Root path to the Zarr archive.

    atom_positions : numpy.ndarray | torch.Tensor
        Predicted 3-D coordinates, shape (num_atoms, 3).
        Typically obtained from ``output['final_atom_positions']``
        after flattening the residue dimension:
        ``positions.reshape(-1, 3)``.

    atom_mask : numpy.ndarray | torch.Tensor
        Binary mask, shape (num_atoms,), indicating valid atom positions.
        Obtained from ``output['final_atom_mask'].reshape(-1)``.

    ptm : float
        Predicted TM-score (pTM) for the run, scalar in [0, 1].

    overwrite : bool
        Replace existing structure data if True.

    Returns
    -------
    None
    """
    
    atom_positions = tensor_to_numpy(atom_positions)
    atom_mask = tensor_to_numpy(atom_mask)
    
    if atom_positions.ndim != 2:
        raise ValueError(f"Expected atom_positions shape (num_atoms, 3), got {atom_positions.shape}")
    
    if atom_positions.shape[1] != 3:
         raise ValueError("Expected atom_positions shape (num_atoms, 3), got {})".format(atom_positions.shape))
    
    num_atoms = atom_positions.shape[0]
    
    if atom_mask.ndim != 1:
        raise ValueError(f"Expected atom_mask shape ({num_atoms},), got {atom_mask.shape}")
    if atom_mask.shape[0] != num_atoms:
        raise ValueError(f"Expected atom_mask shape ({num_atoms},), got {atom_mask.shape}")
    
    root = zarr.open_group(archive_path, mode="a")
    group = root.require_group("structure")
    
    dataArray = ["atom_positions", "atom_mask", "ptm"]
    
    for i in dataArray:
        if i in group and overwrite:
            del group[i]
    
    group.create_dataset("atom_positions", data=atom_positions)
    group.create_dataset("atom_mask", data=atom_mask)
    group["ptm"] = np.array(float(ptm))

# ============================================================
# METHOD 8
# ============================================================

def load_single_representation(archive_path: str, layer_index: int) -> np.ndarray:
    """
    Load the per-residue (single) representation for one Evoformer layer.

    Archive location read:
        representations/single/layer_<XX>

    Parameters
    ----------
    archive_path : str
        Root path to the Zarr archive.

    layer_index : int
        Zero-based Evoformer layer index to retrieve.

    Returns
    -------
    numpy.ndarray
        Per-residue representation, shape (num_residues, single_dim).

    Raises
    ------
    KeyError
        If the requested layer does not exist in the archive.
    """
    root = zarr.open_group(archive_path, mode="r")
    single = root["representations"]["single"]
    layer_key = f"layer_{layer_index:02d}"
    if layer_key not in single:
        raise KeyError(f"Layer not found: representations/single/{layer_key}")
    return np.asarray(single[layer_key])


# ============================================================
# METHOD 9
# ============================================================

def load_pair_representation(archive_path: str, layer_index: int) -> np.ndarray:
    """
    Load the pairwise representation for one Evoformer layer.

    Archive location read:
        representations/pair/layer_<XX>

    Parameters
    ----------
    archive_path : str
        Root path to the Zarr archive.

    layer_index : int
        Zero-based Evoformer layer index to retrieve.

    Returns
    -------
    numpy.ndarray
        Pairwise representation, shape (num_residues, num_residues, pair_dim).

    Raises
    ------
    KeyError
        If the requested layer does not exist in the archive.
    """
    root = zarr.open_group(archive_path, mode="r")
    pair = root["representations"]["pair"]
    layer_key = f"layer_{layer_index:02d}"
    if layer_key not in pair:
        raise KeyError(f"Layer not found: representations/pair/{layer_key}")
    return np.asarray(pair[layer_key])


# ============================================================
# METHOD 10
# ============================================================

def load_triangle_attention(archive_path: str, layer_index: int, head_index: int = None):
    """
    Load triangle-start attention weights for one layer.

    If ``head_index`` is given, return only that head's attention matrix.
    Otherwise return the full tensor for all heads.

    Archive location read:
        attention/triangle_start/layer_<XX>

    Parameters
    ----------
    archive_path : str
        Root path to the Zarr archive.

    layer_index : int
        Zero-based layer index for the triangle attention block.

    head_index : int, optional
        If provided, return only the slice ``[:, :, head_index]``,
        shape (num_residues, num_residues).
        If None, return the full array, shape
        (num_residues, num_residues, num_heads).

    Returns
    -------
    numpy.ndarray
        Shape (num_residues, num_residues) when ``head_index`` is given,
        or (num_residues, num_residues, num_heads) otherwise.

    Raises
    ------
    KeyError
        If the requested layer does not exist in the archive.
    IndexError
        If ``head_index`` is out of range.
    """
    pass


# ============================================================
# METHOD 11
# ============================================================

def validate_archive(archive_path: str) -> bool:
    """
    Validate the integrity of a VizFold Zarr archive.

    Checks that all required top-level groups exist and that every stored
    array has the expected number of dimensions and internally consistent
    shapes.

    Validation rules
    ----------------
    metadata/
        All eight keys must be present.
        ``sequence`` length must equal ``num_residues``.
        ``recycle_info`` length must equal ``num_recycles``.
        ``residue_index`` length must equal ``num_residues``.

    representations/single/
        Each array must be 2-D: (num_residues, single_dim).
        ``num_residues`` must be consistent across all layers.

    representations/pair/
        Each array must be 3-D: (num_residues, num_residues, pair_dim).
        First two dimensions must be equal and match ``num_residues``.

    attention/triangle_start/
        Each array must be 3-D: (num_residues, num_residues, num_heads).
        First two dimensions must be equal and match ``num_residues``.

    structure/
        ``atom_positions`` must be 2-D with last dim == 3.
        ``atom_mask`` must be 1-D with length == atom_positions.shape[0].
        ``ptm`` must be a scalar in [0, 1].

    Parameters
    ----------
    archive_path : str
        Root path to the Zarr archive.

    Returns
    -------
    bool
        True if all checks pass; False (or raises) if any check fails.
    """
    pass


# ============================================================
# METHOD 12  —  top-level orchestrator
# ============================================================

def archive_vizfold_run(
    archive_path: str,
    vizfold_output: dict,
    config,
    sequence: str,
    num_recycles: int,
    recycle_info,
    overwrite: bool = False,
):
    """
    Convert a complete VizFold inference run into a Zarr archive.

    This is the single entry-point that calls all other store_* functions
    in the correct order.  It expects the outputs captured during a VizFold
    forward pass (typically collected via PyTorch forward hooks or returned
    directly by the model) and writes them into ``archive_path``.

    Expected keys inside ``vizfold_output``
    ----------------------------------------
    ``single_representations`` : list of tensors, one per Evoformer layer
        Each tensor has shape (num_residues, single_dim).

    ``pair_representations`` : list of tensors, one per Evoformer layer
        Each tensor has shape (num_residues, num_residues, pair_dim).

    ``triangle_attention_start`` : list of tensors, one per attention layer
        Each tensor has shape (num_residues, num_residues, num_heads).

    ``final_atom_positions`` : tensor, shape (num_residues, 37, 3)
        All-atom predicted coordinates.  Will be reshaped to
        (num_residues * 37, 3) before storing.

    ``final_atom_mask`` : tensor, shape (num_residues, 37)
        Atom presence mask.  Will be reshaped to (num_residues * 37,).

    ``ptm`` : float
        Predicted TM-score for this run.

    Processing steps
    ----------------
    1. ``open_archive``             — initialise the store and group tree
    2. ``store_metadata``           — write all run-level metadata
    3. ``store_single_representation`` (loop) — one call per layer
    4. ``store_pair_representation``   (loop) — one call per layer
    5. ``store_triangle_attention``    (loop) — one call per layer
    6. ``store_structure``          — atom positions, mask, pTM

    Parameters
    ----------
    archive_path : str
        Destination path for the Zarr archive
        (e.g. ``'outputs/run.vizfold.zarr'``).

    vizfold_output : dict
        Dictionary of tensors/arrays collected during inference
        (see expected keys above).

    config : openfold.config.model_config (or equivalent)
        Model configuration object; used to extract ``model_version``
        and ``config_version`` strings for the metadata group.

    sequence : str
        Input amino-acid sequence in one-letter code.

    num_recycles : int
        Number of recycling iterations that were performed.

    recycle_info : array-like, shape (num_recycles,)
        Per-recycle diagnostic scalars logged during inference.

    overwrite : bool
        If True, overwrite any existing archive at ``archive_path``.

    Returns
    -------
    zarr.Group
        The root group of the completed archive.

    Raises
    ------
    KeyError
        If a required key is missing from ``vizfold_output``.
    FileExistsError
        If ``archive_path`` already exists and ``overwrite=False``.
    """
    pass
