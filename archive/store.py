"""
VizFold Archive Utilities - Store

Methods related to writing archive content.
"""

import numpy as np
import zarr

from core import _validate_layer_index, tensor_to_numpy, tensor_to_zarr_array


# ============================================================
# METHOD 3
# ============================================================

def store_single_representation(path, layer_index, single_array, overwrite=False):
    """
    Store per-layer single representation embeddings.

    Single representations capture per-residue (or per-token) embeddings
    from a transformer block. These are distinct from pair representations
    which capture residue-residue relationships.

    This follows the VizFold Inference Trace Archive specification v1.0,
    storing representations at: representations/single/layer_{layer_index:02d}

    Typical shape:
        (num_residues, hidden_dimension)

    Archive layout (VizFold 1.0):
        representations/single/layer_00
        representations/single/layer_01
        ...

    Parameters
    ----------
    path : str
        Root path to the Zarr archive.

    layer_index : int
        Index of the transformer layer (0-indexed).

    single_array : numpy.ndarray
        Single representation tensor with shape (num_residues, hidden_dim).

    overwrite : bool, optional
        Whether to overwrite existing data. Default is False.

    Returns
    -------
    None

    Examples
    --------
    >>> store_single_representation("trace.zarr", 0, layer_0_repr)
    >>> store_single_representation("trace.zarr", 1, layer_1_repr)
    """
    _validate_layer_index(layer_index)
    single_array = tensor_to_numpy(single_array)

    if single_array.ndim != 2:
        raise ValueError(
            f"Expected 2D representation array (num_residues, hidden_dim), "
            f"got {single_array.ndim}D with shape {single_array.shape}"
        )

    layer_name = f"layer_{layer_index:02d}"
    array_path = f"{path.rstrip('/')}::representations/single/{layer_name}"
    tensor_to_zarr_array(array_path, single_array, overwrite=overwrite)


# ============================================================
# METHOD 4
# ============================================================

def store_pair_representation(path, layer_index, pair_array, overwrite=False):
    """
    Store per-layer pair representation embeddings.

    Pair representations capture relationships between residues
    or tokens in the model and are commonly used in protein
    structure prediction models like OpenFold.

    Typical shape:
        (tokens, tokens, pair_dimension)

    Archive layout (VizFold 1.0):
        representations/pair/layer_00
        representations/pair/layer_01
        ...

    Responsibilities:
    -----------------
    - Validate input shape
    - Store pair representation for an explicit layer index

    Parameters
    ----------
    path : str
        Root path to the Zarr archive.

    layer_index : int
        Index of the transformer layer (0-indexed).

    pair_array : numpy.ndarray
        Pair representation tensor.

    overwrite : bool, optional
        Whether to overwrite existing data. Default is False.

    Returns
    -------
    None

    Examples
    --------
    >>> store_pair_representation("trace.zarr", 0, pair_layer_0)
    >>> store_pair_representation("trace.zarr", 1, pair_layer_1)
    >>> store_pair_representation("trace.zarr", 3, pair_layer_3)
    """
    _validate_layer_index(layer_index)
    pair_array = tensor_to_numpy(pair_array)

    if pair_array.ndim != 3:
        raise ValueError(
            f"Expected 3D pair array (tokens, tokens, pair_dim), "
            f"got {pair_array.ndim}D with shape {pair_array.shape}"
        )
    if pair_array.shape[0] != pair_array.shape[1]:
        raise ValueError(
            f"Pair representation must be square in first two dims (tokens x tokens), "
            f"got shape {pair_array.shape}"
        )

    layer_name = f"layer_{layer_index:02d}"
    array_path = f"{path.rstrip('/')}::representations/pair/{layer_name}"
    tensor_to_zarr_array(array_path, pair_array, overwrite=overwrite)


# ============================================================
# METHOD 5
# ============================================================

def store_attention(path, attention_type, layer_index, attention_array, overwrite=False):
    """
    Store attention head maps for a transformer layer by attention type.

    Attention maps describe relationships between tokens and are commonly
    visualized to interpret model behavior. The VizFold spec recognizes
    multiple attention types that serve different purposes:

    - "triangle_start": Triangle attention starting from one edge
    - "triangle_end": Triangle attention ending at one edge
    - "pairwise": Standard pairwise attention between residues

    This follows the VizFold Inference Trace Archive specification v1.0,
    storing attention at: attention/{attention_type}/layer_{layer_index:02d}

    Expected tensor shape:
        (num_heads, tokens, tokens)

    Archive layout (VizFold 1.0):
        attention/triangle_start/layer_00
        attention/triangle_start/layer_01
        attention/triangle_end/layer_00
        attention/pairwise/layer_00
        ...

    Recommended chunking:
        (1, tokens, tokens)

    This chunking allows loading a single attention head without
    loading the entire tensor.

    Parameters
    ----------
    path : str
        Root path to the archive.

    attention_type : str
        Type of attention mechanism. Common values include:
        - "triangle_start": Triangle attention (starting node)
        - "triangle_end": Triangle attention (ending node)
        - "pairwise": Standard pairwise attention

    layer_index : int
        Transformer layer index (0-indexed).

    attention_array : numpy.ndarray
        Attention tensor with shape (num_heads, tokens, tokens).

    overwrite : bool, optional
        Whether to overwrite existing data. Default is False.

    Returns
    -------
    None

    Examples
    --------
    >>> store_attention("trace.zarr", "triangle_start", 0, attn_array)
    >>> store_attention("trace.zarr", "triangle_end", 0, attn_array)
    >>> store_attention("trace.zarr", "pairwise", 0, attn_array)
    """
    attention_array = tensor_to_numpy(attention_array)

    if attention_array.ndim != 3:
        raise ValueError(
            f"Expected 3D attention array (num_heads, tokens, tokens), "
            f"got {attention_array.ndim}D with shape {attention_array.shape}"
        )

    num_heads, tokens_i, tokens_j = attention_array.shape
    if tokens_i != tokens_j:
        raise ValueError(
            f"Attention matrix must be square (tokens x tokens), "
            f"got shape {attention_array.shape}"
        )

    if not attention_type or not isinstance(attention_type, str):
        raise ValueError(
            "attention_type must be a non-empty string (e.g., 'triangle_start', "
            "'triangle_end', 'pairwise')"
        )

    layer_name = f"layer_{layer_index:02d}"
    array_path = f"{path.rstrip('/')}::attention/{attention_type}/{layer_name}"
    chunks = (1, tokens_i, tokens_j)
    tensor_to_zarr_array(array_path, attention_array, chunks=chunks, overwrite=overwrite)


# ============================================================
# METHOD 6
# ============================================================

def store_structure_coordinates(path, atom_positions, atom_mask=None, ptm=None, overwrite=False):
    """
    Store predicted protein structure atom positions and optional confidence fields.

    These positions represent predicted 3D atomic coordinates
    for a protein sequence.

    Expected atom position shape:
        (num_residues, 3) or (num_residues, num_atoms, 3)

    Archive layout (VizFold 1.0):
        structure/atom_positions
        structure/atom_mask (optional)
        structure/ptm (optional)

    Responsibilities:
    -----------------
    - Validate atom position dimensions
    - Store atom position array
    - Optionally store atom mask
    - Optionally store pTM confidence

    Parameters
    ----------
    path : str
        Root path to the Zarr archive.

    atom_positions : numpy.ndarray
        Array of atomic positions with shape
        (num_residues, 3) or (num_residues, num_atoms, 3).

    atom_mask : optional array-like
        Optional atom-presence mask. If provided, first dimension must match
        num_residues.

    ptm : optional float | numpy.ndarray
        Optional predicted TM-score confidence value. Must be scalar.

    overwrite : bool, optional
        Whether to overwrite existing data. Default is False.

    overwrite : bool, optional
        Whether to overwrite existing structure datasets. Default is False.

    Returns
    -------
    None
    """
    atom_positions = tensor_to_numpy(atom_positions)

    # Validate atom position dimensions: (num_residues, 3) or (num_residues, num_atoms, 3)
    if atom_positions.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2D or 3D atom position array, "
            f"got {atom_positions.ndim}D with shape {atom_positions.shape}"
        )
    if atom_positions.shape[-1] != 3:
        raise ValueError(
            f"Expected last dimension to be 3 (x, y, z), "
            f"got {atom_positions.shape[-1]}"
        )

    num_residues = atom_positions.shape[0]
    root = zarr.open(path.rstrip("/"), mode="a")
    structure_group = root.require_group("structure")

    if "atom_positions" in structure_group:
        if not overwrite:
            raise FileExistsError(
                "Dataset already exists at 'structure/atom_positions'. "
                "Set overwrite=True to replace it."
            )
        del structure_group["atom_positions"]
    structure_group["atom_positions"] = zarr.array(atom_positions)

    if atom_mask is not None:
        atom_mask = tensor_to_numpy(atom_mask)
        atom_mask = np.squeeze(atom_mask)
        if atom_mask.shape[0] != num_residues:
            raise ValueError(
                f"atom_mask first dimension ({atom_mask.shape[0]}) must match "
                f"number of residues ({num_residues})"
            )
        if atom_positions.ndim == 3 and atom_mask.ndim > 1:
            num_atoms = atom_positions.shape[1]
            if atom_mask.shape[1] != num_atoms:
                raise ValueError(
                    f"atom_mask second dimension ({atom_mask.shape[1]}) must match "
                    f"num_atoms ({num_atoms}) when atom_positions is 3D"
                )
        if "atom_mask" in structure_group:
            if not overwrite:
                raise FileExistsError(
                    "Dataset already exists at 'structure/atom_mask'. "
                    "Set overwrite=True to replace it."
                )
            del structure_group["atom_mask"]
        structure_group["atom_mask"] = zarr.array(atom_mask)

    if ptm is not None:
        ptm = tensor_to_numpy(ptm)
        ptm = np.asarray(ptm)
        if ptm.size != 1:
            raise ValueError(
                f"ptm must be a scalar value, got shape {ptm.shape}"
            )
        ptm = np.asarray(ptm.item())
        if "ptm" in structure_group:
            if not overwrite:
                raise FileExistsError(
                    "Dataset already exists at 'structure/ptm'. "
                    "Set overwrite=True to replace it."
                )
            del structure_group["ptm"]
        structure_group["ptm"] = zarr.array(ptm)


# ============================================================
# METHOD 9
# ============================================================

def store_metadata(path, model_version, config_version, sequence,
                   num_residues, num_recycles, recycle_info=None,
                   residue_index=None, representation_names=None, overwrite=False):
    """
    Store run-level metadata for a VizFold archive.

    This group records the high-level run context that helps downstream
    tools identify, validate, and reproduce an archive.

    Archive layout (VizFold 1.0):
        metadata/model_version
        metadata/config_version
        metadata/sequence
        metadata/num_residues
        metadata/num_recycles
        metadata/recycle_info
        metadata/residue_index
        metadata/representation_names

    Parameters
    ----------
    path : str
        Root path to the Zarr archive.

    model_version : str
        Model identifier or release version.

    config_version : str
        Configuration identifier or release version.

    sequence : str
        Input sequence for the run.

    num_residues : int
        Number of residues in the sequence.

    num_recycles : int
        Number of recycles used during inference.

    recycle_info : optional array-like
        Additional per-recycle metadata.

    residue_index : optional array-like
        Residue index values for the sequence.

    representation_names : optional array-like
        Ordered names of stored representations.

    overwrite : bool, optional
        Whether to overwrite existing metadata datasets. Default is False.

    Returns
    -------
    None
    """
    archive_path = path.rstrip("/")
    root = zarr.open(archive_path, mode="a")
    root.require_group("metadata")

    if int(num_residues) < 0:
        raise ValueError("num_residues must be non-negative")
    if int(num_recycles) < 0:
        raise ValueError("num_recycles must be non-negative")

    tensor_to_zarr_array(
        f"{archive_path}::metadata/model_version",
        np.asarray(model_version),
        overwrite=overwrite,
    )
    tensor_to_zarr_array(
        f"{archive_path}::metadata/config_version",
        np.asarray(config_version),
        overwrite=overwrite,
    )
    tensor_to_zarr_array(
        f"{archive_path}::metadata/sequence",
        np.asarray(sequence),
        overwrite=overwrite,
    )
    tensor_to_zarr_array(
        f"{archive_path}::metadata/num_residues",
        np.asarray(num_residues, dtype=np.int32),
        overwrite=overwrite,
    )
    tensor_to_zarr_array(
        f"{archive_path}::metadata/num_recycles",
        np.asarray(num_recycles, dtype=np.int32),
        overwrite=overwrite,
    )

    if recycle_info is not None:
        recycle_info_array = tensor_to_numpy(recycle_info)
        tensor_to_zarr_array(
            f"{archive_path}::metadata/recycle_info",
            recycle_info_array,
            overwrite=overwrite,
        )

    if residue_index is not None:
        residue_index_array = tensor_to_numpy(residue_index)
        if residue_index_array.ndim != 1:
            residue_index_array = np.squeeze(residue_index_array)
        if residue_index_array.ndim != 1:
            raise ValueError(
                f"residue_index must be 1D, got shape {residue_index_array.shape}"
            )
        if residue_index_array.shape[0] != int(num_residues):
            raise ValueError(
                f"residue_index length ({residue_index_array.shape[0]}) must match "
                f"num_residues ({int(num_residues)})"
            )
        tensor_to_zarr_array(
            f"{archive_path}::metadata/residue_index",
            residue_index_array,
            overwrite=overwrite,
        )

    if representation_names is not None:
        representation_names_array = tensor_to_numpy(representation_names)
        tensor_to_zarr_array(
            f"{archive_path}::metadata/representation_names",
            representation_names_array,
            overwrite=overwrite,
        )