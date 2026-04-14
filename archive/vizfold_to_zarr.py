import numpy as np
import zarr

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