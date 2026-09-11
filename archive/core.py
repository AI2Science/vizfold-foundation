"""
VizFold Archive Utilities - Core

Shared utility methods and archive validation.
"""

import os

import numpy as np
import zarr


# ============================================================
# METHOD 1
# ============================================================

def tensor_to_numpy(tensor):
    """
    Convert a VizFold tensor into a NumPy array.

    VizFold outputs may come from different frameworks such as:
    - PyTorch tensors
    - NumPy arrays

    This function standardizes these formats so they can be stored
    in a Zarr archive.

    Expected behavior:
    ------------------
    If input is a PyTorch tensor:
        - Detach from computation graph
        - Move to CPU if needed
        - Convert to numpy array

    If input is already a NumPy array:
        - Return it unchanged

    Parameters
    ----------
    tensor : torch.Tensor | numpy.ndarray

    Returns
    -------
    numpy.ndarray
        A CPU-based NumPy representation of the tensor.
    """
    if isinstance(tensor, np.ndarray):
        return tensor

    # Support PyTorch tensors without importing torch as a hard dependency.
    if hasattr(tensor, "detach") and hasattr(tensor, "cpu"):
        return tensor.detach().cpu().numpy()

    return np.asarray(tensor)


# ============================================================
# METHOD 2
# ============================================================

def tensor_to_zarr_array(path, tensor, chunks=None, overwrite=False):
    """
    Convert a tensor into a Zarr array stored at the specified path.

    This function creates a Zarr dataset on disk from a tensor or
    NumPy array.

    Typical usage:
    - Writing activations
    - Writing attention maps
    - Writing pair representations

    Responsibilities:
    -----------------
    - Convert tensor to NumPy if necessary
    - Create a Zarr dataset
    - Apply chunking if specified
    - Optionally overwrite existing data

    Parameters
    ----------
    path : str
        Path inside the archive where the array should be stored.

    tensor : numpy.ndarray | torch.Tensor
        The tensor data to write.

    chunks : tuple, optional
        Chunk size for Zarr storage.

    overwrite : bool
        Whether existing data should be replaced.

    Returns
    -------
    None
    """
    import os
    array = tensor_to_numpy(tensor)
    array = np.asarray(array)

    if "::" in path:
        archive_path, dataset_path = path.split("::", 1)
        if not dataset_path:
            raise ValueError("Dataset path is empty. Use 'archive.zarr::group/dataset'.")

        root = zarr.open(archive_path, mode="a")
        if "/" in dataset_path:
            parent_path, dataset_name = dataset_path.rsplit("/", 1)
            parent_group = root.require_group(parent_path)
        else:
            parent_group = root
            dataset_name = dataset_path

        if dataset_name in parent_group and not overwrite:
            raise FileExistsError(
                f"Dataset already exists at '{dataset_path}'. "
                "Set overwrite=True to replace it."
            )

        if dataset_name in parent_group:
            del parent_group[dataset_name]

        create_kwargs = {
            "data": array,
            "shape": array.shape,
            "dtype": array.dtype,
        }
        if chunks is not None:
            create_kwargs["chunks"] = chunks
        parent_group.create_dataset(dataset_name, **create_kwargs)
        return parent_group[dataset_name]

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"Zarr array already exists at '{path}'. Set overwrite=True to replace it."
        )

    mode = "w" if overwrite else "w-"
    z = zarr.open_array(
        store=path,
        mode=mode,
        shape=array.shape,
        dtype=array.dtype,
        chunks=chunks,
    )
    z[...] = array
    return z


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _validate_layer_index(layer_index):
    """
    Validate a transformer layer index used for per-layer archive paths.
    """
    if not isinstance(layer_index, (int, np.integer)):
        raise TypeError(
            f"layer_index must be an integer, got {type(layer_index).__name__}"
        )
    if layer_index < 0:
        raise ValueError(f"layer_index must be >= 0, got {layer_index}")


# ============================================================
# METHOD 8
# ============================================================

def validate_archive(path, strict=True):
    """
    Validate the integrity of a VizFold trace archive.

    This function checks whether the expected archive structure
    exists and whether key datasets appear to be valid.

    Validation checks may include:
    ------------------------------
    - Presence of required groups:
        layers/
        representations/
        structure/

    - Valid shapes for:
        activations
        attention maps
        pair representations
        structure atom positions

    - Metadata consistency if present.

    The goal is to ensure the archive can be safely used for
    offline visualization and analysis.

    Parameters
    ----------
    path : str
        Root path to the Zarr archive.

    strict : bool, optional
        Whether to enforce complete-archive requirements. Default is True.

    Returns
    -------
    dict
        {'valid', 'strict_mode', 'path', 'errors', 'warnings', 'components_found'}
    """
    report = {
        "valid": True,
        "strict_mode": strict,
        "path": path,
        "errors": [],
        "warnings": [],
        "components_found": {
            "metadata": False,
            "representations/single": False,
            "representations/pair": False,
            "attention": False,
            "structure/atom_positions": False,
        },
    }

    def _fail(message):
        report["valid"] = False
        report["errors"].append(message)
        if strict:
            raise ValueError(message)

    def _warn(message):
        report["warnings"].append(message)

    if not os.path.exists(path):
        msg = f"Archive path does not exist: '{path}'"
        report["valid"] = False
        report["errors"].append(msg)
        if strict:
            raise FileNotFoundError(msg)
        return report

    try:
        root = zarr.open(path, mode="r")
    except Exception as e:
        _fail(f"Failed to open archive as Zarr: {e}")
        return report

    if "structure" not in root or "atom_positions" not in root.get("structure", {}):
        _fail("Missing required dataset: 'structure/atom_positions'")
    else:
        atom_positions = root["structure"]["atom_positions"]
        if atom_positions.ndim not in (2, 3) or atom_positions.shape[-1] != 3:
            _fail(f"'structure/atom_positions' invalid shape: {atom_positions.shape}")
        else:
            report["components_found"]["structure/atom_positions"] = True

    if "metadata" not in root:
        _fail("Missing group: 'metadata'") if strict else _warn("Missing group: 'metadata'")
    else:
        report["components_found"]["metadata"] = True

    reprs = root.get("representations", {})
    if "single" not in reprs:
        _fail("Missing group: 'representations/single'") if strict else _warn("Missing group: 'representations/single'")
    elif not any(k.startswith("layer_") for k in reprs["single"].keys()):
        _fail("'representations/single' has no layer_XX datasets") if strict else _warn("'representations/single' has no layer_XX datasets")
    else:
        report["components_found"]["representations/single"] = True

    if "pair" not in reprs:
        _fail("Missing group: 'representations/pair'") if strict else _warn("Missing group: 'representations/pair'")
    elif not any(k.startswith("layer_") for k in reprs["pair"].keys()):
        _fail("'representations/pair' has no layer_XX datasets") if strict else _warn("'representations/pair' has no layer_XX datasets")
    else:
        report["components_found"]["representations/pair"] = True

    if "attention" not in root or not list(root["attention"].keys()):
        _fail("Missing group: 'attention'") if strict else _warn("Missing group: 'attention'")
    else:
        report["components_found"]["attention"] = True

    return report
