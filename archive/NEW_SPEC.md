# VizFold Archive Utilities - Complete API Specification

**Version:** 2.0 (Post-Refactor)  
**Date:** April 26, 2026  
**Status:** Production Ready  
**Archive Format:** VizFold 1.0 (Zarr-based hierarchical storage)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Complete Method Reference](#complete-method-reference)
4. [User Workflows](#user-workflows)
5. [Archive Structure](#archive-structure)
6. [Validation & Safety](#validation--safety)
7. [Pickle Ingestion & Traceability](#pickle-ingestion--traceability)
8. [API Patterns & Guarantees](#api-patterns--guarantees)

---

## Overview

The **VizFold Archive Utilities** provide a complete API for building, reading, and managing Zarr-based inference trace archives from protein structure prediction models (OpenFold, VizFold, etc.).

### Key Features

- **Incremental Writing**: Build archives layer-by-layer, optionally, safely
- **Read-Optimized Loading**: Selective head loading, metadata retrieval without large tensors
- **Multiple Input Formats**: PyTorch tensors, NumPy arrays, text files, pickle outputs
- **Strict & Lenient Validation**: Support both complete archive validation and incremental workflow validation
- **Traceability**: Key matching records show exactly which pickle keys were matched during ingestion
- **Consistent Overwrite Safety**: All write methods use `overwrite=False` default for protection

### Use Cases

1. **Incremental Trace Building**: Write metadata → per-layer representations → attention → structure
2. **Pickle Ingestion**: Auto-extract and route arrays from OpenFold output `.pkl` files
3. **Visualization Server**: Load specific heads/layers on-demand without full archive in memory
4. **Archive Validation**: Check integrity at any point in workflow (strict for complete, lenient for partial)
5. **Model Analysis**: Extract and organize inference traces for debugging, interpretation, ablation

---

## Architecture

### Three-Module Design

The codebase is split into three focused modules for separation of concerns:

#### **core.py** — Shared Utilities & Validation (156 lines)

Contains low-level tensor conversion, Zarr writing, layer indexing, and archive validation.

- `tensor_to_numpy(tensor)` — Convert PyTorch/NumPy to NumPy
- `tensor_to_zarr_array(path, tensor, chunks, overwrite)` — Write arrays to Zarr with safety
- `_validate_layer_index(layer_index)` — Validate layer indices
- `validate_archive(path, strict=True)` — Comprehensive archive integrity check

**Responsibility**: Foundational I/O and validation shared by both store.py and load.py

---

#### **store.py** — Writing Methods (391 lines)

Contains all methods for writing data into archives.

- `store_single_representation(path, layer_index, single_array, overwrite=False)` — Per-layer residue embeddings
- `store_pair_representation(path, layer_index, pair_array, overwrite=False)` — Per-layer token-token relationships
- `store_attention(path, attention_type, layer_index, attention_array, overwrite=False)` — Attention heads by type
- `store_structure_coordinates(path, atom_positions, atom_mask=None, ptm=None, overwrite=False)` — 3D structure
- `store_metadata(path, model_version, config_version, sequence, num_residues, num_recycles, ...)` — Run context

**Responsibility**: All data ingestion and storage operations with validation and overwrite control

---

#### **load.py** — Reading & Orchestration (768 lines)

Contains all methods for reading data, parsing external formats, and orchestrating writes.

- `load_attention_head(path, attention_type, layer_index, head_index)` — Single head loading
- `ingest_attention_txt(archive_path, txt_file, layer_index, num_tokens, ...)` — Parse text attention format
- `_extract_best_matching_array(container, key_token_patterns)` — Tokenized key matching helper
- `ingest_output_pkl(archive_path, pkl_file, overwrite=False)` — Extract & route arrays from pickle output
- `_load_dataset_as_python_value(dataset)` — Normalize 0-D Zarr values to Python scalars
- `load_metadata(path)` — Read run context as dictionary
- `load_single_representation(path, layer_index)` — Per-layer residue embeddings
- `load_pair_representation(path, layer_index)` — Per-layer token-token relationships
- `ArchiveOrchestrator` — Thin coordination class for sequencing writes with event logging

**Responsibility**: All data retrieval, external format parsing, and write sequencing

---

### Import Graph

```
store.py
  ├── from core import: _validate_layer_index, tensor_to_numpy, tensor_to_zarr_array

load.py
  ├── from core import: _validate_layer_index, tensor_to_numpy, validate_archive
  └── from store import: store_attention, store_metadata, store_pair_representation,
                         store_single_representation, store_structure_coordinates
```

---

## Complete Method Reference

### **core.py Methods**

#### METHOD 1: `tensor_to_numpy(tensor)`

Convert any tensor to NumPy for standardized storage.

```python
def tensor_to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray
```

**Parameters:**
- `tensor` (torch.Tensor | numpy.ndarray): Input from any framework

**Returns:**
- `numpy.ndarray`: CPU-resident NumPy array

**Behavior:**
- PyTorch tensor → detach, move to CPU, convert to NumPy
- NumPy array → return unchanged
- Other → `np.asarray()` fallback

**Errors:** None (always succeeds)

**Example:**
```python
import torch
import numpy as np
from core import tensor_to_numpy

pt_tensor = torch.randn(10, 512)
np_array = tensor_to_numpy(pt_tensor)  # → ndarray (10, 512)

np_input = np.ones((5, 3))
result = tensor_to_numpy(np_input)  # → same ndarray
```

---

#### METHOD 2: `tensor_to_zarr_array(path, tensor, chunks=None, overwrite=False)`

Write tensors directly to Zarr with nested path support and overwrite protection.

```python
def tensor_to_zarr_array(
    path: str,
    tensor: torch.Tensor | np.ndarray,
    chunks: tuple | None = None,
    overwrite: bool = False
) -> zarr.Array
```

**Parameters:**
- `path` (str): Zarr location. Supports two formats:
  - `"archive.zarr::group/dataset"` for nested paths (archive mode)
  - `"file.zarr"` for direct Zarr array (file mode)
- `tensor`: Data to store
- `chunks` (tuple, optional): Zarr chunk dimensions (e.g., `(1, tokens, tokens)`)
- `overwrite` (bool): Whether to replace existing data

**Returns:**
- `zarr.Array`: Zarr array reference

**Errors:**
- `ValueError`: Empty dataset path in archive mode
- `FileExistsError`: Data exists and `overwrite=False`

**Example:**
```python
from core import tensor_to_zarr_array
import numpy as np

# Archive mode: nested path
attention = np.random.randn(8, 128, 128)
tensor_to_zarr_array(
    "trace.zarr::attention/triangle_start/layer_00",
    attention,
    chunks=(1, 128, 128),
    overwrite=False
)

# File mode: direct Zarr array
representation = np.random.randn(128, 768)
tensor_to_zarr_array("reps.zarr", representation)
```

---

#### METHOD 3: `_validate_layer_index(layer_index)`

Validate transformer layer indexing (internal).

```python
def _validate_layer_index(layer_index: int) -> None
```

**Parameters:**
- `layer_index` (int): Layer index to validate

**Errors:**
- `TypeError`: Not an integer
- `ValueError`: Negative index

---

#### METHOD 4: `validate_archive(path, strict=True)`

Comprehensive integrity check with strict/lenient modes.

```python
def validate_archive(path: str, strict: bool = True) -> dict
```

**Parameters:**
- `path` (str): Archive root directory
- `strict` (bool): 
  - `True` (default): Raises exceptions, requires complete archive
  - `False`: Returns warnings, allows partial/incremental archives

**Returns:**
```python
{
    "valid": bool,                    # Overall status
    "strict_mode": bool,              # Mode used
    "path": str,                      # Checked path
    "errors": list[str],              # Critical issues
    "warnings": list[str],            # Soft issues (lenient mode)
    "components_found": {
        "metadata": bool,
        "representations/single": bool,
        "representations/pair": bool,
        "attention": bool,
        "structure/atom_positions": bool
    }
}
```

**Strict Mode (strict=True):**
- Requires: structure/atom_positions, metadata group, layers group (non-empty), representations/pair
- Raises `ValueError` on any missing required component
- Validates shapes: (N, 3) for positions, (N, N, D) for pair

**Lenient Mode (strict=False):**
- Requires: structure/atom_positions only (basic structure)
- Missing optional components → warnings only
- Never raises exceptions
- Ideal for incremental workflows

**Example:**
```python
from core import validate_archive

# Complete archive validation
report = validate_archive("trace.zarr", strict=True)
assert report["valid"]  # Raises ValueError if invalid

# Incremental workflow validation
report = validate_archive("trace.zarr", strict=False)
if report["warnings"]:
    print(f"Warnings: {report['warnings']}")
# Never raises even if missing optional components
```

---

### **store.py Methods**

#### METHOD 5: `store_single_representation(path, layer_index, single_array, overwrite=False)`

Store per-layer single (residue-level) representations.

```python
def store_single_representation(
    path: str,
    layer_index: int,
    single_array: np.ndarray,
    overwrite: bool = False
) -> None
```

**Parameters:**
- `path` (str): Archive root
- `layer_index` (int): Transformer layer (0-indexed)
- `single_array` (np.ndarray): Shape `(num_residues, hidden_dim)`
- `overwrite` (bool): Replace if exists

**Archive Location:**
- `representations/single/layer_00`
- `representations/single/layer_01`
- etc.

**Errors:**
- `ValueError`: Array is not 2D
- `FileExistsError`: Layer exists and `overwrite=False`

**Example:**
```python
from store import store_single_representation

# Layer 0 residue embeddings
single = np.random.randn(128, 512)  # 128 residues, 512-dim embedding
store_single_representation("trace.zarr", 0, single)

# Layer 5 residue embeddings
store_single_representation("trace.zarr", 5, single)
```

---

#### METHOD 6: `store_pair_representation(path, layer_index, pair_array, overwrite=False)`

Store per-layer pair (residue-residue) representations.

```python
def store_pair_representation(
    path: str,
    layer_index: int,
    pair_array: np.ndarray,
    overwrite: bool = False
) -> None
```

**Parameters:**
- `path` (str): Archive root
- `layer_index` (int): Transformer layer (0-indexed)
- `pair_array` (np.ndarray): Shape `(tokens, tokens, pair_dim)` — must be square in first 2 dims
- `overwrite` (bool): Replace if exists

**Archive Location:**
- `representations/pair/layer_00`
- `representations/pair/layer_01`
- etc.

**Errors:**
- `ValueError`: Array is not 3D or not square
- `FileExistsError`: Layer exists and `overwrite=False`

**Example:**
```python
from store import store_pair_representation

# Layer 0 pair embeddings: 128×128 tokens × 128 pair dims
pair = np.random.randn(128, 128, 128)
store_pair_representation("trace.zarr", 0, pair)
```

---

#### METHOD 7: `store_attention(path, attention_type, layer_index, attention_array, overwrite=False)`

Store attention head maps organized by type.

```python
def store_attention(
    path: str,
    attention_type: str,
    layer_index: int,
    attention_array: np.ndarray,
    overwrite: bool = False
) -> None
```

**Parameters:**
- `path` (str): Archive root
- `attention_type` (str): Type identifier (e.g., "triangle_start", "triangle_end", "pairwise")
- `layer_index` (int): Transformer layer (0-indexed)
- `attention_array` (np.ndarray): Shape `(num_heads, tokens, tokens)` — must be square
- `overwrite` (bool): Replace if exists

**Archive Location:**
- `attention/{attention_type}/layer_00`
- `attention/{attention_type}/layer_01`
- etc.

**Chunking:**
- Applied automatically: `(1, tokens, tokens)`
- Enables head-by-head loading without loading entire tensor

**Errors:**
- `ValueError`: Array is not 3D, not square, or `attention_type` is empty
- `FileExistsError`: Layer exists and `overwrite=False`

**Example:**
```python
from store import store_attention

# 8 attention heads, 128×128 tokens
attn = np.random.randn(8, 128, 128)

store_attention("trace.zarr", "triangle_start", 0, attn)
store_attention("trace.zarr", "triangle_end", 0, attn)
store_attention("trace.zarr", "pairwise", 0, attn)
```

---

#### METHOD 8: `store_structure_coordinates(path, atom_positions, atom_mask=None, ptm=None, overwrite=False)`

Store predicted 3D structure with optional confidence fields.

```python
def store_structure_coordinates(
    path: str,
    atom_positions: np.ndarray,
    atom_mask: np.ndarray | None = None,
    ptm: float | np.ndarray | None = None,
    overwrite: bool = False
) -> None
```

**Parameters:**
- `path` (str): Archive root
- `atom_positions` (np.ndarray): Shape `(num_residues, 3)` or `(num_residues, num_atoms, 3)`
  - Last dimension must be 3 (x, y, z coordinates)
- `atom_mask` (np.ndarray, optional): Presence mask matching `atom_positions` shape
  - First dimension must equal `num_residues`
- `ptm` (float or scalar array, optional): Predicted TM-score confidence
- `overwrite` (bool): Replace if components exist

**Archive Location:**
- `structure/atom_positions` (required)
- `structure/atom_mask` (optional)
- `structure/ptm` (optional)

**Overwrite Behavior:**
- Each dataset (`atom_positions`, `atom_mask`, `ptm`) checked independently
- If any exists and `overwrite=False` → `FileExistsError` for that dataset

**Errors:**
- `ValueError`: Wrong coordinate dimensions, atom_mask size mismatch, ptm not scalar
- `FileExistsError`: Component exists and `overwrite=False`

**Example:**
```python
from store import store_structure_coordinates

# CA atom coordinates only
positions = np.random.randn(128, 3)
store_structure_coordinates("trace.zarr", positions)

# Full atom coordinates with mask and confidence
positions = np.random.randn(128, 37, 3)  # 128 residues, 37 atoms/residue
mask = np.ones((128, 37), dtype=bool)
ptm_score = 0.92
store_structure_coordinates(
    "trace.zarr",
    positions,
    atom_mask=mask,
    ptm=ptm_score
)
```

---

#### METHOD 9: `store_metadata(path, model_version, config_version, sequence, num_residues, num_recycles, recycle_info=None, residue_index=None, representation_names=None, overwrite=False)`

Store run-level metadata for archive identification and reproducibility.

```python
def store_metadata(
    path: str,
    model_version: str,
    config_version: str,
    sequence: str,
    num_residues: int,
    num_recycles: int,
    recycle_info: np.ndarray | None = None,
    residue_index: np.ndarray | None = None,
    representation_names: list | None = None,
    overwrite: bool = False
) -> None
```

**Parameters:**
- `path` (str): Archive root
- `model_version` (str): Model identifier or release version (e.g., "openfold-v1.0")
- `config_version` (str): Configuration identifier (e.g., "config-r2")
- `sequence` (str): Input amino acid sequence
- `num_residues` (int): Sequence length (must be ≥ 0)
- `num_recycles` (int): Number of inference recycles (must be ≥ 0)
- `recycle_info` (np.ndarray, optional): Per-recycle metadata array
- `residue_index` (np.ndarray, optional): Residue indices, shape `(num_residues,)`
- `representation_names` (list, optional): Names of stored representation types
- `overwrite` (bool): Replace if exists

**Archive Location:**
- `metadata/model_version`
- `metadata/config_version`
- `metadata/sequence`
- `metadata/num_residues`
- `metadata/num_recycles`
- `metadata/recycle_info` (optional)
- `metadata/residue_index` (optional)
- `metadata/representation_names` (optional)

**Errors:**
- `ValueError`: Negative residue/recycle counts, residue_index length mismatch
- `FileExistsError`: Metadata fields exist and `overwrite=False`

**Example:**
```python
from store import store_metadata

store_metadata(
    "trace.zarr",
    model_version="openfold-v2.1",
    config_version="config-r4",
    sequence="MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASED",
    num_residues=65,
    num_recycles=4,
    residue_index=np.arange(65),
    representation_names=["single", "pair"]
)
```

---

### **load.py Methods**

#### METHOD 10: `load_attention_head(path, attention_type, layer_index, head_index)`

Selectively load a single attention head without loading entire tensor.

```python
def load_attention_head(
    path: str,
    attention_type: str,
    layer_index: int,
    head_index: int
) -> np.ndarray
```

**Parameters:**
- `path` (str): Archive root
- `attention_type` (str): Attention type (e.g., "triangle_start", "pairwise")
- `layer_index` (int): Transformer layer (0-indexed)
- `head_index` (int): Attention head index

**Returns:**
- `numpy.ndarray`: Shape `(tokens, tokens)` — single 2D attention matrix

**Errors:**
- `KeyError`: Attention type or layer not found
- `IndexError`: Head index out of range

**Example:**
```python
from load import load_attention_head

# Load head 2 from layer 0 pairwise attention
head = load_attention_head("trace.zarr", "pairwise", 0, 2)
print(head.shape)  # (128, 128)
```

---

#### METHOD 11: `ingest_attention_txt(archive_path, txt_file, layer_index, num_tokens, attention_type="pairwise", overwrite=False)`

Parse text-format attention and store in archive.

```python
def ingest_attention_txt(
    archive_path: str,
    txt_file: str,
    layer_index: int,
    num_tokens: int,
    attention_type: str = "pairwise",
    overwrite: bool = False
) -> dict
```

**Parameters:**
- `archive_path` (str): Archive root
- `txt_file` (str): Path to text file
- `layer_index` (int): Transformer layer to store under
- `num_tokens` (int): Sequence length (must be > 0)
- `attention_type` (str): Attention type identifier
- `overwrite` (bool): Replace if exists

**Expected Input Format:**
```
Layer 0, Head 0
0 1 0.95
0 5 0.87
1 0 0.92
...
Layer 0, Head 1
0 0 1.0
1 1 0.99
...
```

**Returns:**
```python
{
    "layer_index": int,
    "attention_type": str,
    "num_heads": int,
    "num_tokens": int,
    "source_file": str
}
```

**Errors:**
- `ValueError`: `num_tokens` ≤ 0, malformed file, layer mismatch, index out of bounds
- `FileExistsError`: Attention exists and `overwrite=False`

**Example:**
```python
from load import ingest_attention_txt

result = ingest_attention_txt(
    "trace.zarr",
    "attention_layer_0.txt",
    layer_index=0,
    num_tokens=128,
    attention_type="pairwise"
)
print(f"Ingested {result['num_heads']} heads")  # e.g., 8
```

---

#### METHOD 12: `_extract_best_matching_array(container, key_token_patterns)`

Find best array match using tokenized key patterns (internal helper).

```python
def _extract_best_matching_array(
    container: dict | list,
    key_token_patterns: list[list[str]]
) -> dict | None
```

**Parameters:**
- `container` (dict or list): Nested object to search (e.g., pickle output dict)
- `key_token_patterns` (list[list[str]]): Token patterns to match in priority order

**Returns:**
```python
{
    "array": np.ndarray,           # Matched array
    "matched_key": str,             # Full key path
    "pattern": list[str],           # Pattern that matched
    "shape": tuple                  # Array shape
}
```
Returns `None` if no match found.

**Matching Algorithm:**
1. Tokenize all key paths on non-alphanumeric boundaries (e.g., `"final_atom_positions"` → `["final", "atom", "positions"]`)
2. For each pattern in order, check if all pattern tokens appear in path tokens
3. Return first match (depth-first traversal)
4. Scoring prefers more specific patterns and shorter paths

**Example:**
```python
from load import _extract_best_matching_array

pkl_dict = {
    "final_atom_positions": np.random.randn(128, 37, 3),
    "pair_activations": np.random.randn(128, 128, 64),
    "other": {...}
}

# Match "final_atom_positions"
result = _extract_best_matching_array(
    pkl_dict,
    [["final", "atom", "positions"], ["final", "positions"]]
)
print(result["matched_key"])  # "final_atom_positions"
print(result["shape"])         # (128, 37, 3)

# Match "pair_activations"
result = _extract_best_matching_array(
    pkl_dict,
    [["pair", "representation"], ["pair"]]
)
print(result["matched_key"])  # "pair_activations"
```

---

#### METHOD 13: `ingest_output_pkl(archive_path, pkl_file, overwrite=False)`

Extract arrays from OpenFold/VizFold pickle output and route to archive.

```python
def ingest_output_pkl(
    archive_path: str,
    pkl_file: str,
    overwrite: bool = False
) -> dict
```

**Parameters:**
- `archive_path` (str): Archive root
- `pkl_file` (str): Path to `.pkl` file
- `overwrite` (bool): Replace if components exist

**Auto-Routing Behavior:**

| Key Pattern | Destination | Extraction | Notes |
|-----------|-----------|-----------|-------|
| `["final", "atom", "positions"]` | `structure/atom_positions` | CA atom (index 1) or C (index 0) | Converts (N, 37, 3) → (N, 3) |
| `["final", "atom", "mask"]` | `structure/atom_mask` | CA atom mask (index 1) or first | Optional |
| `["ptm"]` | `structure/ptm` | Scalar value | Optional confidence |
| `["pair", ...]` | `representations/pair/layer_00` | Full 3D array | Stored as layer 0 |

**Returns:**
```python
{
    "source_file": str,
    "stored": list[str],           # Successfully stored paths
    "skipped": list[str],          # Skipped with reasons
    "key_matches": {               # Traceability for each key
        "final_positions": {
            "pattern": list,
            "matched_key": str | None,
            "shape": tuple | None
        },
        "final_atom_mask": {...},
        "ptm": {...},
        "pair_representation": {...}
    }
}
```

**Example Output:**
```python
{
    "source_file": "output.pkl",
    "stored": [
        "structure/atom_positions",
        "structure/atom_mask",
        "structure/ptm",
        "representations/pair/layer_00"
    ],
    "skipped": [],
    "key_matches": {
        "final_positions": {
            "pattern": ["final", "atom", "positions"],
            "matched_key": "final_atom_positions",
            "shape": (128, 37, 3)
        },
        "final_atom_mask": {
            "pattern": ["final", "atom", "mask"],
            "matched_key": "final_atom_mask",
            "shape": (128, 37)
        },
        "ptm": {
            "pattern": ["ptm"],
            "matched_key": "plddt",  # Note: fuzzy match
            "shape": ()
        },
        "pair_representation": {
            "pattern": ["pair"],
            "matched_key": "pair_activations",
            "shape": (128, 128, 64)
        }
    }
}
```

**Errors:**
- `ValueError`: Pickle is not a dict
- `FileExistsError`: Components exist and `overwrite=False` (caught and reported in summary)

**Example:**
```python
from load import ingest_output_pkl

result = ingest_output_pkl("trace.zarr", "openfold_output.pkl")

print(f"Stored: {result['stored']}")
print(f"Skipped: {result['skipped']}")

# Check which keys matched
print(f"Final positions matched: {result['key_matches']['final_positions']['matched_key']}")
```

---

#### METHOD 14: `_load_dataset_as_python_value(dataset)`

Normalize 0-D Zarr datasets to Python scalars (internal helper).

```python
def _load_dataset_as_python_value(dataset: zarr.Array) -> Any
```

**Behavior:**
- 0-D array (shape `()`) → `.item()` to get Python scalar
- n-D array → return as ndarray

---

#### METHOD 15: `load_metadata(path)`

Load run metadata as a dictionary.

```python
def load_metadata(path: str) -> dict
```

**Parameters:**
- `path` (str): Archive root

**Returns:**
```python
{
    "model_version": str,
    "config_version": str,
    "sequence": str,
    "num_residues": int,
    "num_recycles": int,
    "recycle_info": np.ndarray,      # optional
    "residue_index": np.ndarray,     # optional
    "representation_names": list     # optional
}
```

**Errors:**
- `KeyError`: Metadata group missing or required fields incomplete

**Example:**
```python
from load import load_metadata

meta = load_metadata("trace.zarr")
print(f"Model: {meta['model_version']}")
print(f"Sequence: {meta['sequence']}")
print(f"Residues: {meta['num_residues']}")
```

---

#### METHOD 16: `load_single_representation(path, layer_index)`

Load single representation for a specific layer.

```python
def load_single_representation(path: str, layer_index: int) -> np.ndarray
```

**Parameters:**
- `path` (str): Archive root
- `layer_index` (int): Transformer layer (0-indexed)

**Returns:**
- `numpy.ndarray`: Shape `(num_residues, hidden_dim)`

**Errors:**
- `KeyError`: Layer not found

**Example:**
```python
from load import load_single_representation

single = load_single_representation("trace.zarr", 0)
print(single.shape)  # (128, 512)
```

---

#### METHOD 17: `load_pair_representation(path, layer_index)`

Load pair representation for a specific layer.

```python
def load_pair_representation(path: str, layer_index: int) -> np.ndarray
```

**Parameters:**
- `path` (str): Archive root
- `layer_index` (int): Transformer layer (0-indexed)

**Returns:**
- `numpy.ndarray`: Shape `(tokens, tokens, pair_dim)`

**Errors:**
- `KeyError`: Layer not found

**Example:**
```python
from load import load_pair_representation

pair = load_pair_representation("trace.zarr", 0)
print(pair.shape)  # (128, 128, 128)
```

---

#### METHOD 18: `ArchiveOrchestrator`

Thin helper class for sequencing writes and recording events.

```python
class ArchiveOrchestrator:
    def __init__(self, archive_path: str)
    
    def add_metadata(self, *args, **kwargs) -> dict
    def add_single_layer(self, layer_index, single_array, overwrite=False) -> dict
    def add_pair_layer(self, layer_index, pair_array, overwrite=False) -> dict
    def add_attention(self, attention_type, layer_index, attention_array, overwrite=False) -> dict
    def add_structure(self, atom_positions, atom_mask=None, ptm=None, overwrite=False) -> dict
    def validate(self, validator=validate_archive, *args, **kwargs) -> dict
    def summary(self) -> dict
```

**Purpose:**
- Coordinate a sequence of writes
- Record what was written (event log)
- Provide structured summary for debugging

**Key Methods:**

**`add_metadata(...)`** — Record metadata write
**`add_single_layer(layer_index, single_array, overwrite=False)`** — Record single representation write
**`add_pair_layer(layer_index, pair_array, overwrite=False)`** — Record pair representation write
**`add_attention(attention_type, layer_index, attention_array, overwrite=False)`** — Record attention write
**`add_structure(atom_positions, atom_mask=None, ptm=None, overwrite=False)`** — Record structure write
**`validate(validator=validate_archive, *args, **kwargs)`** — Run validation and record result
**`summary()`** — Return event log and archive path

**Example:**
```python
from load import ArchiveOrchestrator

orchestrator = ArchiveOrchestrator("trace.zarr")

# Build archive incrementally with event logging
orchestrator.add_metadata(
    model_version="openfold-v1.0",
    config_version="config-r2",
    sequence="MVLSEGEWQLVL...",
    num_residues=65,
    num_recycles=4
)

orchestrator.add_single_layer(0, single_layer_0)
orchestrator.add_pair_layer(0, pair_layer_0)
orchestrator.add_attention("pairwise", 0, attn_layer_0)
orchestrator.add_structure(atom_positions, atom_mask=mask, ptm=0.92)

# Validate at end
report = orchestrator.validate(strict=False)

# Get summary
summary = orchestrator.summary()
print(summary)
# Output:
# {
#     "archive_path": "trace.zarr",
#     "events": [
#         {"action": "store", "target": "metadata"},
#         {"action": "store", "target": "representations/single/layer_00"},
#         ...
#         {"action": "validate", "target": "archive", "result": {...}}
#     ]
# }
```

---

## User Workflows

### Workflow 1: Complete Archive from OpenFold Output

```python
from load import ingest_output_pkl, ArchiveOrchestrator
from store import store_metadata

pkl_path = "openfold_output.pkl"
archive_path = "trace.zarr"

# Step 1: Ingest pickle (auto-extracts structure, pair representation)
result = ingest_output_pkl(archive_path, pkl_path)
print(f"Stored: {result['stored']}")

# Step 2: Add metadata
store_metadata(
    archive_path,
    model_version="openfold-v2.1",
    config_version="config-r4",
    sequence="MVLSEGEWQLVL...",
    num_residues=65,
    num_recycles=4
)

# Step 3: Add single/pair representations for each layer
from store import store_single_representation, store_pair_representation
for layer_idx in range(48):
    store_single_representation(archive_path, layer_idx, single_layer[layer_idx])
    store_pair_representation(archive_path, layer_idx, pair_layer[layer_idx])

# Step 4: Validate
from core import validate_archive
report = validate_archive(archive_path, strict=True)
assert report["valid"]
```

---

### Workflow 2: Incremental Building with Safety

```python
from load import ArchiveOrchestrator
from core import validate_archive

orchestrator = ArchiveOrchestrator("trace.zarr")

# Build incrementally, safely
try:
    orchestrator.add_metadata(
        model_version="vizfold-v3",
        config_version="config-exp",
        sequence="...",
        num_residues=256,
        num_recycles=8
    )
    
    # Add layer 0
    orchestrator.add_single_layer(0, single_0)
    orchestrator.add_pair_layer(0, pair_0)
    orchestrator.add_attention("triangle_start", 0, attn_start_0)
    orchestrator.add_attention("triangle_end", 0, attn_end_0)
    
    # Validate at checkpoint (lenient)
    report = orchestrator.validate(strict=False)
    print(f"Checkpoint valid: {report['valid']}")
    
    # Add structure when ready
    orchestrator.add_structure(atom_positions, ptm=ptm_score)
    
    # Final validation (strict)
    final_report = orchestrator.validate(strict=True)
    
    print(orchestrator.summary())
    
except FileExistsError:
    print("Data already exists. Set overwrite=True to replace.")
```

---

### Workflow 3: Visualization Server (Selective Loading)

```python
from load import load_metadata, load_attention_head, load_pair_representation

# Load metadata once at startup
meta = load_metadata("trace.zarr")
print(f"Archive: {meta['model_version']} | {meta['sequence']}")

# Load only requested heads on demand
def get_attention_for_visualization(layer: int, head: int, attn_type: str):
    return load_attention_head("trace.zarr", attn_type, layer, head)

def get_pair_for_analysis(layer: int):
    return load_pair_representation("trace.zarr", layer)

# User requests layer 5, head 2, pairwise attention
head_data = get_attention_for_visualization(5, 2, "pairwise")
# Only ~16 KB loaded instead of entire 8 MB tensor
```

---

### Workflow 4: Archive Validation in CI/CD

```python
from core import validate_archive
import sys

# During testing/deployment, enforce strict validation
try:
    report = validate_archive("trace.zarr", strict=True)
    print(f"✓ Archive valid | Components found: {report['components_found']}")
    sys.exit(0)
except ValueError as e:
    print(f"✗ Archive invalid: {e}")
    sys.exit(1)
```

---

## Archive Structure

### VizFold 1.0 Format (Zarr-based)

```
trace.zarr/
├── metadata/                           # Run context
│   ├── model_version                   # str scalar
│   ├── config_version                  # str scalar
│   ├── sequence                        # str scalar
│   ├── num_residues                    # int32 scalar
│   ├── num_recycles                    # int32 scalar
│   ├── recycle_info (optional)         # array
│   ├── residue_index (optional)        # array (num_residues,)
│   └── representation_names (optional) # array
│
├── representations/
│   ├── single/                         # Per-residue embeddings
│   │   ├── layer_00                    # (num_residues, hidden_dim)
│   │   ├── layer_01
│   │   └── ...
│   │
│   └── pair/                           # Token-token relationships
│       ├── layer_00                    # (tokens, tokens, pair_dim)
│       ├── layer_01
│       └── ...
│
├── attention/
│   ├── triangle_start/
│   │   ├── layer_00                    # (num_heads, tokens, tokens)
│   │   ├── layer_01
│   │   └── ...
│   │
│   ├── triangle_end/
│   │   ├── layer_00
│   │   └── ...
│   │
│   └── pairwise/
│       ├── layer_00
│       └── ...
│
└── structure/                          # 3D coordinates
    ├── atom_positions                  # (num_residues, 3) or (N, num_atoms, 3)
    ├── atom_mask (optional)            # Same shape as positions
    └── ptm (optional)                  # Scalar float
```

### Layer Naming Convention

All per-layer datasets use zero-padded naming:
- `layer_00`, `layer_01`, ..., `layer_09`, ..., `layer_47`

This enables:
- Lexicographic sorting
- Fixed-width parsing
- Scalability to 1000+ layers

### Chunking Strategy

| Component | Chunking | Rationale |
|-----------|----------|-----------|
| single representations | None (default) | Small per-layer, full load acceptable |
| pair representations | None (default) | Square matrix, full load typical |
| attention | `(1, tokens, tokens)` | Head-by-head loading in visualization |
| structure/atom_positions | None (default) | Typically small (128–1024 residues) |

---

## Validation & Safety

### Overwrite Protection

All store methods default to `overwrite=False`:

```python
# This raises FileExistsError if layer_00 exists
store_pair_representation(archive, 0, pair_array)

# Explicit re-run protection
store_pair_representation(archive, 0, pair_array, overwrite=False)

# Intentional replacement
store_pair_representation(archive, 0, updated_pair, overwrite=True)
```

### Strict vs. Lenient Validation

| Mode | strict=True | strict=False |
|------|-----------|------------|
| Use Case | Production, complete archives | Development, incremental builds |
| Behavior | Raises exceptions | Returns report with warnings |
| Required Components | metadata, structure, representations/pair, ≥1 layer | structure/atom_positions only |
| Optional Components | Error if missing | Warnings if missing |
| Return Type | Raises or returns valid report | Always returns report (never raises) |

**Example:**
```python
# Production validation
try:
    report = validate_archive("trace.zarr", strict=True)
    # If this returns, archive is guaranteed complete
except ValueError as e:
    print(f"Archive incomplete: {e}")

# Development validation
report = validate_archive("trace.zarr", strict=False)
if report["warnings"]:
    print(f"Warnings: {report['warnings']}")
# Never raises, safe for incremental workflows
```

---

## Pickle Ingestion & Traceability

### Key Matching Algorithm

The `_extract_best_matching_array()` function uses **tokenized matching** instead of substring search for robustness.

**Tokenization:**
```
"final_atom_positions" → ["final", "atom", "positions"]
"final_positions" → ["final", "positions"]
"finalAtomPositions" → ["final", "atom", "positions"]
```

**Matching:**
1. For pattern `["final", "atom", "positions"]`, check if ALL tokens appear in path tokens
2. Return first match (depth-first)
3. Score based on pattern specificity and path length

**Benefits:**
- Avoids false positives (e.g., "atom" wouldn't match "atom_positions" incorrectly)
- Handles naming variations (snake_case, camelCase, PascalCase)
- Provides traceability: shows exactly what matched

### Traceability Example

```python
result = ingest_output_pkl("trace.zarr", "output.pkl")

# View exact matches
print(result["key_matches"])
# Output:
# {
#     "final_positions": {
#         "pattern": ["final", "atom", "positions"],
#         "matched_key": "final_atom_positions",
#         "shape": (128, 37, 3)
#     },
#     "pair_representation": {
#         "pattern": ["pair"],
#         "matched_key": "pair_representation",
#         "shape": (128, 128, 128)
#     }
# }
```

This allows users to:
- Verify correct keys were matched
- Debug ingestion failures
- Audit data flow for compliance/reproducibility

---

## API Patterns & Guarantees

### Pattern 1: Consistent Overwrite Default

**All store methods** use `overwrite=False` by default:

```python
store_single_representation(path, layer, array)              # ✓ raises if exists
store_pair_representation(path, layer, array)               # ✓ raises if exists
store_attention(path, type, layer, array)                   # ✓ raises if exists
store_structure_coordinates(path, coords)                   # ✓ raises if exists
store_metadata(path, ...)                                   # ✓ raises if exists
ingest_attention_txt(path, file, layer, tokens)            # ✓ raises if exists
ingest_output_pkl(path, pkl)                               # ✓ raises if exists
```

**Rationale**: Protect against accidental overwrites in production workflows.

---

### Pattern 2: Layer Indexing (0-indexed, zero-padded naming)

All layer storage uses zero-padded layer names:

```python
store_single_representation(path, 0, array)  # → representations/single/layer_00
store_single_representation(path, 5, array)  # → representations/single/layer_05
store_single_representation(path, 47, array) # → representations/single/layer_47
```

---

### Pattern 3: Nested Path Parsing with `::`

`tensor_to_zarr_array()` supports both direct and nested paths:

```python
# Direct Zarr array file
tensor_to_zarr_array("file.zarr", array)

# Nested within archive using :: separator
tensor_to_zarr_array("archive.zarr::group/subgroup/dataset", array)
```

---

### Pattern 4: Error Propagation

| Error Type | Raised By | Typical Cause | Handling |
|-----------|-----------|---------------|----------|
| `ValueError` | All validate functions | Invalid input (shape, type, range) | Validate inputs before calling |
| `FileExistsError` | All store functions | Data exists and overwrite=False | Set overwrite=True or use lenient mode |
| `KeyError` | All load functions | Component not found | Check archive structure first |
| `IndexError` | `load_attention_head()` | Head index out of range | Check num_heads from metadata |

---

### Pattern 5: Tensor Flexibility

All methods accept mixed tensor types:

```python
# PyTorch tensor
store_single_representation(path, 0, torch_tensor)

# NumPy array
store_single_representation(path, 0, numpy_array)

# Python list
store_single_representation(path, 0, [[1, 2], [3, 4]])

# All converted to NumPy internally via tensor_to_numpy()
```

---

## Testing Checklist

### Overwrite Safety
- [ ] New data: `overwrite=False` → stores successfully
- [ ] Existing data: `overwrite=False` → raises `FileExistsError`
- [ ] Existing data: `overwrite=True` → replaces successfully
- [ ] Consistency across all 7 store functions

### Strict Validation
- [ ] Complete archive → `valid=True`
- [ ] Missing metadata → raises `ValueError`
- [ ] Missing layers → raises `ValueError`
- [ ] Missing pair representation → raises `ValueError`
- [ ] Invalid coordinate shape → raises `ValueError`

### Lenient Validation
- [ ] Complete archive → `valid=True`
- [ ] Missing optional components → warnings only
- [ ] Never raises exceptions
- [ ] Returns detailed report

### Pickle Ingestion
- [ ] `ingest_output_pkl()` extracts and routes all components
- [ ] `key_matches` traceability shows matched keys and shapes
- [ ] Correctly extracts CA atom from (N, 37, 3) positions
- [ ] Handles missing optional components gracefully

### Loaders
- [ ] `load_metadata()` returns complete dict
- [ ] `load_single_representation()` returns correct shape
- [ ] `load_pair_representation()` returns correct shape
- [ ] `load_attention_head()` returns 2D matrix without loading full tensor

### Orchestrator
- [ ] `ArchiveOrchestrator.summary()` records all events
- [ ] `ArchiveOrchestrator.validate()` integration works
- [ ] Event log shows correct sequence of writes

---

## Implementation Notes

### Why Three Modules?

1. **core.py (shared)**: All store and load operations depend on tensor conversion and validation
2. **store.py (write-only)**: Focused on input validation and archive construction
3. **load.py (read + orchestration)**: Includes loading, parsing external formats, and write coordination

This split enables:
- Independent testing of each concern
- Clear import dependencies (no circular imports)
- Easier team onboarding (each module has one responsibility)
- Future extensibility (new stores don't touch loads, etc.)

---

## Performance Characteristics

| Operation | Complexity | Memory | Notes |
|-----------|-----------|--------|-------|
| `tensor_to_zarr_array()` | O(n) write | O(n) | Streaming possible with zarr chunks |
| `load_attention_head()` | O(1) seek + O(tokens²) read | O(tokens²) | Head chunking enables selective loading |
| `ingest_output_pkl()` | O(n) parse | O(n) | Single pass through pickle dict |
| `validate_archive()` | O(m) where m = num datasets | O(1) | Metadata-only checks by default |
| `_extract_best_matching_array()` | O(k) depth-first | O(path depth) | Early termination on match |

---

## Changelog from Previous Spec

### Version 1.0 → 2.0

**New in v2.0:**
1. Refactored into three modules (core.py, store.py, load.py)
2. Added `ArchiveOrchestrator` class for write sequencing
3. Enhanced pickle ingestion with `_extract_best_matching_array()` and `key_matches` traceability
4. Standardized `overwrite=False` default across all store methods
5. Added strict/lenient validation modes in `validate_archive()`
6. Implemented selective head loading via chunking
7. Added `store_metadata()` as first-class method
8. Added `load_metadata()` paired loader
9. Unified `tensor_to_numpy()` for all input types

**Breaking Changes:**
- None (all methods maintain backward compatibility)

**Deprecated:**
- None

---

## Contact & Support

For questions or issues with the VizFold Archive Utilities API:
1. Check the user workflows above
2. Review the method reference for your specific use case
3. Run `validate_archive(..., strict=False)` to check archive integrity
4. Inspect `ingest_output_pkl()` results for `key_matches` traceability
