# Archive Utilities - Method Specifications & Flow

Complete specification of all methods with input/output flows and error handling based on latest implementation.

---

## **METHOD 1: `tensor_to_numpy(tensor)`**

**Purpose:** Convert any tensor type to NumPy array

**Input:**
- `tensor`: torch.Tensor | numpy.ndarray | array-like

**Flow:**
1. Check if already numpy array → return as-is
2. Check if has PyTorch attributes (`detach`, `cpu`) → detach, move to CPU, convert
3. Fallback → use `np.asarray()`

**Output:** `numpy.ndarray`

**Errors:** None (always succeeds with best-effort conversion)

---

## **METHOD 2: `tensor_to_zarr_array(path, tensor, chunks=None, overwrite=False)`**

**Purpose:** Store any tensor directly to Zarr array with overwrite control

**Input:**
- `path`: str (can be `archive.zarr::group/dataset` or file path)
- `tensor`: torch.Tensor | numpy.ndarray
- `chunks`: tuple (optional)
- `overwrite`: bool = False

**Flow:**
1. Convert tensor to NumPy via `tensor_to_numpy()`
2. Parse path:
   - If contains `::` → **archive mode** (nested path)
     - Split into `archive_path` and `dataset_path`
     - Open archive in append mode
     - Create parent groups recursively
     - Check if dataset exists
3. Path parsing:
   - If `::` not present → **file mode** (direct Zarr array)
4. **Overwrite Logic:**
   - If data exists AND `overwrite=False` → `FileExistsError` ✗
   - If data exists AND `overwrite=True` → delete and replace ✓
   - If no data exists → create new ✓
5. Write array to location with optional chunking

**Output:** Zarr array reference

**Errors:**
- `ValueError`: Empty dataset path
- `FileExistsError`: Data exists and `overwrite=False`

---

## **METHOD 3: `store_layer_activation(path, layer_index, activation_array, overwrite=False)`**

**Purpose:** Store transformer layer activations with consistent overwrite handling

**Input:**
- `path`: str (Zarr archive root)
- `layer_index`: int
- `activation_array`: numpy.ndarray (expected shape: `(tokens, hidden_dim)`)
- `overwrite`: bool = False

**Flow:**
1. Convert to NumPy via `tensor_to_numpy()`
2. **Validate shape:**
   - Expect 2D array
   - If not 2D → `ValueError` ✗
3. Open Zarr archive in append mode
4. Create/get `layers/{layer_index}` group
5. **Overwrite Check:**
   - If `activation` exists AND `overwrite=False` → `FileExistsError` ✗
   - If `activation` exists AND `overwrite=True` → delete first ✓
6. Store array in group

**Output:** None

**Errors:**
- `ValueError`: Wrong tensor dimensions
- `FileExistsError`: Activation exists and `overwrite=False`

---

## **METHOD 4: `store_pair_representation(path, pair_array, overwrite=False)`**

**Purpose:** Store pair representation embeddings with validation

**Input:**
- `path`: str (Zarr archive root)
- `pair_array`: numpy.ndarray (expected shape: `(tokens, tokens, pair_dim)`)
- `overwrite`: bool = False

**Flow:**
1. Convert to NumPy via `tensor_to_numpy()`
2. **Validate shape:**
   - Expect 3D array → if not → `ValueError` ✗
   - First two dims must be equal (square matrix) → if not → `ValueError` ✗
3. Open Zarr archive in append mode
4. Get/create `representations` group
5. **Overwrite Check:**
   - If `pair` exists AND `overwrite=False` → `FileExistsError` ✗
   - If `pair` exists AND `overwrite=True` → delete first ✓
6. Store array in group

**Output:** None

**Errors:**
- `ValueError`: Wrong dimensions or non-square shape
- `FileExistsError`: Pair exists and `overwrite=False`

---

## **METHOD 5: `store_attention_heads(path, layer_index, attention_array, overwrite=False)`**

**Purpose:** Store attention maps with smart chunking and overwrite control

**Input:**
- `path`: str (Zarr archive root)
- `layer_index`: int
- `attention_array`: numpy.ndarray (expected shape: `(num_heads, tokens, tokens)`)
- `overwrite`: bool = False

**Flow:**
1. Convert to NumPy (handles PyTorch tensors directly with `.detach().cpu()`)
2. **Validate shape:**
   - Expect 3D array → if not → `ValueError` ✗
   - Middle two dims must be equal (square per head) → if not → `ValueError` ✗
   - Extract `num_heads`, `tokens_i`, `tokens_j`
3. Open Zarr archive in append mode
4. Get/create `layers/{layer_index}` group
5. **Overwrite Check:**
   - If `attention` exists AND `overwrite=False` → `FileExistsError` ✗
   - If `attention` exists AND `overwrite=True` → delete first ✓
6. Set chunks to `(1, tokens_i, tokens_j)` for efficient head-by-head loading
7. Store array with chunking

**Output:** None

**Errors:**
- `ValueError`: Wrong dimensions or non-square attention matrix
- `FileExistsError`: Attention exists and `overwrite=False`

---

## **METHOD 6: `store_structure_coordinates(path, coordinates, residue_types=None, overwrite=False)`**

**Purpose:** Store protein structure coordinates with optional residue types

**Input:**
- `path`: str (Zarr archive root)
- `coordinates`: numpy.ndarray (expected shape: `(num_residues, 3)`)
- `residue_types`: numpy.ndarray or None (expected shape: `(num_residues,)`)
- `overwrite`: bool = False

**Flow:**
1. Convert coordinates to NumPy
2. **Validate coordinates:**
   - Expect 2D → if not → `ValueError` ✗
   - Last dim must be 3 (x, y, z) → if not → `ValueError` ✗
   - Extract `num_residues`
3. Open Zarr archive in append mode
4. Get/create `structure` group
5. **Overwrite Check for coordinates:**
   - If exists AND `overwrite=False` → `FileExistsError` ✗
   - If exists AND `overwrite=True` → delete first ✓
6. Store coordinates array
7. **If residue_types provided:**
   - Convert to NumPy
   - Validate length matches `num_residues` → if not → `ValueError` ✗
   - **Overwrite Check for residue_types:**
     - If exists AND `overwrite=False` → `FileExistsError` ✗
     - If exists AND `overwrite=True` → delete first ✓
   - Store residue_types array
8. **If residue_types not provided:** Skip storage

**Output:** None

**Errors:**
- `ValueError`: Wrong coordinate shape or mismatched residue_types length
- `FileExistsError`: Any component exists and `overwrite=False`

---

## **METHOD 7: `load_attention_head(path, layer_index, head_index)` [STUB]**

**Purpose:** Load single attention head for efficient visualization

**Input:**
- `path`: str (Zarr archive root)
- `layer_index`: int
- `head_index`: int

**Current Status:** Not implemented (`pass`)

**Expected Flow (when implemented):**
1. Open archive in read mode
2. Navigate to `layers/{layer_index}/attention`
3. Load only the specific head: `attention[head_index]`
4. Return as `(tokens, tokens)` numpy array

---

## **METHOD 8: `validate_archive(path, strict=True)`**

**Purpose:** Validate archive integrity with flexible strictness

**Input:**
- `path`: str (Zarr archive root)
- `strict`: bool = True

**Flow:**

### **ALWAYS CHECK (both modes):**
1. Path exists → if not → Error/Warn ✗
2. Can open as Zarr → if not → Error/Warn ✗
3. `structure/` group exists → if not → Error ✗
4. `structure/coordinates` dataset exists → if not → Error ✗
5. Coordinates shape is `(N, 3)` → if not → Error ✗

### **STRICT MODE (strict=True):**
1. All checks above → if fail → `ValueError` with message, `report['valid']=False` ✗
2. `layers/` group must exist → if not → Raise `ValueError` ✗
3. `layers/` must have ≥1 layer → if empty → Raise `ValueError` ✗
4. `representations/` group must exist → if not → Raise `ValueError` ✗
5. `representations/pair` must exist → if not → Raise `ValueError` ✗
6. `representations/pair` shape is `(N, N, D)` → if not → Raise `ValueError` ✗
7. If all pass → return `report['valid']=True` ✓

### **LENIENT MODE (strict=False):**
1. Basic checks (path, Zarr format, structure/coordinates) → if fail → Error + Warning ✗
2. Optional groups → if missing → Warning, continue ⚠
3. Layer data → if `layers/` empty → Warning ⚠
4. Representations → if missing → Warning ⚠
5. Shape validation issues → Warning ⚠
6. Always return report without raising exceptions ✓

**Output:** `dict` with:
```python
{
    'valid': bool,              # overall status
    'strict_mode': bool,        # mode used
    'path': str,                # checked path
    'errors': list,             # critical issues
    'warnings': list,           # soft issues
    'components_found': dict    # component status
}
```

**Errors (Strict Only):**
- `FileNotFoundError`: Path doesn't exist
- `ValueError`: Zarr open failed, required component missing, or shape invalid

---

## **METHOD 9: `ingest_attention_txt(archive_path, txt_file, layer_index, num_tokens, overwrite=False)`**

**Purpose:** Parse attention text file and store in archive

**Input:**
- `archive_path`: str (Zarr archive root)
- `txt_file`: str (path to `.txt` file)
- `layer_index`: int
- `num_tokens`: int
- `overwrite`: bool = False

**Flow:**
1. **Validate num_tokens:**
   - Must be > 0 → if not → `ValueError` ✗

2. **Parse text file:**
   - Pattern: `Layer <idx>, Head <idx>`
   - Each attention entry: `<res_i> <res_j> <score>`
   - Line handling:
     - Skip blank lines
     - Parse headers → extract layer & head indices
     - Layer index in file must match arg → if not → `ValueError` ✗
     - Parse entries → extract indices and scores
     - Validate indices 0 ≤ res_i, res_j < num_tokens → if not → `ValueError` ✗
     - Expect exactly 3 fields per row → if not → `ValueError` ✗

3. **Build attention array:**
   - Create `(max_head+1, num_tokens, num_tokens)` array of zeros
   - Fill entries from parsed heads
   - dtype = `float32`

4. **Store via `store_attention_heads()`:**
   - Pass `overwrite` parameter through
   - Will raise `FileExistsError` if exists and `overwrite=False`

5. **Return metadata:**
   ```python
   {
       'layer_index': int,
       'num_heads': int,
       'num_tokens': int,
       'source_file': str
   }
   ```

**Errors:**
- `ValueError`: num_tokens ≤ 0, malformed file, layer mismatch, index out of bounds
- `FileExistsError`: Attention exists and `overwrite=False` (from `store_attention_heads`)

---

## **METHOD 10: `ingest_output_pkl(archive_path, pkl_file, overwrite=False)`**

**Purpose:** Extract and route arrays from OpenFold/VizFold pickle output

**Input:**
- `archive_path`: str (Zarr archive root)
- `pkl_file`: str (path to `.pkl` file)
- `overwrite`: bool = False

**Flow:**

1. **Load pickle:**
   - Load file → if not dict → `ValueError` ✗

2. **Extract `final_positions`:**
   - Search for keys containing: `["final", "atom", "positions"]`
   - Via `_extract_first_matching_array()` depth-first search
   - If found:
     - Validate shape is 3D and last dim = 3 → if not → Skip + message ⚠
     - Extract CA atom (index 1): `coords = positions[:, 1, :]`
     - Call `store_structure_coordinates(archive_path, coords, residue_types, overwrite)`
     - Catch `FileExistsError` → add to skipped list ⚠
     - If success → add "structure/coordinates" to `stored` ✓
   - If not found → add to `skipped` ⚠

3. **Extract `residue_types`:**
   - First try: `["aatype"]`
   - Fallback: `["residue", "type"]`
   - If found during coordinates store → already handled ✓

4. **Extract `pair` representation:**
   - Search for keys containing: `["pair"]`
   - If found:
     - Validate 3D and shape[0] == shape[1] → if not → Skip + message ⚠
     - Call `store_pair_representation(archive_path, pair_array, overwrite)`
     - Catch `FileExistsError` → add to skipped ⚠
     - If success → add "representations/pair" to `stored` ✓
   - If not found → add to `skipped` ⚠

5. **Return summary:**
   ```python
   {
       'source_file': str,       # pickle path
       'stored': [list],         # successfully stored components
       'skipped': [list]         # skipped with reasons
   }
   ```

**Errors:**
- `ValueError`: Pickle not a dict
- `FileExistsError`: Components exist and `overwrite=False` (caught and reported in summary)

---

## **HELPER: `_extract_first_matching_array(container, key_substrings)`**

**Purpose:** Depth-first search for first matching array in nested dict/list structure

**Input:**
- `container`: dict or list
- `key_substrings`: list of strings

**Flow:**
1. Recursively traverse dict/list structure
2. For each key path: lowercase and check if ALL substrings present
3. Try convert value to numpy array via `tensor_to_numpy()`
4. Return first match found (depth-first order)
5. Return `None` if no matches

**Output:** `numpy.ndarray | None`

---

## **SUMMARY: Overwrite & Validation Behavior**

### **Overwrite Parameter Behavior**

| Action | Default | When overwrite=False | When overwrite=True |
|--------|---------|----------------------|---------------------|
| **Store new data** | ✓ | ✓ | ✓ |
| **Data exists** | ✗ | Raise `FileExistsError` | Delete & replace |
| **store_layer_activation** | `overwrite=False` | Protective | Allows update |
| **store_pair_representation** | `overwrite=False` | Protective | Allows update |
| **store_attention_heads** | `overwrite=False` | Protective | Allows update |
| **store_structure_coordinates** | `overwrite=False` | Protective | Allows update |
| **ingest_attention_txt** | `overwrite=False` | Protective | Allows update |
| **ingest_output_pkl** | `overwrite=False` | Protective | Allows update |

### **Validation Parameter Behavior**

| Validation Check | strict=True | strict=False |
|-----------------|------------|-------------|
| **Path exists** | Error ✗ | Error + Warning |
| **Can open Zarr** | Error ✗ | Error + Warning |
| **structure/coordinates** | Required ✓ | Required ✓ |
| **layers/** group | Required ✓ | Optional ⚠ |
| **layers/ not empty** | Required ✓ | Optional ⚠ |
| **representations/pair** | Required ✓ | Optional ⚠ |
| **Coordinate shape** | (N, 3) ✓ | (N, 3) ✓ |
| **Pair shape** | (N, N, D) ✓ | (N, N, D) ⚠ |
| **Exception on failure** | Yes | No |
| **Use case** | Production | Incremental |

---

## **Testing Checklist**

### **Overwrite Functionality**
- [ ] New data: `overwrite=False` → stores ✓
- [ ] Existing data: `overwrite=False` → `FileExistsError` ✗
- [ ] Existing data: `overwrite=True` → replaces ✓
- [ ] All storage functions consistent ✓

### **Validation - Strict Mode**
- [ ] Complete archive → `valid=True` ✓
- [ ] Missing structure → raises `ValueError` ✗
- [ ] Missing layers → raises `ValueError` ✗
- [ ] Missing pair → raises `ValueError` ✗
- [ ] Bad coordinate shape → raises `ValueError` ✗
- [ ] No exceptions raised for lenient mode ✓

### **Validation - Lenient Mode**
- [ ] Complete archive → `valid=True` ✓
- [ ] Missing optional groups → warns ⚠
- [ ] Empty layers → warns ⚠
- [ ] Missing pair → warns ⚠
- [ ] Never raises exceptions ✓
- [ ] Returns report with warnings ✓

### **Ingestion Functions**
- [ ] `ingest_attention_txt` passes overwrite ✓
- [ ] `ingest_output_pkl` passes overwrite ✓
- [ ] `FileExistsError` caught and reported ✓
- [ ] Skipped items tracked ✓

---

## **Usage Examples**

### **Safe First-Time Write**
```python
store_attention_heads(archive, 0, attn)
# overwrite=False (default) → stores if new
```

### **Protected Re-run**
```python
store_attention_heads(archive, 0, attn)
# raises FileExistsError if exists
```

### **Intentional Update**
```python
store_attention_heads(archive, 0, attn_refined, overwrite=True)
# replaces existing data
```

### **Incremental Building**
```python
validate_archive(archive, strict=False)
# returns warnings but doesn't fail
```

### **Final Validation**
```python
validate_archive(archive, strict=True)
# raises if anything required is missing
```
