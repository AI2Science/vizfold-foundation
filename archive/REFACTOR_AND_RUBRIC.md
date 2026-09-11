# Archive Utilities Refactor & PR Rubric Guide

## Overview

This document outlines the necessary refactors to align our archive utilities with the VizFold 1.0 specification (Group #39) and the requirements for meeting the 45% PR rubric.

### Why Refactor?

The VizFold Inference Trace Archive specification (v1.0, March 2026) defines a formal standard for Zarr-based storage of inference traces. Our current implementation uses a different path structure and organization. Aligning with this spec ensures:

- ✅ Interoperability with other VizFold tools
- ✅ Compatibility with standard visualization/analysis pipelines
- ✅ Future-proofing for ecosystem adoption
- ✅ Clear, documented standards compliance

---

## Part 1: Required Method Refactors

### Current vs. VizFold 1.0 Specification

#### Archive Structure Comparison

**Current Structure:**
```
archive.zarr/
├── structure/
│   ├── coordinates
│   └── residue_types
├── representations/
│   └── pair
└── layers/
    ├── 0/
    │   ├── activation
    │   └── attention
    └── 1/
        ├── activation
        └── attention
```

**VizFold 1.0 Structure:**
```
archive.zarr/
├── metadata/
│   ├── model_version
│   ├── config_version
│   ├── sequence
│   ├── num_residues
│   ├── num_recycles
│   ├── recycle_info
│   ├── residue_index
│   └── representation_names
├── representations/
│   ├── single/
│   │   ├── layer_00
│   │   ├── layer_01
│   │   └── ...
│   └── pair/
│       ├── layer_00
│       ├── layer_01
│       └── ...
├── attention/
│   └── triangle_start/
│       ├── layer_00
│       ├── layer_01
│       └── ...
└── structure/
    ├── atom_positions
    ├── atom_mask
    └── ptm
```

---

## Method Refactors Required

### 1. ✏️ Rename & Update `store_layer_activation()`

**Current Signature:**
```python
def store_layer_activation(path, layer_index, activation_array, overwrite=False)
```

**New Signature:**
```python
def store_single_representation(path, layer_index, representation_array, overwrite=False)
```

**Changes:**
- **Function name:** `store_layer_activation` → `store_single_representation`
- **Path:** `layers/{layer_index}/activation` → `representations/single/layer_{layer_index:02d}`
- **Parameter name:** `activation_array` → `representation_array` (more general)
- **Docstring update:** Explain this stores per-residue representations

**Implementation Note:**
```python
# OLD:
layer_group = root.require_group(f"layers/{layer_index}")
layer_group["activation"] = zarr.array(array)

# NEW:
repr_group = root.require_group("representations/single")
repr_group[f"layer_{layer_index:02d}"] = zarr.array(array)
```

**Why:**
- Aligns with spec's `representations/single/layer_XX` structure
- Per-layer naming (`layer_00`, `layer_01`) matches spec convention
- Clearer intent: "single" vs "pair" representations

---

### 2. ✏️ Refactor `store_attention_heads()` → `store_attention()`

**Current Signature:**
```python
def store_attention_heads(path, layer_index, attention_array, overwrite=False)
```

**New Signature:**
```python
def store_attention(path, attention_type, layer_index, attention_array, overwrite=False)
```

**Changes:**
- **Function name:** `store_attention_heads` → `store_attention`
- **New parameter:** `attention_type` (e.g., "triangle_start", "triangle_end", "pairwise")
- **Path:** `layers/{layer_index}/attention` → `attention/{attention_type}/layer_{layer_index:02d}`
- **Docstring:** Explain different attention types and why they matter

**Implementation Note:**
```python
# OLD:
layer_group = root.require_group(f"layers/{layer_index}")
layer_group["attention"] = zarr.array(array, chunks=chunks)

# NEW:
attn_group = root.require_group(f"attention/{attention_type}")
attn_group[f"layer_{layer_index:02d}"] = zarr.array(array, chunks=chunks)
```

**Why:**
- VizFold spec recognizes multiple attention types (triangle_start, triangle_end, etc.)
- Our current function can't differentiate attention types
- Per-layer, per-type organization enables selective loading

**Usage Examples:**
```python
store_attention(archive, "triangle_start", 0, attn_array)
store_attention(archive, "triangle_end", 0, attn_array)
store_attention(archive, "pairwise", 0, attn_array)
```

---

### 3. ✏️ Update `store_pair_representation()`

**Current Signature:**
```python
def store_pair_representation(path, pair_array, overwrite=False)
```

**New Signature:**
```python
def store_pair_representation(path, layer_index, pair_array, overwrite=False)
```

**Changes:**
- **New parameter:** `layer_index` (required, currently assumed implicit)
- **Path:** `representations/pair` → `representations/pair/layer_{layer_index:02d}`
- **Docstring:** Clarify this is per-layer pair representations

**Implementation Note:**
```python
# OLD:
repr_group = root.require_group("representations")
repr_group["pair"] = zarr.array(array)

# NEW:
repr_group = root.require_group("representations/pair")
repr_group[f"layer_{layer_index:02d}"] = zarr.array(array)
```

**Why:**
- VizFold spec stores pair representations per-layer, not as single array
- Enables incremental addition of layers
- Clearer semantics: which layer's pair representations?

**Usage Examples:**
```python
store_pair_representation(archive, 0, pair_layer_0)
store_pair_representation(archive, 1, pair_layer_1)
# Can add layer 3 later without layer 2
store_pair_representation(archive, 3, pair_layer_3)
```

---

### 4. ✏️ Update `store_structure_coordinates()`

**Changes to paths & field names:**
- **`coordinates` → `atom_positions`** (spec uses "atom_positions")
- **Add optional:** `atom_mask` parameter
- **Add optional:** `ptm` parameter (predicted TM-score/confidence)

**Implementation Note:**
```python
# OLD:
structure_group["coordinates"] = zarr.array(coordinates)
structure_group["residue_types"] = zarr.array(residue_types)

# NEW:
structure_group["atom_positions"] = zarr.array(coordinates)
# Spec also includes:
structure_group["atom_mask"] = zarr.array(atom_mask)  # optional
structure_group["ptm"] = zarr.array(ptm)              # scalar confidence
```

**Why:**
- "atom_positions" is more precise than "coordinates"
- Spec defines optional atom_mask and ptm fields
- Maintains consistency with VizFold terminology

---

### 5. ✏️ Add `store_metadata()` - NEW FUNCTION

**Signature:**
```python
def store_metadata(path, model_version, config_version, sequence, 
                   num_residues, num_recycles, recycle_info=None,
                   residue_index=None, representation_names=None):
```

**Implementation:**
```python
root = zarr.open(path, mode='a')
metadata_group = root.require_group("metadata")

# Store all metadata as scalar or 1D arrays
metadata_group["model_version"] = np.array(model_version, dtype=object)
metadata_group["config_version"] = np.array(config_version, dtype=object)
metadata_group["sequence"] = np.array(sequence, dtype=object)
metadata_group["num_residues"] = np.array(num_residues, dtype=np.int32)
metadata_group["num_recycles"] = np.array(num_recycles, dtype=np.int32)

if recycle_info is not None:
    metadata_group["recycle_info"] = zarr.array(recycle_info)
if residue_index is not None:
    metadata_group["residue_index"] = zarr.array(residue_index)
if representation_names is not None:
    metadata_group["representation_names"] = zarr.array(representation_names, dtype=object)
```

**Why:**
- Metadata is required by VizFold spec
- Enables reproducibility and traceability
- Necessary for archive validation and tooling

---

### 6. ✏️ Update `validate_archive()`

**Changes:**

**Strict Mode (strict=True) must now check:**
- ✅ `metadata/` group exists with required fields
- ✅ `structure/` group with `atom_positions` (new name)
- ✅ `representations/single/` with at least one `layer_XX`
- ✅ `representations/pair/` with at least one `layer_XX`
- ✅ `attention/` group with at least one attention type
- ✅ Layer numbering is sequential (layer_00, layer_01, etc.)

**Lenient Mode (strict=False):**
- ⚠ Warn if metadata missing
- ⚠ Warn if attention types missing
- ✅ Require only structure data (core)

**Implementation Example:**
```python
# Check for new structure
if 'representations' in root:
    repr_group = root['representations']
    if 'single' not in repr_group:
        # Handle error/warning based on strict mode
    if 'pair' in repr_group:
        # Validate layer_00, layer_01 naming
        for layer_key in repr_group['pair'].keys():
            if not layer_key.startswith('layer_'):
                # Invalid naming
```

---

## Part 2: Demo Requirements

### Demo Deliverables

Create a reproducible demonstration showing:

#### 1. **Overwrite Protection Works**
```python
# Scenario: Attempt to write to same layer twice
store_single_representation(archive, 0, representation)
store_single_representation(archive, 0, representation)  # Should raise FileExistsError
# Then with overwrite=True, should succeed
store_single_representation(archive, 0, representation_updated, overwrite=True)
```

**Expected Output:**
```
First attempt: FileExistsError ✓
Second attempt (overwrite=True): Success ✓
```

#### 2. **Validation Modes Work**
```python
# Incomplete archive
store_structure_coordinates(archive, coords)
validate_archive(archive, strict=False)  # Should warn but not fail
# Output: warnings about missing representations

validate_archive(archive, strict=True)   # Should raise ValueError
# Output: Error about required components
```

**Expected Output:**
```
strict=False: Valid=False, warnings=[...] ✓
strict=True: Raises ValueError ✓
```

#### 3. **Pickle Ingestion Traceability**
```python
summary = ingest_output_pkl(archive, "model_output.pkl")
print(summary['key_matches'])
# Shows what keys were searched for, what was matched
```

**Expected Output:**
```
{
    'final_positions': {
        'pattern': ['final', 'atom', 'positions'],
        'matched_key': 'output/final_atom_positions',
        'shape': (128, 37, 3)
    },
    'residue_types': {
        'pattern': ['aatype'],
        'matched_key': 'metadata/aatype',
        'shape': (128,)
    },
    'pair_representation': {
        'pattern': ['pair'],
        'matched_key': None,
        'shape': None
    }
}
```

#### 4. **Per-Layer Organization Works**
```python
# Add multiple layers incrementally
for i in [0, 1, 5]:  # Note: can skip layer 2, 3, 4
    store_single_representation(archive, i, layer_reps[i])
    
# Validate archive still works
result = validate_archive(archive, strict=False)
# Should show layers 0, 1, 5 present, others missing (but that's OK in lenient mode)
```

---

## Part 3: PR Rubric Checklist

### Group Component (20%)

#### Functional Correctness & Completeness (8%)
- [ ] All refactored methods work end-to-end
- [ ] Overwrite parameter functions correctly in all scenarios
- [ ] Validation correctly identifies complete/incomplete archives
- [ ] Pickle ingestion handles edge cases (missing components, wrong shapes)
- [ ] No crashes or unhandled exceptions

#### Output / Demonstration (4%)
- [ ] Create `DEMO.md` with reproducible steps
- [ ] Include screenshots showing:
  - Code changes in editor
  - Test execution output
  - Validation report examples
- [ ] Create process video (5-10 min) demonstrating:
  - Walking through refactored functions
  - Running demo scenarios
  - Explaining design decisions

#### Issue Alignment & Documentation (4%)
- [ ] PR explicitly links to Issue #40 (e.g., "Closes #40")
- [ ] PR description explains:
  - What feedback was addressed (consistency, robustness, traceability)
  - Why each refactor was necessary
  - How new structure aligns with VizFold spec
- [ ] Include clear testing/verification steps in PR

#### Code Quality & Design (4%)
- [ ] Code is clean and readable
- [ ] Function names are clear and descriptive
- [ ] Follows naming conventions (snake_case, descriptive terms)
- [ ] Appropriate use of helper functions
- [ ] Error messages are informative

### Individual Component (25%)

#### Commit Quality & Clarity (10%)
- [ ] 4-6 logical commits (not one giant commit)
- [ ] Each commit has clear, descriptive message
- [ ] Commit messages follow convention:
  - `feat: add new functionality`
  - `refactor: reorganize code structure`
  - `docs: update documentation`
  - `fix: resolve issue with X`
- [ ] Example good commits:
  ```
  refactor: update archive structure to match VizFold 1.0 spec
  
  - Reorganize layers into representations/single/
  - Add per-layer naming (layer_00, layer_01, etc)
  - Update paths for attention types
  
  feat: add metadata storage support
  - Store model_version, config_version, sequence
  - Track num_residues and num_recycles
  - Enable reproducibility and traceability
  
  docs: add comprehensive code comments and docstrings
  - Explain design decisions in store_* functions
  - Add examples of correct usage
  - Document edge cases
  ```

#### Contribution Significance (8%)
- [ ] Demonstrates understanding of feedback
- [ ] Shows problem-solving (refactoring for spec compliance)
- [ ] Meaningful improvements to robustness/safety
- [ ] Clear value addition (not just minor tweaks)

#### Collaboration & Integration (4%)
- [ ] PR description invites feedback
- [ ] Code cleanly integrates (no conflicts)
- [ ] Responsive to code review comments
- [ ] Provides constructive review of teammate's code

#### Documentation & Code Comments (3%)
- [ ] Functions have detailed docstrings explaining:
  - Purpose and semantics
  - Why design decisions were made
  - Edge cases handled
- [ ] Comments explain non-obvious code sections
- [ ] Examples in docstrings show correct usage
- [ ] Update METHOD_SPECIFICATIONS.md with new paths

### Bonus Opportunities (+6%)

#### PR Code Review (up to 3%)
- [ ] Leave 3+ quality reviews on teammate's PRs
- [ ] Reviews include:
  - Specific line references
  - Actionable suggestions
  - Identification of potential bugs/edge cases
  - Compliments on good decisions

#### Process Video Bonus (3%)
- [ ] Screen recording with audio (5-10 minutes)
- [ ] Shows:
  - Running the demo scenarios
  - Walking through code changes
  - Explaining the rationale
  - Demonstrating that everything works
- [ ] Audio is clear and explanatory
- [ ] Reasonable editing allowed (can skip long processing)

---

## Verification Checklist

Before submitting PR, verify:

- [ ] All refactored methods use new VizFold spec paths
- [ ] All method signatures match documentation
- [ ] Overwrite parameter works in all functions
- [ ] Validation modes (strict=True/False) behave correctly
- [ ] Pickle ingestion returns key_matches with full traceability
- [ ] Demo runs without errors (reproducible steps)
- [ ] Code is commented appropriately
- [ ] Commits are logical and well-messaged
- [ ] PR description is comprehensive
- [ ] Issue #40 is linked
- [ ] DEMO.md is included
- [ ] Process video is recorded and included

---

## Files to Update

1. **Archive Utils/outline** - Main implementation
   - Refactor all method signatures and paths
   - Update docstrings and comments

2. **METHOD_SPECIFICATIONS.md** - Update with new paths
   - Replace `layers/{idx}/` with `representations/single/layer_XX`
   - Replace `attention` with `attention/{type}/layer_XX`
   - Add metadata/ section
   - Update validation spec

3. **New: DEMO.md** - Create demo guide
   - Step-by-step reproducible scenarios
   - Expected outputs
   - Screenshots

4. **README.md** (existing project) - Update if needed
   - Link to new refactored utilities
   - Mention VizFold 1.0 compliance

---

## Questions & Clarifications

**Q: Do we need backward compatibility?**  
A: No. This is a comprehensive refactor. Old format is replaced.

**Q: What if we don't have all metadata fields?**  
A: In lenient mode, missing metadata is a warning. In strict mode, it fails. Use lenient during development.

**Q: Can we store layers out of order?**  
A: Yes! The layer_XX naming supports sparse storage (layer_00, layer_03, layer_05).

**Q: Do attention types have to be "triangle_start"?**  
A: Not necessarily. Use what makes sense for your model. Common ones are triangle_start, triangle_end, pairwise. Document your choice.

---

## Success Criteria

The refactor is complete and successful when:

1. ✅ All methods use VizFold 1.0 spec paths
2. ✅ Demo scenarios run without errors
3. ✅ Validation correctly identifies complete/incomplete archives
4. ✅ Code is clean, commented, and well-organized
5. ✅ PR includes demo, video, and comprehensive description
6. ✅ All rubric checklist items are met
