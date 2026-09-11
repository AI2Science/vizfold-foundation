# Dimensionality Reduction for OpenFold Representations

**Quick guide for t-SNE, PCA, UMAP, and autoencoder visualization of intermediate representations.**

---

## Quick Start

```bash
# Install dependencies
pip install scikit-learn umap-learn

# Visualize from pickle file
python generate_tsne_from_pickle.py your_protein_reps.pt

# Full analysis
python demo_tsne_reduction.py --demo all --input_reps your_protein.pt
```

---

## What It Does

**Problem**: OpenFold representations are 256-dimensional (impossible to visualize)  
**Solution**: Compress to 2D using dimensionality reduction  
**Result**: See clusters, patterns, and layer evolution

---

## Methods

| Method | Speed | Best For | Training Needed |
|--------|-------|----------|-----------------|
| **PCA** | <1s | Quick overview, variance analysis | No |
| **t-SNE** | 30s-2min | Finding clusters, local structure | No |
| **UMAP** | 5-15s | General visualization | No |
| **Autoencoder** | 2-5min | Task-specific features | Yes (tiny model) |

---

## Files

- `visualize_dimensionality_reduction.py` - Core module (PCA, t-SNE, UMAP, autoencoder)
- `generate_tsne_from_pickle.py` - Simple script: `python generate_tsne_from_pickle.py file.pt`
- `demo_tsne_reduction.py` - Full demo with 3 modes
- `visualize_from_alphafold_pickle.py` - Works with AlphaFold pickle format
- `tests/test_dimensionality_reduction.py` - 23 unit tests

---

## Usage Examples

### Basic Visualization
```python
from visualize_dimensionality_reduction import *
from visualize_intermediate_reps_utils import load_intermediate_reps_from_disk

# Load data
reps = load_intermediate_reps_from_disk('protein_reps.pt')

# Prepare and reduce
data = prepare_representations_for_reduction(reps['msa'][47], 'residue')
tsne_result = apply_tsne(data, n_components=2)

# Visualize
plot_2d_embedding(tsne_result, title='t-SNE Layer 47', save_path='output.png')
```

### Layer Progression
```python
# Compare multiple layers
results = visualize_layer_progression(
    {0: reps['msa'][0], 23: reps['msa'][23], 47: reps['msa'][47]},
    method='tsne',
    save_dir='outputs/'
)
```

### Method Comparison
```python
# Compare all methods side-by-side
compare_reduction_methods(
    data,
    methods=['pca', 'tsne', 'umap'],
    save_dir='outputs/comparison'
)
```

---

## Data Preparation Modes

- `flatten_mode='residue'` - Each residue = 1 point (for MSA/Single)
- `flatten_mode='global'` - Aggregate to 1 point per protein/layer
- `flatten_mode='pairwise'` - Each residue-residue pair = 1 point (for Pair)

---

## What Outputs Show

**Each dot** = 1 residue in your protein  
**Nearby dots** = Similar representations/features  
**Clusters** = Groups of residues (e.g., alpha helices, beta sheets)  
**Separate groups** = Different structural/functional regions  
**Layer progression** = How representations evolve (scattered → organized)

---

## Integration with Jayanth's Work

```python
# Jayanth's extraction
from visualize_intermediate_reps_utils import load_intermediate_reps_from_disk
reps = load_intermediate_reps_from_disk('reps.pt')

# Your dimensionality reduction
from visualize_dimensionality_reduction import run_complete_dimensionality_reduction_analysis
results = run_complete_dimensionality_reduction_analysis(
    representations=reps['msa'],
    output_dir='outputs/',
    methods=['pca', 'tsne', 'umap']
)
```

---

## Testing

```bash
python -m pytest tests/test_dimensionality_reduction.py -v
```

**23 tests** covering all functionality.

---

## Troubleshooting

- **"scikit-learn not available"** → `pip install scikit-learn`
- **"UMAP not available"** → `pip install umap-learn` (optional)
- **t-SNE slow** → Use PCA first, or switch to UMAP
- **Out of memory** → Use `flatten_mode='global'` or analyze fewer layers

---

## PR Information

**Built on**: Jayanth's PR #11 (intermediate representation extraction)  
**Adds**: Dimensionality reduction methods for visualization  
**Addresses**: Issue #8 (Intermediate Representation Visualization)  
**Files**: ~2700 lines of code + tests

---

**That's it! Everything you need in one place.** 🚀

