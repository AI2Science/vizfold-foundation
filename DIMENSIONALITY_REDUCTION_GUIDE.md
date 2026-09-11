# Dimensionality Reduction for OpenFold Representations

Tools for visualizing 256-dimensional intermediate representations using PCA, t-SNE, UMAP, and autoencoders.

## Installation

```bash
pip install scikit-learn umap-learn
```

## Quick Start

```bash
# Basic visualization from pickle file
python generate_tsne_from_pickle.py your_protein_reps.pt

# Full analysis pipeline
python demo_tsne_reduction.py --demo all --input_reps your_protein.pt
```

## Methods

| Method | Speed | Use Case | Training |
|--------|-------|----------|----------|
| PCA | <1s | Variance analysis, quick overview | No |
| t-SNE | 30s-2min | Cluster detection, local structure | No |
| UMAP | 5-15s | General visualization | No |
| Autoencoder | 2-5min | Learned representations | Yes |

## Files

- `visualize_dimensionality_reduction.py` - Core module
- `generate_tsne_from_pickle.py` - Quick visualization script
- `demo_tsne_reduction.py` - Demo with single layer, progression, and full analysis modes
- `visualize_from_alphafold_pickle.py` - AlphaFold pickle format support
- `tests/test_dimensionality_reduction.py` - Unit tests

## Usage

### Basic Visualization

```python
from visualize_dimensionality_reduction import *
from visualize_intermediate_reps_utils import load_intermediate_reps_from_disk

reps = load_intermediate_reps_from_disk('protein_reps.pt')
data = prepare_representations_for_reduction(reps['msa'][47], 'residue')
tsne_result = apply_tsne(data, n_components=2)
plot_2d_embedding(tsne_result, title='t-SNE Layer 47', save_path='output.png')
```

### Layer Progression

```python
results = visualize_layer_progression(
    {0: reps['msa'][0], 23: reps['msa'][23], 47: reps['msa'][47]},
    method='tsne',
    save_dir='outputs/'
)
```

### Method Comparison

```python
compare_reduction_methods(
    data,
    methods=['pca', 'tsne', 'umap'],
    save_dir='outputs/comparison'
)
```

## Data Preparation

Three flatten modes:

- `flatten_mode='residue'` - One point per residue (MSA/Single)
- `flatten_mode='global'` - One point per protein/layer
- `flatten_mode='pairwise'` - One point per residue pair (Pair)

## Output Interpretation

- Each point = one residue
- Nearby points = similar representations
- Clusters = residue groups (e.g., secondary structure)
- Separate groups = different structural regions
- Layer progression = representation evolution over layers

## Integration

Works with intermediate representations from `visualize_intermediate_reps_utils.py`:

```python
from visualize_intermediate_reps_utils import load_intermediate_reps_from_disk
from visualize_dimensionality_reduction import run_complete_dimensionality_reduction_analysis

reps = load_intermediate_reps_from_disk('reps.pt')
results = run_complete_dimensionality_reduction_analysis(
    representations=reps['msa'],
    output_dir='outputs/',
    methods=['pca', 'tsne', 'umap']
)
```

## Testing

```bash
python -m pytest tests/test_dimensionality_reduction.py -v
```

## Troubleshooting

- Missing scikit-learn: `pip install scikit-learn`
- Missing UMAP: `pip install umap-learn` (optional)
- t-SNE too slow: Try PCA first, or use UMAP
- Out of memory: Use `flatten_mode='global'` or reduce number of layers

## Notes

Built on PR #11 (intermediate representation extraction).  
Addresses Issue #8 (Intermediate Representation Visualization).
