# Vizfold Foundations

This repository has two main components:

1. Model inference & feature extraction: Run protein structure prediction models and extract intermediate activations (hidden representations) and attention maps from any chosen layer.
2. Visualization & analysis: Explore, visualize, and analyze the extracted activations and attention maps.

---

Link to Openfold implimentation - [README_vizfold_openfold.md](https://github.com/vizfold/vizfold-foundation/blob/main/README_vizfold_openfold.md)

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
See the [LICENSE](./LICENSE) file for details.

---

## ✨ NEW: Intermediate Representation Visualization & Dimensionality Reduction

This fork extends OpenFold visualization with tools for analyzing intermediate representations from OpenFold's 48-layer network.

### Features:

**Intermediate Representation Extraction** (by Jayanth):
- Extract MSA, Pair, and Single representations from any layer
- Heatmap visualizations
- Residue feature analysis
- Layer convergence analysis
- Stratified layer sampling

**Advanced Dimensionality Reduction** (by Shreyas):
- **t-SNE**: Reveal clusters and local structure
- **PCA**: Understand main variance directions  
- **UMAP**: Balance local and global structure
- **Autoencoder**: Learn non-linear representations
- **Layer Progression**: Visualize representation evolution across layers
- **Method Comparison**: Side-by-side analysis of all techniques

### Quick Start:

```bash
# Run complete dimensionality reduction analysis
python demo_tsne_reduction.py \
    --input_reps demo_outputs/demo_protein_intermediate_reps.pt \
    --output_dir outputs/tsne_analysis \
    --demo all
```

### Documentation:

- **Intermediate Representations**: See `INTERMEDIATE_REPS_README.md` (Jayanth's work)

### Additional Dependencies:

```bash
pip install scikit-learn umap-learn  # For dimensionality reduction
```

**Related Issue**: [Issue #8 - Intermediate Representations](https://github.com/vizfold/attention-viz-demo/issues/8)  
**Pull Request**: #11

---
