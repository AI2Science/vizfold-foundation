"""
Generate t-SNE visualization from a pickle file containing intermediate representations.

Usage:
    python generate_tsne_from_pickle.py <pickle_file> [--output_dir OUTPUT_DIR] [--layer LAYER]

Example:
    python generate_tsne_from_pickle.py demo_outputs/demo_protein_intermediate_reps.pt --layer 47
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import Jayanth's utilities
from visualize_intermediate_reps_utils import load_intermediate_reps_from_disk

# Import your dimensionality reduction utilities
from visualize_dimensionality_reduction import (
    prepare_representations_for_reduction,
    apply_tsne,
    apply_pca,
    apply_umap,
    plot_2d_embedding,
    compare_reduction_methods,
    SKLEARN_AVAILABLE,
    UMAP_AVAILABLE
)


def generate_tsne_visualization(pickle_file, output_dir='outputs/tsne_from_pickle', 
                                layer_idx=None, methods=['pca', 'tsne', 'umap']):
    """
    Generate t-SNE and other dimensionality reduction visualizations from a pickle file.
    
    Args:
        pickle_file: Path to .pt pickle file with intermediate representations
        output_dir: Directory to save outputs
        layer_idx: Specific layer to visualize (None = use last layer)
        methods: List of methods to apply ('pca', 'tsne', 'umap')
    """
    
    print("\n" + "="*70)
    print("Generating t-SNE Visualization from Pickle File")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Step 1: Load the pickle file
    print(f"\n[Step 1] Loading pickle file: {pickle_file}")
    try:
        reps = load_intermediate_reps_from_disk(pickle_file)
        print(f"✓ Loaded successfully!")
        print(f"  Available representations:")
        for rep_type in ['msa', 'pair', 'single']:
            if rep_type in reps and reps[rep_type]:
                print(f"    - {rep_type}: {len(reps[rep_type])} layers")
    except Exception as e:
        print(f"✗ Error loading pickle file: {e}")
        return
    
    # Step 2: Select layer
    if 'msa' not in reps or not reps['msa']:
        print("\n✗ No MSA representations found in pickle file")
        return
    
    available_layers = sorted(reps['msa'].keys())
    if layer_idx is None:
        layer_idx = available_layers[-1]  # Use last layer
    
    if layer_idx not in available_layers:
        print(f"\n✗ Layer {layer_idx} not found. Available layers: {available_layers}")
        return
    
    print(f"\n[Step 2] Using layer {layer_idx}")
    msa_rep = reps['msa'][layer_idx]
    print(f"  MSA representation shape: {msa_rep.shape}")
    
    # Step 3: Prepare data
    print(f"\n[Step 3] Preparing data for dimensionality reduction...")
    data = prepare_representations_for_reduction(msa_rep, flatten_mode='residue')
    print(f"  Prepared data shape: {data.shape}")
    print(f"  {data.shape[0]} residues × {data.shape[1]} features")
    
    # Step 4: Apply dimensionality reduction methods
    print(f"\n[Step 4] Applying dimensionality reduction methods...")
    results = {}
    
    # Filter methods based on availability
    available_methods = []
    if 'pca' in methods and SKLEARN_AVAILABLE:
        available_methods.append('pca')
    if 'tsne' in methods and SKLEARN_AVAILABLE:
        available_methods.append('tsne')
    if 'umap' in methods and UMAP_AVAILABLE:
        available_methods.append('umap')
    
    if not available_methods:
        print("\n✗ No methods available. Install: pip install scikit-learn umap-learn")
        return
    
    print(f"  Methods to apply: {', '.join(available_methods)}")
    
    # Apply each method
    for method in available_methods:
        print(f"\n  Applying {method.upper()}...")
        try:
            if method == 'pca':
                from visualize_dimensionality_reduction import apply_pca
                reduced, pca_model = apply_pca(data, n_components=2)
                print(f"    ✓ PCA complete - explained variance: {pca_model.explained_variance_ratio_.sum():.2%}")
                results[method] = reduced
                
            elif method == 'tsne':
                from visualize_dimensionality_reduction import apply_tsne
                reduced = apply_tsne(data, n_components=2, perplexity=30, n_iter=1000)
                print(f"    ✓ t-SNE complete")
                results[method] = reduced
                
            elif method == 'umap':
                from visualize_dimensionality_reduction import apply_umap
                reduced = apply_umap(data, n_components=2, n_neighbors=15)
                print(f"    ✓ UMAP complete")
                results[method] = reduced
                
        except Exception as e:
            print(f"    ✗ {method.upper()} failed: {e}")
    
    # Step 5: Generate visualizations
    print(f"\n[Step 5] Generating visualizations...")
    
    for method, reduced in results.items():
        # Individual plot
        save_path = os.path.join(output_dir, f'{method}_layer{layer_idx}.png')
        plot_2d_embedding(
            reduced,
            title=f'{method.upper()}: MSA Representations (Layer {layer_idx})',
            save_path=save_path
        )
        print(f"  ✓ Saved {method.upper()} plot: {save_path}")
    
    # Comparison plot if multiple methods
    if len(results) > 1:
        comparison_path = os.path.join(output_dir, f'comparison_layer{layer_idx}.png')
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        for ax, (method, reduced) in zip(axes, results.items()):
            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=30)
            ax.set_title(f'{method.upper()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved comparison plot: {comparison_path}")
    
    # Step 6: Save numerical results
    results_file = os.path.join(output_dir, f'reduced_representations_layer{layer_idx}.npz')
    np.savez(results_file, **results)
    print(f"\n[Step 6] Saved numerical results: {results_file}")
    
    # Summary
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(results)} visualizations:")
    for method in results.keys():
        print(f"  - {method.upper()} 2D embedding")
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFiles created:")
    print(f"  - Individual plots: {', '.join([f'{m}_layer{layer_idx}.png' for m in results.keys()])}")
    if len(results) > 1:
        print(f"  - Comparison plot: comparison_layer{layer_idx}.png")
    print(f"  - Numerical data: reduced_representations_layer{layer_idx}.npz")
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Generate t-SNE visualization from intermediate representation pickle file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use last layer (default)
  python generate_tsne_from_pickle.py demo_outputs/demo_protein_intermediate_reps.pt
  
  # Specify layer
  python generate_tsne_from_pickle.py demo_outputs/demo_protein_intermediate_reps.pt --layer 47
  
  # Specify output directory
  python generate_tsne_from_pickle.py my_protein_reps.pt --output_dir my_results/
  
  # Use only specific methods
  python generate_tsne_from_pickle.py my_protein_reps.pt --methods pca tsne
        """
    )
    
    parser.add_argument(
        'pickle_file',
        type=str,
        help='Path to pickle file (.pt) with intermediate representations'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/tsne_from_pickle',
        help='Directory to save output visualizations (default: outputs/tsne_from_pickle)'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=None,
        help='Layer index to visualize (default: last layer)'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['pca', 'tsne', 'umap'],
        default=['pca', 'tsne', 'umap'],
        help='Dimensionality reduction methods to apply (default: all)'
    )
    
    args = parser.parse_args()
    
    # Check if pickle file exists
    if not os.path.exists(args.pickle_file):
        print(f"Error: Pickle file not found: {args.pickle_file}")
        sys.exit(1)
    
    # Generate visualizations
    generate_tsne_visualization(
        args.pickle_file,
        output_dir=args.output_dir,
        layer_idx=args.layer,
        methods=args.methods
    )


if __name__ == '__main__':
    main()

