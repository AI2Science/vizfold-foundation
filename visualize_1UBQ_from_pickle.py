"""
Visualize 1UBQ representations from AlphaFold pickle file.

Note: AlphaFold pickle files contain FINAL layer representations only,
not intermediate layers from all 48 Evoformer blocks. To get intermediate
layers, you need to run OpenFold inference with hooks enabled.

However, we can still visualize the final representations with t-SNE
to demonstrate the methodology.

Usage:
    python visualize_1UBQ_from_pickle.py [--pickle_file PATH] [--output_dir DIR]
"""

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import dimensionality reduction utilities
try:
    from visualize_dimensionality_reduction import (
        prepare_representations_for_reduction,
        apply_tsne,
        apply_pca,
        plot_2d_embedding,
        SKLEARN_AVAILABLE
    )
except ImportError as e:
    print(f"Error importing utilities: {e}")
    sys.exit(1)


def visualize_1UBQ_from_pickle(pickle_file=None, output_dir='outputs/1UBQ_from_pickle'):
    """
    Visualize 1UBQ representations from AlphaFold pickle file.
    
    Args:
        pickle_file: Path to AlphaFold pickle file
        output_dir: Output directory for visualizations
    """
    
    print("\n" + "="*70)
    print("Visualizing 1UBQ from AlphaFold Pickle File")
    print("="*70)
    
    if not SKLEARN_AVAILABLE:
        print("\n✗ Error: scikit-learn required for t-SNE")
        print("Install with: pip install scikit-learn")
        return
    
    # Find pickle file if not provided
    if pickle_file is None:
        pickle_file = 'proteins_pickle/1UBQ_7686d_all_rank_004_alphafold2_ptm_model_1_seed_000.pickle'
    
    if not os.path.exists(pickle_file):
        print(f"\n✗ Pickle file not found: {pickle_file}")
        return
    
    # Load pickle file
    print(f"\n[Step 1] Loading pickle file: {pickle_file}")
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        print("  ✓ Loaded successfully!")
    except Exception as e:
        print(f"  ✗ Error loading pickle: {e}")
        return
    
    # Check available representations
    print("\n[Step 2] Checking available representations...")
    if 'representations' not in data:
        print("  ✗ No 'representations' key found in pickle file")
        return
    
    reps = data['representations']
    print(f"  Available representations: {list(reps.keys())}")
    
    # Important note
    print("\n" + "="*70)
    print("⚠  IMPORTANT NOTE:")
    print("="*70)
    print("AlphaFold pickle files contain FINAL layer representations only,")
    print("not intermediate layers from all 48 Evoformer blocks.")
    print("\nTo visualize intermediate layers (0, 23, 47, etc.), you need to:")
    print("  1. Run OpenFold inference with hooks enabled")
    print("  2. Extract intermediate representations from all 48 layers")
    print("  3. Save them using save_intermediate_reps_to_disk()")
    print("\nHowever, we can still visualize the FINAL layer representations")
    print("to demonstrate how t-SNE works with protein representations.")
    print("="*70 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize available representations
    if 'single' in reps:
        print("\n[Step 3] Visualizing SINGLE representation (final layer)...")
        single_rep = reps['single']  # Shape: (76, 256)
        
        # Convert to numpy if needed
        if hasattr(single_rep, 'numpy'):
            single_rep = single_rep.numpy()
        elif hasattr(single_rep, 'detach'):
            single_rep = single_rep.detach().cpu().numpy()
        single_rep = np.array(single_rep).astype(np.float32)
        
        print(f"  Single representation shape: {single_rep.shape}")
        
        # Prepare data (already flattened per residue)
        data_ready = single_rep  # Already (76 residues, 256 features)
        
        # Apply PCA
        print("\n[Step 4] Applying PCA...")
        embedding_pca, pca_model = apply_pca(data_ready, n_components=2)
        print(f"  PCA embedding shape: {embedding_pca.shape}")
        print(f"  Variance explained: {pca_model.explained_variance_ratio_}")
        print(f"  Total variance: {pca_model.explained_variance_ratio_.sum():.3f}")
        
        # Apply t-SNE
        print("\n[Step 5] Applying t-SNE (this may take a minute)...")
        perplexity = min(30, data_ready.shape[0] // 4)
        embedding_tsne = apply_tsne(data_ready, n_components=2, perplexity=perplexity)
        print(f"  t-SNE embedding shape: {embedding_tsne.shape}")
        
        # Create visualizations
        print("\n[Step 6] Creating visualizations...")
        
        # PCA plot
        pca_path = os.path.join(output_dir, '1UBQ_single_PCA.png')
        plot_2d_embedding(
            embedding_pca,
            title="1UBQ Single Representation - PCA (Final Layer)",
            save_path=pca_path,
            labels=None
        )
        print(f"  ✓ Saved PCA plot: {pca_path}")
        
        # t-SNE plot
        tsne_path = os.path.join(output_dir, '1UBQ_single_tSNE.png')
        plot_2d_embedding(
            embedding_tsne,
            title="1UBQ Single Representation - t-SNE (Final Layer)",
            save_path=tsne_path,
            labels=None
        )
        print(f"  ✓ Saved t-SNE plot: {tsne_path}")
        
        # Side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # PCA
        ax1.scatter(embedding_pca[:, 0], embedding_pca[:, 1], alpha=0.7, s=50)
        ax1.set_title('PCA (Final Layer)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('PC1', fontsize=12)
        ax1.set_ylabel('PC2', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # t-SNE
        ax2.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], alpha=0.7, s=50)
        ax2.set_title('t-SNE (Final Layer)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE 1', fontsize=12)
        ax2.set_ylabel('t-SNE 2', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('1UBQ Single Representation Comparison (Final Layer Only)', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        comparison_path = os.path.join(output_dir, '1UBQ_single_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved comparison plot: {comparison_path}")
    
    if 'pair' in reps:
        print("\n[Step 7] Visualizing PAIR representation (final layer)...")
        pair_rep = reps['pair']  # Shape: (76, 76, 128)
        
        # Convert to numpy
        if hasattr(pair_rep, 'numpy'):
            pair_rep = pair_rep.numpy()
        elif hasattr(pair_rep, 'detach'):
            pair_rep = pair_rep.detach().cpu().numpy()
        pair_rep = np.array(pair_rep).astype(np.float32)
        
        print(f"  Pair representation shape: {pair_rep.shape}")
        
        # For pair representation, we can extract residue-level features
        # by averaging over pairs
        residue_features = pair_rep.mean(axis=1)  # Average over second residue dimension
        print(f"  Residue features shape (averaged): {residue_features.shape}")
        
        # Apply t-SNE
        perplexity = min(30, residue_features.shape[0] // 4)
        embedding_pair = apply_tsne(residue_features, n_components=2, perplexity=perplexity)
        
        # Visualize
        pair_path = os.path.join(output_dir, '1UBQ_pair_tSNE.png')
        plot_2d_embedding(
            embedding_pair,
            title="1UBQ Pair Representation - t-SNE (Final Layer, Averaged)",
            save_path=pair_path,
            labels=None
        )
        print(f"  ✓ Saved pair t-SNE plot: {pair_path}")
    
    print("\n" + "="*70)
    print("✓ Complete! Generated visualizations from FINAL layer only.")
    print(f"\nOutput directory: {output_dir}")
    print("\nTo visualize INTERMEDIATE layers (0, 23, 47, etc.), you need:")
    print("  - Run OpenFold inference with hooks to capture all 48 layers")
    print("  - Use generate_tsne_1UBQ_simple.py with intermediate .pt file")
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 1UBQ from AlphaFold pickle')
    parser.add_argument('--pickle_file', type=str, default=None,
                       help='Path to AlphaFold pickle file')
    parser.add_argument('--output_dir', type=str, default='outputs/1UBQ_from_pickle',
                       help='Output directory')
    
    args = parser.parse_args()
    
    visualize_1UBQ_from_pickle(
        pickle_file=args.pickle_file,
        output_dir=args.output_dir
    )


