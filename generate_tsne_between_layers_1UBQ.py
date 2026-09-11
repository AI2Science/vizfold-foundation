"""
Generate t-SNE visualization showing how 1UBQ representations evolve between layers.

This script loads intermediate representations for 1UBQ and applies t-SNE to multiple
layers to visualize how representations change from early to late layers.

Usage:
    python generate_tsne_between_layers_1UBQ.py [--pickle_file PATH] [--layers 0,23,47] [--output_dir DIR]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import utilities
from visualize_intermediate_reps_utils import load_intermediate_reps_from_disk
from visualize_dimensionality_reduction import (
    prepare_representations_for_reduction,
    apply_tsne,
    apply_pca,
    visualize_layer_progression,
    SKLEARN_AVAILABLE
)


def generate_tsne_between_layers(pickle_file=None, output_dir='outputs/1UBQ_tsne_layers', 
                                 layers=[0, 23, 47], rep_type='msa'):
    """
    Generate t-SNE visualization comparing multiple layers for 1UBQ.
    
    Args:
        pickle_file: Path to .pt file (if None, searches for 1UBQ files)
        output_dir: Directory to save outputs
        layers: List of layer indices to compare
        rep_type: Type of representation ('msa', 'pair', 'single')
    """
    
    print("\n" + "="*70)
    print("Generating t-SNE Between Layers for 1UBQ (Ubiquitin)")
    print("="*70)
    
    if not SKLEARN_AVAILABLE:
        print("\n✗ Error: scikit-learn required for t-SNE")
        print("Install with: pip install scikit-learn")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Find pickle file if not provided
    if pickle_file is None:
        # Look for 1UBQ pickle files
        possible_paths = [
            'proteins_pickle/1UBQ_7686d_all_rank_004_alphafold2_ptm_model_1_seed_000.pickle',
            'outputs/*1UBQ*.pt',
            '*.pt'
        ]
        
        print("\n[Step 1] Searching for 1UBQ intermediate representation file...")
        pickle_file = None
        
        # Try to find the pickle file
        if os.path.exists('proteins_pickle/1UBQ_7686d_all_rank_004_alphafold2_ptm_model_1_seed_000.pickle'):
            print("  Found AlphaFold pickle file - need to extract representations first")
            print("  Please run OpenFold inference to generate intermediate representations")
            print("  Or provide path to .pt file with --pickle_file argument")
            return
        
        # Try to find .pt file
        import glob
        pt_files = glob.glob('**/*1UBQ*.pt', recursive=True)
        if pt_files:
            pickle_file = pt_files[0]
            print(f"  Found: {pickle_file}")
        else:
            print("  ✗ No 1UBQ .pt file found")
            print("  Please provide path with --pickle_file argument")
            return
    
    # Step 1: Load the pickle file
    print(f"\n[Step 1] Loading pickle file: {pickle_file}")
    try:
        reps = load_intermediate_reps_from_disk(pickle_file)
        print(f"✓ Loaded successfully!")
        
        # Check available representation types
        available_reps = {k: len(v) if isinstance(v, dict) else 1 
                         for k, v in reps.items() if v is not None}
        print(f"  Available representations:")
        for rep_type_name, num_layers in available_reps.items():
            print(f"    - {rep_type_name}: {num_layers} layers")
            
    except Exception as e:
        print(f"✗ Error loading pickle file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Select representation type and layers
    if rep_type not in reps or not reps[rep_type]:
        print(f"\n✗ No {rep_type} representations found")
        print(f"  Available types: {list(reps.keys())}")
        # Try to use first available type
        rep_type = list(reps.keys())[0]
        print(f"  Using {rep_type} instead")
    
    available_layers = sorted(reps[rep_type].keys())
    print(f"\n[Step 2] Using {rep_type.upper()} representations")
    print(f"  Available layers: {min(available_layers)} to {max(available_layers)}")
    
    # Filter to requested layers that exist
    layers_to_use = [l for l in layers if l in available_layers]
    if not layers_to_use:
        print(f"  ✗ None of requested layers {layers} found")
        print(f"  Using layers 0, {len(available_layers)//2}, {max(available_layers)} instead")
        layers_to_use = [0, len(available_layers)//2, max(available_layers)]
        layers_to_use = [l for l in layers_to_use if l in available_layers]
    
    print(f"  Visualizing layers: {layers_to_use}")
    
    # Extract representations for selected layers
    layer_reps = {layer_idx: reps[rep_type][layer_idx] for layer_idx in layers_to_use}
    
    # Step 3: Prepare data for each layer
    print(f"\n[Step 3] Preparing data for each layer...")
    layer_data = {}
    for layer_idx, rep in layer_reps.items():
        data = prepare_representations_for_reduction(rep, flatten_mode='residue')
        layer_data[layer_idx] = data
        print(f"  Layer {layer_idx}: {data.shape} ({data.shape[0]} residues × {data.shape[1]} features)")
    
    # Step 4: Apply t-SNE to each layer separately
    print(f"\n[Step 4] Applying t-SNE to each layer...")
    embeddings = {}
    
    # Combine all layers for consistent t-SNE embedding space
    print("  Combining all layers for joint t-SNE (preserves relationships)...")
    all_data = np.vstack([layer_data[l] for l in layers_to_use])
    all_labels = np.concatenate([[l] * len(layer_data[l]) for l in layers_to_use])
    
    print(f"  Combined shape: {all_data.shape}")
    print(f"  Computing joint t-SNE embedding...")
    
    # Apply t-SNE to combined data
    combined_embedding = apply_tsne(all_data, n_components=2, perplexity=min(30, all_data.shape[0]//4))
    
    # Split back into per-layer embeddings
    start_idx = 0
    for layer_idx in layers_to_use:
        end_idx = start_idx + len(layer_data[layer_idx])
        embeddings[layer_idx] = combined_embedding[start_idx:end_idx]
        start_idx = end_idx
        print(f"  Layer {layer_idx}: {embeddings[layer_idx].shape}")
    
    # Step 5: Create visualization
    print(f"\n[Step 5] Creating visualization...")
    
    # Option 1: Side-by-side comparison
    n_layers = len(layers_to_use)
    fig, axes = plt.subplots(1, n_layers, figsize=(6*n_layers, 6))
    if n_layers == 1:
        axes = [axes]
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
    
    for idx, (layer_idx, embedding) in enumerate(embeddings.items()):
        ax = axes[idx]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           alpha=0.7, s=50, c=[idx], cmap='viridis')
        ax.set_title(f'Layer {layer_idx}', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'1UBQ {rep_type.upper()} Representation Evolution (t-SNE)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    side_by_side_path = os.path.join(output_dir, '1UBQ_tsne_layers_comparison.png')
    plt.savefig(side_by_side_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved side-by-side comparison: {side_by_side_path}")
    plt.close()
    
    # Option 2: Overlay all layers with different colors
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for idx, (layer_idx, embedding) in enumerate(embeddings.items()):
        ax.scatter(embedding[:, 0], embedding[:, 1], 
                  alpha=0.6, s=50, label=f'Layer {layer_idx}',
                  c=[colors[idx]])
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title('1UBQ Layer Evolution (All Layers Overlay)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    overlay_path = os.path.join(output_dir, '1UBQ_tsne_layers_overlay.png')
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved overlay plot: {overlay_path}")
    plt.close()
    
    # Option 3: Use the built-in layer progression function
    print(f"\n[Step 6] Creating layer progression visualization...")
    try:
        progression_results = visualize_layer_progression(
            layer_reps,
            method='tsne',
            n_components=2,
            save_dir=output_dir,
            rep_type=rep_type,
            flatten_mode='residue'
        )
        print(f"  ✓ Saved layer progression: {output_dir}/{rep_type}_evolution_tsne.png")
    except Exception as e:
        print(f"  ⚠ Layer progression function error: {e}")
    
    print(f"\n{'='*70}")
    print("✓ Complete! Generated t-SNE visualizations:")
    print(f"  1. Side-by-side comparison: {side_by_side_path}")
    print(f"  2. Overlay plot: {overlay_path}")
    print(f"  3. Layer progression: {output_dir}/{rep_type}_evolution_tsne.png")
    print(f"\nOutput directory: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate t-SNE between layers for 1UBQ')
    parser.add_argument('--pickle_file', type=str, default=None,
                       help='Path to .pt file with intermediate representations')
    parser.add_argument('--layers', type=str, default='0,23,47',
                       help='Comma-separated list of layer indices (e.g., "0,23,47")')
    parser.add_argument('--output_dir', type=str, default='outputs/1UBQ_tsne_layers',
                       help='Output directory for visualizations')
    parser.add_argument('--rep_type', type=str, default='msa', choices=['msa', 'pair', 'single'],
                       help='Type of representation to visualize')
    
    args = parser.parse_args()
    
    # Parse layers
    layers = [int(l.strip()) for l in args.layers.split(',')]
    
    generate_tsne_between_layers(
        pickle_file=args.pickle_file,
        output_dir=args.output_dir,
        layers=layers,
        rep_type=args.rep_type
    )


