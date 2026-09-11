"""
Generate t-SNE visualization between layers for 1UBQ.

This script will:
1. Try to load intermediate representations (.pt file)
2. If not found, guide you to extract them first
3. Generate t-SNE visualizations comparing layers 0, 23, and 47

To get intermediate representations, you need to run OpenFold inference with hooks enabled.
See: visualize_intermediate_reps_utils.py for extraction methods.

Usage:
    python generate_tsne_1UBQ_simple.py [--pickle_file PATH]
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import utilities
try:
    from visualize_intermediate_reps_utils import load_intermediate_reps_from_disk
    from visualize_dimensionality_reduction import (
        prepare_representations_for_reduction,
        apply_tsne,
        visualize_layer_progression,
        SKLEARN_AVAILABLE
    )
except ImportError as e:
    print(f"Error importing utilities: {e}")
    print("Make sure you're in the correct directory")
    sys.exit(1)


def find_1UBQ_files():
    """Find 1UBQ-related files."""
    files = {}
    
    # Look for .pt files (intermediate representations)
    pt_files = glob.glob('**/*1UBQ*.pt', recursive=True)
    if pt_files:
        files['pt'] = pt_files[0]
    
    # Look for AlphaFold pickle files
    pickle_files = glob.glob('**/*1UBQ*.pickle', recursive=True)
    if pickle_files:
        files['pickle'] = pickle_files[0]
    
    return files


def generate_tsne_1UBQ(pickle_file=None, layers=[0, 23, 47], output_dir='outputs/1UBQ_tsne'):
    """Generate t-SNE visualization for 1UBQ."""
    
    print("\n" + "="*70)
    print("Generating t-SNE Between Layers for 1UBQ (Ubiquitin)")
    print("="*70)
    
    if not SKLEARN_AVAILABLE:
        print("\n✗ Error: scikit-learn required for t-SNE")
        print("Install with: pip install scikit-learn")
        return
    
    # Find files if not provided
    if pickle_file is None:
        print("\n[Step 1] Searching for 1UBQ files...")
        files = find_1UBQ_files()
        
        if 'pt' in files:
            pickle_file = files['pt']
            print(f"  ✓ Found .pt file: {pickle_file}")
        elif 'pickle' in files:
            print(f"  Found AlphaFold pickle: {files['pickle']}")
            print("  ⚠ AlphaFold pickle files don't contain intermediate layer representations")
            print("  You need to run OpenFold inference with hooks to extract intermediate representations")
            print("\n  To extract intermediate representations:")
            print("  1. Run OpenFold inference with hooks enabled:")
            print("     from visualize_intermediate_reps_utils import *")
            print("     INTERMEDIATE_REPS.enable()")
            print("     hooks = register_evoformer_hooks(model)")
            print("     output = model(batch)")
            print("  2. Save representations:")
            print("     save_intermediate_reps_to_disk({...}, output_dir, '1UBQ')")
            return
        else:
            print("  ✗ No 1UBQ files found")
            print("\n  To generate intermediate representations:")
            print("  1. Run OpenFold inference with Jayanth's hooks enabled")
            print("  2. Save intermediate representations to a .pt file")
            return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load representations
    print(f"\n[Step 2] Loading intermediate representations: {pickle_file}")
    try:
        reps = load_intermediate_reps_from_disk(pickle_file)
        print("  ✓ Loaded successfully!")
        
        # Check what's available
        if 'msa' in reps and reps['msa']:
            available_layers = sorted(reps['msa'].keys())
            print(f"  MSA layers available: {min(available_layers)} to {max(available_layers)}")
        else:
            print("  ✗ No MSA representations found")
            return
            
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Select layers to visualize
    available_layers = sorted(reps['msa'].keys())
    layers_to_use = [l for l in layers if l in available_layers]
    
    if not layers_to_use:
        print(f"\n  ⚠ Requested layers {layers} not found")
        print(f"  Available layers: {min(available_layers)} to {max(available_layers)}")
        # Use first, middle, last
        mid_layer = available_layers[len(available_layers)//2]
        layers_to_use = [available_layers[0], mid_layer, available_layers[-1]]
        print(f"  Using layers: {layers_to_use}")
    
    print(f"\n[Step 3] Visualizing layers: {layers_to_use}")
    
    # Extract representations
    layer_reps = {layer_idx: reps['msa'][layer_idx] for layer_idx in layers_to_use}
    
    # Generate visualization using built-in function
    print(f"\n[Step 4] Applying t-SNE and generating visualizations...")
    try:
        results = visualize_layer_progression(
            layer_reps,
            method='tsne',
            n_components=2,
            save_dir=output_dir,
            rep_type='msa',
            flatten_mode='residue'
        )
        
        print(f"\n✓ Success! Generated t-SNE visualization")
        print(f"  Saved to: {output_dir}/msa_evolution_tsne.png")
        
        # Also create side-by-side comparison
        print(f"\n[Step 5] Creating side-by-side comparison...")
        create_side_by_side_tsne(results, layers_to_use, output_dir)
        
        print(f"\n{'='*70}")
        print("✓ Complete! Generated visualizations:")
        print(f"  1. Layer progression: {output_dir}/msa_evolution_tsne.png")
        print(f"  2. Side-by-side: {output_dir}/1UBQ_tsne_layers_comparison.png")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"  ✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


def create_side_by_side_tsne(embeddings_dict, layers, output_dir):
    """Create side-by-side comparison of layers."""
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(6*n_layers, 6))
    if n_layers == 1:
        axes = [axes]
    
    for idx, layer_idx in enumerate(layers):
        embedding = embeddings_dict[layer_idx]
        ax = axes[idx]
        
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=50)
        ax.set_title(f'Layer {layer_idx}', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('1UBQ MSA Representation Evolution (t-SNE)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, '1UBQ_tsne_layers_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved side-by-side comparison: {save_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate t-SNE between layers for 1UBQ')
    parser.add_argument('--pickle_file', type=str, default=None,
                       help='Path to .pt file with intermediate representations')
    parser.add_argument('--layers', type=str, default='0,23,47',
                       help='Comma-separated list of layer indices')
    parser.add_argument('--output_dir', type=str, default='outputs/1UBQ_tsne',
                       help='Output directory')
    
    args = parser.parse_args()
    
    layers = [int(l.strip()) for l in args.layers.split(',')]
    
    generate_tsne_1UBQ(
        pickle_file=args.pickle_file,
        layers=layers,
        output_dir=args.output_dir
    )


