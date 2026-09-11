"""
Demonstration script for t-SNE and dimensionality reduction analysis of OpenFold representations.

This script shows how to use the dimensionality reduction utilities to analyze
intermediate representations extracted from OpenFold's 48-layer network.

Author: Shreyas, Boyang
Building on: Jayanth's intermediate representation extraction utilities

Usage:
    python demo_tsne_reduction.py [--input_reps PATH] [--output_dir DIR] [--demo DEMO_TYPE] [--layer LAYER] [--layers LAYERS] [--interactive] [--output_dir DIR]

Example:
    python demo_tsne_reduction.py --input_reps demo_outputs/demo_protein_intermediate_reps.pt --demo single --layer 47 --interactive --output_dir outputs/tsne_demo_interactive
"""

import os
import numpy as np
import torch
import argparse

# Import Jayanth's utilities
from visualize_intermediate_reps_utils import (
    load_intermediate_reps_from_disk,
    INTERMEDIATE_REPS
)

# Import new dimensionality reduction utilities
from visualize_dimensionality_reduction import (
    run_complete_dimensionality_reduction_analysis,
    compare_reduction_methods,
    visualize_layer_progression,
    prepare_representations_for_reduction,
    plot_pca_variance_explained,
    plot_tsne_perplexity_comparison
)

# Normalize legacy representation keys to the expected schema
def _normalize_rep_keys(reps: dict) -> dict:
    if any(k in reps for k in ['msa', 'pair', 'single']):
        return reps

    normalized = {}
    if 'msa_layers' in reps:
        normalized['msa'] = reps.get('msa_layers', {})
    if 'pair_layers' in reps:
        normalized['pair'] = reps.get('pair_layers', {})
    if 'single_layers' in reps:
        normalized['single'] = reps.get('single_layers', {})

    # Fallback to final tensors if layer dicts are missing/empty
    if not normalized.get('msa') and 'final_msa' in reps:
        normalized['msa'] = {0: reps['final_msa']}
    if not normalized.get('pair') and 'final_pair' in reps:
        normalized['pair'] = {0: reps['final_pair']}

    return normalized


def demo_single_layer_analysis(intermediate_reps, output_dir, layer_idx=47, interactive: bool = False):
    """
    Demonstrate dimensionality reduction on a single layer.
    
    Args:
        intermediate_reps: Dictionary of intermediate representations
        output_dir: Directory to save outputs
        layer_idx: Layer to analyze
    """
    print(f"\n{'='*70}")
    print(f" DEMO 1: Single Layer Analysis (Layer {layer_idx})")
    print(f"{'='*70}\n")
    
    available_layers = sorted(intermediate_reps.get('msa', {}).keys())
    if not available_layers:
        raise ValueError("No MSA layers available in the provided representations.")
    if layer_idx not in available_layers:
        print(f"Layer {layer_idx} not found. Available layers: {available_layers}")
        layer_idx = available_layers[-1]
        print(f"Falling back to layer {layer_idx}")

    # Extract MSA representation for specified layer
    msa_rep = intermediate_reps['msa'][layer_idx]
    print(f"MSA representation shape: {msa_rep.shape}")
    
    # Prepare data
    msa_data = prepare_representations_for_reduction(msa_rep, flatten_mode='residue')
    print(f"Prepared data shape: {msa_data.shape} (residues × features)")
    
    # Create output directory
    single_layer_dir = os.path.join(output_dir, f'single_layer_{layer_idx}')
    os.makedirs(single_layer_dir, exist_ok=True)
    
    # 1. PCA Variance Analysis
    print("\n--- PCA Variance Analysis ---")
    variance_ext = 'html' if interactive else 'png'
    pca_variance_path = os.path.join(single_layer_dir, f'pca_variance.{variance_ext}')
    plot_pca_variance_explained(
        msa_data,
        max_components=min(50, msa_data.shape[1]),
        save_path=pca_variance_path,
        interactive=interactive,
    )
    
    # 2. Compare all methods
    print("\n--- Comparing Dimensionality Reduction Methods ---")
    comparison_dir = os.path.join(single_layer_dir, 'method_comparison')
    results = compare_reduction_methods(
        msa_data,
        methods=['pca', 'tsne', 'umap'],
        n_components=2,
        save_dir=comparison_dir,
        interactive=interactive,
    )
    
    # 3. t-SNE Perplexity Comparison
    print("\n--- t-SNE Perplexity Analysis ---")
    perplexity_ext = 'html' if interactive else 'png'
    perplexity_path = os.path.join(single_layer_dir, f'tsne_perplexity.{perplexity_ext}')
    plot_tsne_perplexity_comparison(msa_data, perplexities=[5, 20, 50],
                                   save_path=perplexity_path, interactive=interactive)
    
    print(f"\n✓ Single layer analysis complete!")
    print(f"  Results saved to: {single_layer_dir}")


def demo_layer_progression(intermediate_reps, output_dir, layer_subset=None, interactive: bool = False):
    """
    Demonstrate layer progression visualization.
    
    Args:
        intermediate_reps: Dictionary of intermediate representations
        output_dir: Directory to save outputs
        layer_subset: List of layers to analyze (None for default selection)
    """
    print(f"\n{'='*70}")
    print(" DEMO 2: Layer Progression Analysis")
    print(f"{'='*70}\n")
    
    if layer_subset is None:
        # Select representative layers
        all_layers = sorted(intermediate_reps['msa'].keys())
        n_layers = len(all_layers)
        layer_subset = [all_layers[0], all_layers[n_layers//4], all_layers[n_layers//2], 
                       all_layers[3*n_layers//4], all_layers[-1]]
    
    print(f"Analyzing layer progression for layers: {layer_subset}")
    
    # Extract representations for selected layers
    msa_layer_reps = {layer: intermediate_reps['msa'][layer] 
                      for layer in layer_subset 
                      if layer in intermediate_reps['msa']}
    
    # Create output directory
    progression_dir = os.path.join(output_dir, 'layer_progression')
    os.makedirs(progression_dir, exist_ok=True)
    
    # Visualize with different methods
    for method in ['pca', 'tsne', 'umap']:
        print(f"\n--- {method.upper()} Layer Progression ---")
        method_dir = os.path.join(progression_dir, method)
        
        results = visualize_layer_progression(
            msa_layer_reps,
            method=method,
            n_components=2,
            save_dir=method_dir,
            rep_type='msa',
            flatten_mode='residue',
            interactive=interactive,
        )
    
    print(f"\n✓ Layer progression analysis complete!")
    print(f"  Results saved to: {progression_dir}")


def demo_comprehensive_analysis(intermediate_reps, output_dir, layer_subset=None, interactive: bool = False):
    """
    Run comprehensive analysis using the all-in-one function.
    
    Args:
        intermediate_reps: Dictionary of intermediate representations
        output_dir: Directory to save outputs
        layer_subset: List of layers to analyze
    """
    print(f"\n{'='*70}")
    print(" DEMO 3: Comprehensive Analysis (All-in-One)")
    print(f"{'='*70}\n")
    
    if layer_subset is None:
        all_layers = sorted(intermediate_reps['msa'].keys())
        n_layers = len(all_layers)
        layer_subset = [all_layers[0], all_layers[n_layers//3], 
                       all_layers[2*n_layers//3], all_layers[-1]]
    
    print(f"Running comprehensive analysis on {len(layer_subset)} layers: {layer_subset}")
    
    # MSA Analysis
    print("\n>>> MSA Representation Analysis")
    msa_dir = os.path.join(output_dir, 'comprehensive_msa')
    msa_results = run_complete_dimensionality_reduction_analysis(
        representations=intermediate_reps['msa'],
        output_dir=msa_dir,
        rep_type='msa',
        methods=['pca', 'tsne', 'umap'],
        flatten_mode='residue',
        layer_subset=layer_subset,
        interactive=interactive,
    )
    
    # Pair Analysis (if available)
    if 'pair' in intermediate_reps and intermediate_reps['pair']:
        print("\n>>> Pair Representation Analysis")
        pair_dir = os.path.join(output_dir, 'comprehensive_pair')
        pair_results = run_complete_dimensionality_reduction_analysis(
            representations=intermediate_reps['pair'],
            output_dir=pair_dir,
            rep_type='pair',
            methods=['pca', 'umap'],  # Skip t-SNE for pair (can be slow)
            flatten_mode='pairwise',
            layer_subset=[layer_subset[0], layer_subset[-1]],  # Just first and last
            interactive=interactive,
        )
    
    print(f"\n✓ Comprehensive analysis complete!")
    print(f"  Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate t-SNE and dimensionality reduction for OpenFold representations"
    )
    parser.add_argument(
        '--input_reps',
        type=str,
        default='./demo_outputs/demo_protein_intermediate_reps.pt',
        help='Path to saved intermediate representations (.pt file)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/tsne_demo',
        help='Directory to save visualization outputs'
    )
    parser.add_argument(
        '--demo',
        type=str,
        choices=['single', 'progression', 'comprehensive', 'all'],
        default='all',
        help='Which demo to run'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=47,
        help='Layer index for single layer demo'
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=None,
        help='Layer indices for progression/comprehensive demos'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Use Plotly and save interactive HTML visualizations for 2D embeddings'
    )
    
    args = parser.parse_args()
    
    # Load intermediate representations
    print(f"\n{'='*70}")
    print(" Loading Intermediate Representations")
    print(f"{'='*70}\n")
    
    if not os.path.exists(args.input_reps):
        print(f"ERROR: Input file not found: {args.input_reps}")
        print("\nPlease run OpenFold inference first to generate intermediate representations.")
        print("See Jayanth's demo notebooks for how to extract representations.")
        return
    
    intermediate_reps = load_intermediate_reps_from_disk(args.input_reps)
    intermediate_reps = _normalize_rep_keys(intermediate_reps)
    print(f"✓ Loaded representations from: {args.input_reps}")
    print(f"  MSA layers: {sorted(intermediate_reps.get('msa', {}).keys())}")
    print(f"  Pair layers: {sorted(intermediate_reps.get('pair', {}).keys())}")
    print(f"  Single layers: {sorted(intermediate_reps.get('single', {}).keys())}")

    # Clamp requested layer to available layers (for demo pickle with limited layers)
    available_layers = sorted(intermediate_reps.get('msa', {}).keys())
    if args.layer not in available_layers:
        print(f"Requested layer {args.layer} not in available layers {available_layers}, defaulting to {available_layers[-1]}")
        args.layer = available_layers[-1]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}")
    
    # Run requested demo(s)
    if args.demo in ['single', 'all']:
        demo_single_layer_analysis(intermediate_reps, args.output_dir, args.layer, interactive=args.interactive)
    
    if args.demo in ['progression', 'all']:
        demo_layer_progression(intermediate_reps, args.output_dir, args.layers, interactive=args.interactive)
    
    if args.demo in ['comprehensive', 'all']:
        demo_comprehensive_analysis(intermediate_reps, args.output_dir, args.layers, interactive=args.interactive)
    
    # Final summary
    print(f"\n{'='*70}")
    print(" ALL DEMOS COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated outputs include:")
    print("  ✓ PCA variance analysis")
    print("  ✓ t-SNE, PCA, and UMAP 2D embeddings")
    print("  ✓ Method comparison plots")
    print("  ✓ Layer progression visualizations")
    print("  ✓ Perplexity comparison for t-SNE")
    print("\nNext steps:")
    print("  - Explore the output directory")
    print("  - Try different layer subsets")
    print("  - Compare results across different proteins")
    print("  - Integrate with attention visualizations")


if __name__ == "__main__":
    main()


