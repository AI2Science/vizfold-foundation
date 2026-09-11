#!/usr/bin/env python3

import torch
import numpy as np
import os
from visualize_intermediate_reps_utils import *

print("OpenFold 48-Layer Multi-Layer Analysis Demo")
print("=" * 60)

# Create mock protein data
print("\nCreating mock protein data...")
n_seq = 15
n_res = 100
c_m = 256
c_z = 128

# Generate base representations
base_msa = torch.randn(n_seq, n_res, c_m)
base_pair = torch.randn(n_res, n_res, c_z)

print(f"Base MSA shape: {base_msa.shape}")
print(f"Base Pair shape: {base_pair.shape}")

# Simulate 48 layers of processing
print("\nSimulating 48 Evoformer layers...")
n_layers = 48

msa_layers = {}
pair_layers = {}

# Create realistic layer evolution with gradual changes
for layer_idx in range(n_layers):
    # Add progressive noise that decreases over layers (simulating convergence)
    noise_factor = 0.3 * np.exp(-layer_idx / 20)  # Exponential decay
    
    # MSA evolution
    noise_msa = torch.randn_like(base_msa) * noise_factor
    msa_layers[layer_idx] = base_msa + noise_msa
    
    # Pair evolution  
    noise_pair = torch.randn_like(base_pair) * noise_factor
    pair_layers[layer_idx] = base_pair + noise_pair
    
    if (layer_idx + 1) % 12 == 0:
        print(f"  Generated layers 0-{layer_idx}")

print(f"✓ Created {n_layers} layers for both MSA and Pair representations")

# Create output directory
output_dir = "demo_outputs"
os.makedirs(output_dir, exist_ok=True)

# 1. Multi-layer evolution plot with stratified sampling
print("\n" + "="*60)
print("1. Multi-Layer Evolution Analysis")
print("="*60)

print("  Creating evolution plot across all 48 layers...")
fig = plot_multilayer_evolution(
    msa_layers, 
    residue_indices=[10, 25, 50, 75, 90],
    save_path=f"{output_dir}/multilayer_evolution.png",
    rep_type='msa',
    layer_sampling='uniform'
)
plt.close(fig)
print("  ✓ Multi-layer evolution plot saved")

# 2. Stratified layer comparison
print("\n" + "="*60)
print("2. Stratified Layer Comparison")
print("="*60)

# Use grouped strategy for comparison
sampled_layers = stratified_layer_sampling(n_layers=48, strategy='grouped')
print(f"  Sampling layers: {sampled_layers}")

print("  Creating stratified MSA comparison...")
fig = plot_stratified_layer_comparison(
    msa_layers,
    layer_indices=sampled_layers,
    save_path=f"{output_dir}/stratified_msa_comparison.png",
    rep_type='msa',
    aggregate_method='mean'
)
plt.close(fig)
print("  ✓ Stratified MSA comparison saved")

print("  Creating stratified Pair comparison...")
fig = plot_stratified_layer_comparison(
    pair_layers,
    layer_indices=sampled_layers,
    save_path=f"{output_dir}/stratified_pair_comparison.png",
    rep_type='pair',
    aggregate_method='mean'
)
plt.close(fig)
print("  ✓ Stratified Pair comparison saved")

# 3. Convergence analysis
print("\n" + "="*60)
print("3. Layer Convergence Analysis")
print("="*60)

print("  Analyzing MSA representation convergence...")
fig = plot_layer_convergence_analysis(
    msa_layers,
    save_path=f"{output_dir}/msa_convergence_analysis.png",
    rep_type='msa'
)
plt.close(fig)
print("  ✓ MSA convergence analysis saved")

print("  Analyzing Pair representation convergence...")
fig = plot_layer_convergence_analysis(
    pair_layers,
    save_path=f"{output_dir}/pair_convergence_analysis.png",
    rep_type='pair'
)
plt.close(fig)
print("  ✓ Pair convergence analysis saved")

# 4. Compare different sampling strategies
print("\n" + "="*60)
print("4. Sampling Strategy Comparison")
print("="*60)

strategies = ['uniform', 'grouped', 'logarithmic']
for strategy in strategies:
    layers = stratified_layer_sampling(n_layers=48, strategy=strategy)
    print(f"  {strategy:12} strategy: {len(layers)} layers sampled -> {layers[:5]}...{layers[-2:]}")

# 5. Generate mock contact map and enhanced pair visualization
print("\n" + "="*60)
print("5. Enhanced Visualizations with Contact Overlay")
print("="*60)

print("  Generating mock contact map...")
contact_map = generate_mock_contact_map(n_res, contact_probability=0.15, seed=42)
print(f"  Contact map generated: {contact_map.shape}")

print("  Creating enhanced pair heatmap (layer 47)...")
fig = plot_pair_representation_heatmap(
    pair_layers[47], 
    47,
    f"{output_dir}/pair_layer47_with_contacts.png",
    contact_map=contact_map,
    show_correlation=True
)
plt.close(fig)
print("  ✓ Enhanced pair heatmap saved")

# 6. Layer-specific analysis
print("\n" + "="*60)
print("6. Layer-Specific Detailed Analysis")
print("="*60)

key_layers = [0, 12, 24, 36, 47]
print(f"  Analyzing key layers: {key_layers}")

for layer_idx in key_layers:
    print(f"  Layer {layer_idx:2d} - Generating MSA heatmap...")
    fig = plot_msa_representation_heatmap(
        msa_layers[layer_idx],
        layer_idx,
        f"{output_dir}/msa_layer{layer_idx:02d}.png",
        highlight_residue=50,
        custom_ticks=[0, 25, 50, 75]
    )
    plt.close(fig)

print("  ✓ All layer-specific visualizations saved")

# 7. Save layer data
print("\n" + "="*60)
print("7. Saving Multi-Layer Data")
print("="*60)

multilayer_data = {
    'msa_layers': msa_layers,
    'pair_layers': pair_layers,
    'n_layers': n_layers,
    'layer_info': {
        'uniform_sample': stratified_layer_sampling(48, 'uniform'),
        'grouped_sample': stratified_layer_sampling(48, 'grouped'),
        'logarithmic_sample': stratified_layer_sampling(48, 'logarithmic')
    }
}

save_intermediate_reps_to_disk(multilayer_data, output_dir, "multilayer_48")
print(f"  ✓ Saved 48-layer data to {output_dir}/multilayer_48_intermediate_reps.pt")

# Summary
print("\n" + "="*60)
print("DEMO COMPLETE!")
print("="*60)
print(f"\nGenerated visualizations in: {output_dir}/")
print("\nVisualization Summary:")
print("  ✓ Multi-layer evolution plot (48 layers)")
print("  ✓ Stratified MSA comparison (13 layers)")
print("  ✓ Stratified Pair comparison (13 layers)")
print("  ✓ MSA convergence analysis")
print("  ✓ Pair convergence analysis")
print("  ✓ Enhanced pair heatmap with contacts")
print(f"  ✓ {len(key_layers)} layer-specific MSA heatmaps")
print(f"\nTotal layers analyzed: {n_layers}")
print(f"Total files generated: {len(os.listdir(output_dir))}")

print("\n" + "="*60)
print("Ready for publication-quality multi-layer analysis!")
print("="*60)

