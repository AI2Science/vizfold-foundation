#!/usr/bin/env python3

import torch
import numpy as np
import os
import tempfile
from visualize_intermediate_reps_utils import *

print("Testing intermediate representation visualizations...")
test_output_dir = tempfile.mkdtemp(prefix="test_")
print(f"Output dir: {test_output_dir}")

print("\n1. Testing MSA visualizations...")
msa_tensor = torch.randn(10, 50, 256)

for method in ['mean', 'max', 'norm']:
    save_path = os.path.join(test_output_dir, f"msa_{method}.png")
    fig = plot_msa_representation_heatmap(msa_tensor, 0, save_path, method)
    plt.close(fig)
    print(f"  Created {method} MSA heatmap")

print("\n2. Testing Pair visualizations...")
pair_tensor = torch.randn(50, 50, 128)

for cmap in ['RdBu_r', 'viridis']:
    save_path = os.path.join(test_output_dir, f"pair_{cmap}.png")
    fig = plot_pair_representation_heatmap(pair_tensor, 0, save_path, cmap=cmap)
    plt.close(fig)
    print(f"  Created {cmap} pair heatmap")

print("\n3. Testing evolution plots...")
msa_layers = {i: torch.randn(10, 50, 256) for i in range(5)}
pair_layers = {i: torch.randn(50, 50, 128) for i in range(5)}

fig = plot_representation_evolution(msa_layers, 10, 
                                   os.path.join(test_output_dir, "msa_evo.png"), 
                                   'msa')
plt.close(fig)
print("  Created MSA evolution plot")

fig = plot_representation_evolution(pair_layers, 10,
                                   os.path.join(test_output_dir, "pair_evo.png"),
                                   'pair')
plt.close(fig)
print("  Created Pair evolution plot")

print("\n4. Testing channel visualizations...")
fig = plot_channel_specific_heatmap(msa_tensor, 0, 64, 
                                   os.path.join(test_output_dir, "msa_ch64.png"), 
                                   'msa')
plt.close(fig)
print("  Created MSA channel heatmap")

fig = plot_channel_specific_heatmap(pair_tensor, 0, 32,
                                   os.path.join(test_output_dir, "pair_ch32.png"),
                                   'pair')
plt.close(fig)
print("  Created Pair channel heatmap")

print("\n5. Testing save/load...")
intermediate_reps = {
    'msa': msa_layers,
    'pair': pair_layers,
    'final_msa': msa_tensor,
    'final_pair': pair_tensor
}

save_intermediate_reps_to_disk(intermediate_reps, test_output_dir, "test")
loaded_reps = load_intermediate_reps_from_disk(
    os.path.join(test_output_dir, "test_intermediate_reps.pt"))
print(f"  Saved and loaded {len(loaded_reps)} representations")

print("\n6. Testing structure extraction...")
mock_output = {
    'sm': {
        'frames': torch.randn(8, 50, 7),
        'angles': torch.randn(8, 50, 7, 2),
        'positions': torch.randn(8, 50, 14, 3),
        'single': torch.randn(50, 384)
    },
    'final_atom_positions': torch.randn(50, 37, 3)
}

structure_reps = extract_structure_representations(mock_output)
print(f"  Extracted structure with {len(structure_reps)} components")

print(f"\nAll tests completed! Outputs in: {test_output_dir}")
files = os.listdir(test_output_dir)
print(f"Generated {len(files)} files")

