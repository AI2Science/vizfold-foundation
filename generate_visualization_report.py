#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from PIL import Image
import os
import numpy as np
import torch
import datetime
from visualize_intermediate_reps_utils import *

def create_visualization_report():
    """Generate a comprehensive PDF report with all enhanced visualizations."""
    
    print("Creating visualization report...")
    
    # Create mock data (same as demo)
    print("Generating mock protein data...")
    n_seq = 15
    n_res = 100
    c_m = 256
    c_z = 128

    # Generate base representations
    base_msa = torch.randn(n_seq, n_res, c_m)
    base_pair = torch.randn(n_res, n_res, c_z)

    mock_output = {
        'msa': base_msa,
        'pair': base_pair,
        'sm': {
            'frames': torch.randn(8, n_res, 7),
            'angles': torch.randn(8, n_res, 7, 2),
            'positions': torch.randn(8, n_res, 14, 3),
            'single': torch.randn(n_res, 384)
        },
        'final_atom_positions': torch.randn(n_res, 37, 3)
    }

    # Extract representations
    msa_final = extract_msa_representations(mock_output)
    pair_final = extract_pair_representations(mock_output)
    
    # Create 48 layers of mock data with realistic convergence
    print("Simulating 48 Evoformer layers...")
    n_layers = 48
    msa_layers = {}
    pair_layers = {}
    
    for layer_idx in range(n_layers):
        noise_factor = 0.3 * np.exp(-layer_idx / 20)
        noise_msa = torch.randn_like(base_msa) * noise_factor
        msa_layers[layer_idx] = base_msa + noise_msa
        noise_pair = torch.randn_like(base_pair) * noise_factor
        pair_layers[layer_idx] = base_pair + noise_pair

    # Generate contact map
    mock_contact_map = generate_mock_contact_map(n_res, contact_probability=0.15, seed=42)
    
    # Create PDF report
    pdf_path = "demo_outputs/visualization_report.pdf"
    os.makedirs("demo_outputs", exist_ok=True)
    
    with matplotlib.backends.backend_pdf.PdfPages(pdf_path) as pdf:
        
        # Page 1: Multi-Layer Evolution (48 layers)
        print("  Creating 48-layer evolution plot...")
        fig = plot_multilayer_evolution(
            msa_layers, 
            residue_indices=[10, 25, 50, 75, 90],
            save_path="temp_multilayer_evolution.png",
            rep_type='msa',
            layer_sampling='uniform'
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Stratified MSA Comparison (13 layers)
        print("  Creating stratified MSA comparison...")
        sampled_layers = stratified_layer_sampling(n_layers=48, strategy='grouped')
        fig = plot_stratified_layer_comparison(
            msa_layers,
            layer_indices=sampled_layers,
            save_path="temp_stratified_msa.png",
            rep_type='msa',
            aggregate_method='mean'
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Stratified Pair Comparison (13 layers)
        print("  Creating stratified pair comparison...")
        fig = plot_stratified_layer_comparison(
            pair_layers,
            layer_indices=sampled_layers,
            save_path="temp_stratified_pair.png",
            rep_type='pair',
            aggregate_method='mean'
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 4: MSA Convergence Analysis
        print("  Creating MSA convergence analysis...")
        fig = plot_layer_convergence_analysis(
            msa_layers,
            save_path="temp_msa_convergence.png",
            rep_type='msa'
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 5: Pair Convergence Analysis
        print("  Creating pair convergence analysis...")
        fig = plot_layer_convergence_analysis(
            pair_layers,
            save_path="temp_pair_convergence.png",
            rep_type='pair'
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 6: Enhanced Pair Heatmap with Contact Overlay
        print("  Creating enhanced pair heatmap with contact overlay...")
        fig = plot_pair_representation_heatmap(pair_layers[47], 47, "temp_pair.png",
                                             contact_map=mock_contact_map, show_correlation=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # Clean up temporary files
    temp_files = [
        "temp_multilayer_evolution.png", "temp_stratified_msa.png", 
        "temp_stratified_pair.png", "temp_msa_convergence.png",
        "temp_pair_convergence.png", "temp_pair.png"
    ]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"\n✅ Visualization report generated: {pdf_path}")
    print(f"📄 Report contains 6 pages with 48-layer multi-layer analysis")
    
    return pdf_path

if __name__ == "__main__":
    report_path = create_visualization_report()
    print(f"\n🎉 Your visualization report is ready: {report_path}")
    print("You can now open this PDF to view all enhanced visualizations!")
