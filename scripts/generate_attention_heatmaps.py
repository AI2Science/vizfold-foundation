#!/usr/bin/env python3
"""
CLI script for generating attention heatmap visualizations.

This script generates heatmap visualizations of OpenFold attention mechanisms,
enabling cross-head comparison and pattern recognition that complements
existing arc diagrams and PyMOL overlays.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the visualization modules
sys.path.append(str(Path(__file__).parent.parent))

from visualize_attention_heatmap_utils import (
    plot_all_heads_heatmap,
    plot_combined_attention_heatmap
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate attention heatmap visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate MSA Row attention heatmap
  python scripts/generate_attention_heatmaps.py \\
    --attention_dir ./outputs/attention_files_6KWC_demo_tri_18 \\
    --output_dir ./outputs/heatmap_visualizations \\
    --protein 6KWC \\
    --layer 47 \\
    --attention_type msa_row

  # Generate Triangle Start attention heatmap
  python scripts/generate_attention_heatmaps.py \\
    --attention_dir ./outputs/attention_files_6KWC_demo_tri_18 \\
    --output_dir ./outputs/heatmap_visualizations \\
    --protein 6KWC \\
    --layer 47 \\
    --attention_type triangle_start

  # Generate combined heatmap (both MSA Row and Triangle Start)
  python scripts/generate_attention_heatmaps.py \\
    --attention_dir ./outputs/attention_files_6KWC_demo_tri_18 \\
    --output_dir ./outputs/heatmap_visualizations \\
    --protein 6KWC \\
    --layer 47 \\
    --attention_type combined

  # Generate heatmaps for multiple layers
  python scripts/generate_attention_heatmaps.py \\
    --attention_dir ./outputs/attention_files_6KWC_demo_tri_18 \\
    --output_dir ./outputs/heatmap_visualizations \\
    --protein 6KWC \\
    --layers 40 45 47 50 \\
    --attention_type msa_row
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--attention_dir",
        type=str,
        required=True,
        help="Directory containing attention text files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output PNG files"
    )
    
    parser.add_argument(
        "--protein",
        type=str,
        required=True,
        help="Protein identifier (e.g., '6KWC')"
    )
    
    # Optional arguments
    parser.add_argument(
        "--attention_type",
        type=str,
        choices=["msa_row", "triangle_start", "combined"],
        default="combined",
        help="Type of attention to visualize (default: combined)"
    )
    
    parser.add_argument(
        "--layer",
        type=int,
        default=47,
        help="Layer number to visualize (default: 47)"
    )
    
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Multiple layer numbers to visualize (overrides --layer)"
    )
    
    parser.add_argument(
        "--seq_length",
        type=int,
        help="Sequence length (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--fasta_path",
        type=str,
        default="./examples/monomer/fasta_dir_6KWC/6KWC.fasta",
        help="Path to FASTA file for sequence length detection"
    )
    
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["global", "per_head"],
        default="global",
        help="Normalization method (default: global)"
    )
    
    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name (default: viridis)"
    )
    
    parser.add_argument(
        "--figsize_per_head",
        type=float,
        nargs=2,
        default=[2.0, 2.0],
        metavar=("WIDTH", "HEIGHT"),
        help="Size of each subplot in inches (default: 2.0 2.0)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution in DPI (default: 300)"
    )
    
    parser.add_argument(
        "--residue_indices",
        nargs='+',
        type=int,
        help="Residue indices for triangle_start attention (default: [18])"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.attention_dir):
        print(f"Error: Attention directory not found: {args.attention_dir}")
        sys.exit(1)
    
    # Determine layers to process
    if args.layers:
        layers = args.layers
    else:
        layers = [args.layer]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each layer
    for layer_idx in layers:
        print(f"\nProcessing layer {layer_idx}...")
        
        try:
            if args.attention_type == "combined":
                # Generate combined heatmap
                output_path = plot_combined_attention_heatmap(
                    attention_dir=args.attention_dir,
                    output_dir=args.output_dir,
                    protein=args.protein,
                    layer_idx=layer_idx,
                    seq_length=args.seq_length,
                    fasta_path=args.fasta_path,
                    normalization=args.normalization,
                    colormap=args.colormap,
                    figsize_per_head=tuple(args.figsize_per_head),
                    dpi=args.dpi,
                    save_to_png=True,
                    residue_indices=args.residue_indices
                )
                
                if args.verbose:
                    print(f"Generated combined heatmap: {output_path}")
            
            else:
                # Generate individual attention type heatmap
                output_path = plot_all_heads_heatmap(
                    attention_dir=args.attention_dir,
                    output_dir=args.output_dir,
                    protein=args.protein,
                    attention_type=args.attention_type,
                    layer_idx=layer_idx,
                    seq_length=args.seq_length,
                    fasta_path=args.fasta_path,
                    normalization=args.normalization,
                    colormap=args.colormap,
                    figsize_per_head=tuple(args.figsize_per_head),
                    dpi=args.dpi,
                    save_to_png=True,
                    residue_indices=args.residue_indices
                )
                
                if args.verbose:
                    print(f"Generated {args.attention_type} heatmap: {output_path}")
        
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue
        except Exception as e:
            print(f"Error processing layer {layer_idx}: {e}")
            continue
    
    print(f"\nHeatmap generation complete! Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()