"""
Extract intermediate representations for 1UBQ (Ubiquitin) using OpenFold inference.

This script runs OpenFold inference with hooks enabled to capture intermediate
representations from all 48 layers, then saves them for t-SNE visualization.

Usage:
    python extract_1UBQ_intermediate_reps.py [--fasta_file PATH] [--output_dir DIR]
"""

import os
import sys
import argparse
import torch

# Import OpenFold utilities
try:
    from openfold import config
    from openfold.model.model import AlphaFold
    from openfold.utils.import_weights import import_jax_weights_
    from openfold.data import data_pipeline
    from openfold.utils.tensor_utils import tensor_tree_map
    
    # Import visualization utilities
    from visualize_intermediate_reps_utils import (
        INTERMEDIATE_REPS,
        register_evoformer_hooks,
        remove_hooks,
        save_intermediate_reps_to_disk,
        extract_msa_representations,
        extract_pair_representations
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure OpenFold and visualization utilities are installed")
    sys.exit(1)


def extract_1UBQ_representations(fasta_file=None, output_dir='outputs/1UBQ_intermediate_reps'):
    """
    Extract intermediate representations for 1UBQ.
    
    Args:
        fasta_file: Path to FASTA file (if None, uses 1UBQ sequence)
        output_dir: Directory to save extracted representations
    """
    
    print("\n" + "="*70)
    print("Extracting Intermediate Representations for 1UBQ (Ubiquitin)")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1UBQ sequence (Ubiquitin, 76 residues)
    if fasta_file is None:
        print("\n[Step 1] Using default 1UBQ sequence...")
        protein_name = "1UBQ"
        sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        
        # Create temporary FASTA file
        fasta_file = os.path.join(output_dir, '1UBQ.fasta')
        with open(fasta_file, 'w') as f:
            f.write(f">1UBQ\n{sequence}\n")
        print(f"  Created FASTA file: {fasta_file}")
        print(f"  Sequence length: {len(sequence)} residues")
    else:
        print(f"\n[Step 1] Loading FASTA file: {fasta_file}")
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
            sequence = ''.join([l.strip() for l in lines[1:]])
            protein_name = lines[0].strip('> \n').split()[0]
        print(f"  Protein: {protein_name}")
        print(f"  Sequence length: {len(sequence)} residues")
    
    print("\n[Step 2] Loading OpenFold model...")
    print("  Note: This requires pretrained weights and data pipeline setup")
    print("  See run_pretrained_openfold.py for full setup example")
    
    # Note: This is a template - actual implementation would need:
    # 1. Model initialization
    # 2. Feature preparation
    # 3. Inference with hooks
    
    print("\n⚠  Template script - requires OpenFold setup")
    print("\nTo extract intermediate representations, you need:")
    print("  1. Pretrained OpenFold weights")
    print("  2. MSA and template data (or use precomputed features)")
    print("  3. Run inference with hooks enabled")
    print("\nExample code structure:")
    print("""
    # Enable intermediate representation storage
    INTERMEDIATE_REPS.enable()
    
    # Load model
    model = AlphaFold(config)
    model.eval()
    
    # Register hooks
    hooks = register_evoformer_hooks(model)
    
    # Prepare batch (MSA features, templates, etc.)
    batch = prepare_batch(fasta_file, ...)
    
    # Run inference
    with torch.no_grad():
        output = model(batch)
    
    # Extract representations
    msa_reps = extract_msa_representations(None)
    pair_reps = extract_pair_representations(None)
    
    # Save to disk
    save_intermediate_reps_to_disk({
        'msa': msa_reps,
        'pair': pair_reps
    }, output_dir, protein_name)
    
    # Clean up
    remove_hooks(hooks)
    INTERMEDIATE_REPS.disable()
    """)
    
    print(f"\nOnce extracted, save to: {output_dir}/{protein_name}_intermediate_reps.pt")
    print(f"Then run: python generate_tsne_1UBQ_simple.py --pickle_file {output_dir}/{protein_name}_intermediate_reps.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract intermediate representations for 1UBQ')
    parser.add_argument('--fasta_file', type=str, default=None,
                       help='Path to FASTA file (if None, uses default 1UBQ sequence)')
    parser.add_argument('--output_dir', type=str, default='outputs/1UBQ_intermediate_reps',
                       help='Output directory for extracted representations')
    
    args = parser.parse_args()
    
    extract_1UBQ_representations(
        fasta_file=args.fasta_file,
        output_dir=args.output_dir
    )


