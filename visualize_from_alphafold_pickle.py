"""
Generate t-SNE visualizations from AlphaFold/OpenFold pickle output files.

This script adapts to standard AlphaFold output format and extracts
representations for visualization.

Usage:
    python visualize_from_alphafold_pickle.py <pickle_file> [--output_dir OUTPUT_DIR]

Example:
    python visualize_from_alphafold_pickle.py proteins_pickle/1G1J_*.pickle
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
        apply_pca,
        plot_2d_embedding,
        SKLEARN_AVAILABLE
    )
    REDUCTION_AVAILABLE = True
except ImportError:
    REDUCTION_AVAILABLE = False
    print("Warning: Could not import dimensionality reduction utilities")


def load_alphafold_pickle(pickle_path):
    """
    Load AlphaFold/OpenFold pickle file and extract representations.
    
    Args:
        pickle_path: Path to pickle file
    
    Returns:
        Dictionary with extracted data
    """
    print(f"Loading pickle file: {pickle_path}")
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded successfully!")
        print(f"  Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())[:10]}")  # Show first 10 keys
        
        return data
    
    except Exception as e:
        print(f"✗ Error loading pickle: {e}")
        return None


def extract_representations(data):
    """
    Extract usable representations from AlphaFold pickle.
    
    Different formats are possible:
    1. Direct intermediate representations (Jayanth's format)
    2. Standard AlphaFold output_dict
    3. Model parameters
    
    Returns:
        Dictionary with representations or None
    """
    print("\nExtracting representations...")
    
    if not isinstance(data, dict):
        print("✗ Data is not a dictionary")
        return None
    
    # Check for different possible formats
    representations = {}
    
    # Format 1: Jayanth's intermediate representation format
    if 'msa' in data and isinstance(data['msa'], dict):
        print("✓ Found Jayanth's intermediate representation format")
        representations['msa'] = data['msa']
        representations['pair'] = data.get('pair', {})
        representations['single'] = data.get('single', {})
        return representations
    
    # Format 2: Standard AlphaFold output_dict
    # Look for embeddings or representations
    possible_rep_keys = [
        'structure_module',
        'representations',
        'msa_first_row',
        'pair',
        'single',
        'evoformer_output'
    ]
    
    for key in possible_rep_keys:
        if key in data:
            value = data[key]
            if hasattr(value, 'shape'):
                print(f"✓ Found '{key}' with shape {value.shape}")
                representations[key] = value
            elif isinstance(value, dict):
                print(f"✓ Found '{key}' (nested dict)")
                representations[key] = value
    
    # Format 3: Direct array (single representation)
    if not representations and hasattr(data, 'shape'):
        print(f"✓ Data is a single array with shape {data.shape}")
        representations['data'] = data
    
    if not representations:
        print("✗ Could not find recognizable representations")
        print(f"  Available keys: {list(data.keys())[:20]}")
        return None
    
    return representations


def prepare_data_for_visualization(representations, key='msa', layer=-1):
    """
    Prepare representations for dimensionality reduction.
    
    Args:
        representations: Extracted representations
        key: Which representation to use
        layer: Which layer (if multi-layer)
    
    Returns:
        2D numpy array ready for visualization
    """
    print(f"\nPreparing data for visualization...")
    print(f"  Using key: '{key}'")
    
    if key not in representations:
        # Try to find any usable data
        print(f"  Key '{key}' not found, trying alternatives...")
        available_keys = list(representations.keys())
        if available_keys:
            key = available_keys[0]
            print(f"  Using '{key}' instead")
        else:
            print("✗ No data available")
            return None
    
    data = representations[key]
    
    # Handle different formats
    # Multi-layer format (dict of layers)
    if isinstance(data, dict):
        if layer == -1:
            layer = max(data.keys())
        print(f"  Using layer: {layer}")
        data = data[layer]
    
    # Convert to numpy if needed
    if hasattr(data, 'cpu'):  # PyTorch tensor
        data = data.cpu().numpy()
    elif hasattr(data, 'numpy'):  # TensorFlow
        data = data.numpy()
    
    # Reshape if needed
    original_shape = data.shape
    print(f"  Original shape: {original_shape}")
    
    # Flatten to 2D (samples, features)
    if len(original_shape) == 1:
        # 1D array - make it 2D
        data = data.reshape(1, -1)
    elif len(original_shape) == 2:
        # Already 2D - good!
        pass
    elif len(original_shape) == 3:
        # (batch, seq_len, channels) - flatten batch and seq
        data = data.reshape(-1, original_shape[-1])
    elif len(original_shape) == 4:
        # (batch, seq1, seq2, channels) - flatten everything except channels
        data = data.reshape(-1, original_shape[-1])
    else:
        print(f"✗ Unsupported shape: {original_shape}")
        return None
    
    print(f"  Prepared shape: {data.shape}")
    print(f"  {data.shape[0]} samples × {data.shape[1]} features")
    
    return data


def visualize_protein(pickle_path, output_dir='outputs/alphafold_viz'):
    """
    Main function to visualize protein from AlphaFold pickle.
    """
    print("="*70)
    print("AlphaFold Pickle Visualization")
    print("="*70)
    
    # Extract protein name from filename
    protein_name = os.path.basename(pickle_path).split('_')[0]
    print(f"\nProtein: {protein_name}")
    
    # Create output directory
    output_dir = os.path.join(output_dir, protein_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load pickle
    data = load_alphafold_pickle(pickle_path)
    if data is None:
        return False
    
    # Extract representations
    representations = extract_representations(data)
    if representations is None:
        return False
    
    # Try to visualize each available representation
    success = False
    for key in representations.keys():
        print(f"\n{'='*70}")
        print(f"Attempting to visualize: {key}")
        print(f"{'='*70}")
        
        # Prepare data
        viz_data = prepare_data_for_visualization(representations, key=key)
        if viz_data is None:
            print(f"✗ Could not prepare {key} for visualization")
            continue
        
        # Check if we have too few samples
        if viz_data.shape[0] < 3:
            print(f"✗ Too few samples ({viz_data.shape[0]}) for meaningful visualization")
            continue
        
        # Check if we have features
        if viz_data.shape[1] < 2:
            print(f"✗ Too few features ({viz_data.shape[1]}) for dimensionality reduction")
            continue
        
        # Apply PCA
        if not SKLEARN_AVAILABLE:
            print("✗ scikit-learn not available, skipping visualization")
            continue
        
        print(f"\nApplying PCA...")
        try:
            pca_result, pca_model = apply_pca(viz_data, n_components=2)
            variance = pca_model.explained_variance_ratio_
            print(f"  ✓ PCA complete")
            print(f"    Explained variance: {variance.sum():.2%}")
            
            # Create visualization
            save_path = os.path.join(output_dir, f'{protein_name}_{key}_pca.png')
            plot_2d_embedding(
                pca_result,
                title=f'PCA: {protein_name} - {key}\n({viz_data.shape[0]} samples, {variance.sum():.1%} variance)',
                save_path=save_path
            )
            print(f"  ✓ Saved visualization: {save_path}")
            success = True
            
        except Exception as e:
            print(f"  ✗ Error during visualization: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if success:
        print(f"\n{'='*70}")
        print("✓ Visualization Complete!")
        print(f"{'='*70}")
        print(f"\nResults saved to: {output_dir}/")
        print(f"Protein: {protein_name}")
        return True
    else:
        print(f"\n{'='*70}")
        print("✗ No visualizations could be created")
        print(f"{'='*70}")
        print("\nThis pickle file may not contain suitable representations.")
        print("Expected formats:")
        print("  1. Jayanth's intermediate representation format (dict with 'msa', 'pair', 'single')")
        print("  2. AlphaFold output_dict with embeddings")
        print("  3. Numpy/Torch arrays with shape (n_samples, n_features)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Visualize representations from AlphaFold/OpenFold pickle files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_from_alphafold_pickle.py proteins_pickle/1G1J_*.pickle
  python visualize_from_alphafold_pickle.py proteins_pickle/1UBQ_*.pickle --output_dir my_viz/
  
Available proteins:
  1G1J - Small protein (1.2 MB)
  1UBQ - Ubiquitin (4.6 MB)
  6KWC - Demo protein (22 MB)
  1STM - Large protein (81 MB)
        """
    )
    
    parser.add_argument(
        'pickle_file',
        type=str,
        help='Path to AlphaFold/OpenFold pickle file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/alphafold_viz',
        help='Output directory for visualizations (default: outputs/alphafold_viz)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pickle_file):
        print(f"Error: File not found: {args.pickle_file}")
        sys.exit(1)
    
    # Visualize
    success = visualize_protein(args.pickle_file, args.output_dir)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()

