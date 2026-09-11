"""
Create example visualizations to show what t-SNE and dimensionality reduction outputs look like.
This uses synthetic data to demonstrate the visualization styles.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Set random seed for reproducibility
np.random.seed(42)

def create_output_directory():
    """Create output directory for examples."""
    output_dir = 'example_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_synthetic_protein_data(n_residues=100, n_features=256):
    """
    Generate synthetic protein-like data with some structure.
    Simulates what OpenFold representations might look like.
    """
    # Create 3 distinct clusters (representing different protein regions)
    cluster1 = np.random.randn(30, n_features) * 0.5 + np.array([2, 1] + [0]*(n_features-2))
    cluster2 = np.random.randn(40, n_features) * 0.5 + np.array([-1, 2] + [0]*(n_features-2))
    cluster3 = np.random.randn(30, n_features) * 0.5 + np.array([0, -2] + [0]*(n_features-2))
    
    # Combine
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Add some noise to other dimensions
    data[:, 2:] += np.random.randn(n_residues, n_features-2) * 0.3
    
    # Labels for coloring (e.g., by secondary structure or layer)
    labels = np.array([0]*30 + [1]*40 + [2]*30)
    
    return data, labels

def apply_pca_simple(data, n_components=2):
    """Simple PCA implementation."""
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # Compute covariance matrix
    cov = np.cov(data_centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Convert to real (eigenvalues should be real for covariance matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Sort by eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project data
    pca_result = data_centered @ eigenvectors[:, :n_components]
    
    # Variance explained
    variance_explained = eigenvalues[:n_components] / eigenvalues.sum()
    
    return pca_result, variance_explained

def create_pca_visualization(output_dir):
    """Create PCA visualization example."""
    print("\n[1/6] Creating PCA visualization...")
    
    # Generate data
    data, labels = generate_synthetic_protein_data()
    pca_result, variance = apply_pca_simple(data, n_components=2)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                        c=labels, cmap='viridis', 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({variance[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({variance[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA: MSA Representations (Layer 47)\nExample Output', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Protein Region / Layer', rotation=270, labelpad=20)
    
    # Add annotation
    ax.text(0.02, 0.98, 'Each point = 1 residue\nNearby points = similar features', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: example_pca.png")

def create_tsne_visualization(output_dir):
    """Create t-SNE-like visualization example."""
    print("\n[2/6] Creating t-SNE visualization...")
    
    # Generate data with more pronounced clustering
    data, labels = generate_synthetic_protein_data()
    
    # Simulate t-SNE result (emphasizes local structure)
    # In reality, this would use sklearn's TSNE
    tsne_result = apply_pca_simple(data, n_components=2)[0]
    # Exaggerate cluster separation for t-SNE effect
    tsne_result[:30] = tsne_result[:30] * 1.5 + np.array([3, 2])
    tsne_result[30:70] = tsne_result[30:70] * 1.5 + np.array([-3, 1])
    tsne_result[70:] = tsne_result[70:] * 1.5 + np.array([0, -3])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                        c=labels, cmap='plasma', 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title('t-SNE: MSA Representations (Layer 47)\nExample Output', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Protein Region / Layer', rotation=270, labelpad=20)
    
    # Add annotation
    ax.text(0.02, 0.98, 'Clusters visible!\nSimilar residues group together', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: example_tsne.png")

def create_umap_visualization(output_dir):
    """Create UMAP-like visualization example."""
    print("\n[3/6] Creating UMAP visualization...")
    
    data, labels = generate_synthetic_protein_data()
    
    # Simulate UMAP (balance of local and global structure)
    umap_result = apply_pca_simple(data, n_components=2)[0]
    umap_result[:30] = umap_result[:30] * 1.2 + np.array([2, 1.5])
    umap_result[30:70] = umap_result[30:70] * 1.2 + np.array([-2, 1])
    umap_result[70:] = umap_result[70:] * 1.2 + np.array([0, -2])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], 
                        c=labels, cmap='coolwarm', 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('UMAP Component 1', fontsize=12)
    ax.set_ylabel('UMAP Component 2', fontsize=12)
    ax.set_title('UMAP: MSA Representations (Layer 47)\nExample Output', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Protein Region / Layer', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_umap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: example_umap.png")

def create_comparison_plot(output_dir):
    """Create side-by-side comparison of all methods."""
    print("\n[4/6] Creating method comparison...")
    
    data, labels = generate_synthetic_protein_data()
    
    # Generate results for each method
    pca_result, _ = apply_pca_simple(data, n_components=2)
    
    tsne_result = pca_result.copy()
    tsne_result[:30] = tsne_result[:30] * 1.5 + np.array([3, 2])
    tsne_result[30:70] = tsne_result[30:70] * 1.5 + np.array([-3, 1])
    tsne_result[70:] = tsne_result[70:] * 1.5 + np.array([0, -3])
    
    umap_result = pca_result.copy()
    umap_result[:30] = umap_result[:30] * 1.2 + np.array([2, 1.5])
    umap_result[30:70] = umap_result[30:70] * 1.2 + np.array([-2, 1])
    umap_result[70:] = umap_result[70:] * 1.2 + np.array([0, -2])
    
    # Create subplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    methods = [
        ('PCA', pca_result, 'viridis'),
        ('t-SNE', tsne_result, 'plasma'),
        ('UMAP', umap_result, 'coolwarm')
    ]
    
    for ax, (name, result, cmap) in zip(axes, methods):
        scatter = ax.scatter(result[:, 0], result[:, 1], 
                           c=labels, cmap=cmap, 
                           alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Component 1', fontsize=11)
        ax.set_ylabel('Component 2', fontsize=11)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Add single colorbar
    cbar = plt.colorbar(scatter, ax=axes[-1])
    cbar.set_label('Protein Region', rotation=270, labelpad=20)
    
    plt.suptitle('Method Comparison: MSA Representations (Layer 47)', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: example_comparison.png")

def create_layer_progression(output_dir):
    """Create layer progression visualization."""
    print("\n[5/6] Creating layer progression...")
    
    # Simulate how representations change across layers
    layers = [0, 11, 23, 35, 47]
    
    fig, axes = plt.subplots(1, len(layers), figsize=(20, 4))
    
    for idx, (ax, layer) in enumerate(zip(axes, layers)):
        # Generate data that becomes more structured in later layers
        data, _ = generate_synthetic_protein_data()
        result = apply_pca_simple(data, n_components=2)[0]
        
        # Add convergence effect (later layers more clustered)
        convergence_factor = 1.0 - (idx / len(layers)) * 0.5
        result = result * convergence_factor
        
        # Color by layer
        colors = np.full(100, layer)
        
        scatter = ax.scatter(result[:, 0], result[:, 1], 
                           c=colors, cmap='viridis', vmin=0, vmax=47,
                           alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2', fontsize=10)
        ax.set_title(f'Layer {layer}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add annotation for first and last
        if idx == 0:
            ax.text(0.5, 0.02, 'Scattered', transform=ax.transAxes, 
                   ha='center', fontsize=9, style='italic')
        elif idx == len(layers) - 1:
            ax.text(0.5, 0.02, 'Converged', transform=ax.transAxes, 
                   ha='center', fontsize=9, style='italic')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[-1])
    cbar.set_label('Layer Index', rotation=270, labelpad=20)
    
    plt.suptitle('Layer Progression: MSA Representation Evolution', 
                fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_layer_progression.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: example_layer_progression.png")

def create_pca_variance_plot(output_dir):
    """Create PCA variance explained plot."""
    print("\n[6/6] Creating PCA variance analysis...")
    
    data, _ = generate_synthetic_protein_data()
    
    # Compute PCA for many components
    data_centered = data - np.mean(data, axis=0)
    cov = np.cov(data_centered.T)
    eigenvalues, _ = np.linalg.eig(cov)
    eigenvalues = np.real(eigenvalues)  # Ensure real values
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Variance explained
    variance_explained = eigenvalues / eigenvalues.sum()
    cumulative_variance = np.cumsum(variance_explained)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    n_components = min(50, len(variance_explained))
    ax1.bar(range(1, n_components+1), variance_explained[:n_components], alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Variance Explained by Each Component', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Cumulative variance
    ax2.plot(range(1, n_components+1), cumulative_variance[:n_components], 
            'o-', linewidth=2, markersize=4, color='darkblue')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance', linewidth=2)
    ax2.axhline(y=0.99, color='g', linestyle='--', label='99% variance', linewidth=2)
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_pca_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: example_pca_variance.png")

def create_readme(output_dir):
    """Create README explaining the visualizations."""
    readme_content = """# Example Visualization Outputs

These are example outputs showing what the t-SNE and dimensionality reduction code generates.

## Files Generated:

### 1. `example_pca.png`
- **Method**: Principal Component Analysis (PCA)
- **Shows**: Linear projection onto top 2 principal components
- **Interpretation**: Points represent residues, colors represent protein regions/layers
- **Key Feature**: Shows main directions of variance

### 2. `example_tsne.png`
- **Method**: t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Shows**: Non-linear embedding that reveals clusters
- **Interpretation**: Nearby points = similar representations
- **Key Feature**: Excellent for finding groups of similar residues

### 3. `example_umap.png`
- **Method**: UMAP (Uniform Manifold Approximation and Projection)
- **Shows**: Balanced local and global structure
- **Interpretation**: Combines benefits of PCA and t-SNE
- **Key Feature**: Fast and preserves more structure than t-SNE

### 4. `example_comparison.png`
- **Shows**: All three methods side-by-side
- **Purpose**: Compare how different methods reveal different aspects
- **Use**: Identify robust patterns visible across multiple methods

### 5. `example_layer_progression.png`
- **Shows**: How representations evolve from layer 0 to layer 47
- **Interpretation**: 
  - Early layers (0, 11): More scattered, less organized
  - Late layers (35, 47): More converged, refined features
- **Key Insight**: Watch representations become more structured

### 6. `example_pca_variance.png`
- **Shows**: How much variance each principal component captures
- **Left plot**: Individual component contributions
- **Right plot**: Cumulative variance (how many components needed)
- **Key Info**: Tells you if dimensionality reduction is losing important info

## What These Represent:

Each **point** in the scatter plots represents:
- One **residue** in the protein sequence
- Its **256-dimensional representation** from OpenFold
- **Compressed to 2D** for human visualization

**Colors** represent:
- Different protein regions (alpha helices, beta sheets, loops)
- Or different layers (in progression plots)
- Or any other label you want to explore

## How to Interpret:

- **Nearby points**: Residues with similar representations/features
- **Clusters**: Groups of residues processed similarly by OpenFold
- **Outliers**: Residues with unique features
- **Progression**: How network refines representations through 48 layers

## Real vs. Example Data:

These examples use **synthetic data** to demonstrate the visualization styles.

**Real data** from OpenFold will show:
- Actual protein structural patterns
- Secondary structure clustering
- Domain boundaries
- Functional site groupings

## Next Steps:

Run on your protein:
```bash
python generate_tsne_from_pickle.py your_protein_reps.pt
```

You'll get similar plots but with **real biological patterns**!
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print("\n   ✓ Saved: README.md")

def main():
    """Create all example visualizations."""
    print("="*70)
    print("Creating Example Visualizations")
    print("="*70)
    print("\nGenerating synthetic protein data and creating example plots...")
    
    output_dir = create_output_directory()
    print(f"\nOutput directory: {output_dir}/")
    
    try:
        create_pca_visualization(output_dir)
        create_tsne_visualization(output_dir)
        create_umap_visualization(output_dir)
        create_comparison_plot(output_dir)
        create_layer_progression(output_dir)
        create_pca_variance_plot(output_dir)
        create_readme(output_dir)
        
        print("\n" + "="*70)
        print("✓ All Example Visualizations Created!")
        print("="*70)
        print(f"\nGenerated 6 example plots in: {output_dir}/")
        print("\nFiles created:")
        print("  1. example_pca.png - PCA visualization")
        print("  2. example_tsne.png - t-SNE visualization")
        print("  3. example_umap.png - UMAP visualization")
        print("  4. example_comparison.png - All methods compared")
        print("  5. example_layer_progression.png - Layer evolution")
        print("  6. example_pca_variance.png - Variance analysis")
        print("  7. README.md - Explanation of outputs")
        print("\nView these images to see what your protein visualizations will look like!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

