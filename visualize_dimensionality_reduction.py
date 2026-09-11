"""
Dimensionality reduction visualizations for OpenFold intermediate representations.

This module extends the base visualization utilities with advanced dimensionality 
reduction techniques including t-SNE, PCA, UMAP, and autoencoder-based methods.

Designed to work with the intermediate representations extracted by 
visualize_intermediate_reps_utils.py (implemented by Jayanth).

Author: Shreyas
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Dimensionality reduction libraries
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. PCA and t-SNE will not work.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")


class RepresentationAutoencoder(nn.Module):
    """
    Autoencoder for learning compressed representations of protein features.
    
    Architecture:
        - Encoder: Linear layers with ReLU and optional dropout
        - Latent space: Bottleneck representation
        - Decoder: Symmetric to encoder
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 2, 
                 hidden_dims: List[int] = None, dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space (typically 2 or 3 for visualization)
            hidden_dims: List of hidden layer dimensions. If None, uses [input_dim//2, input_dim//4]
            dropout: Dropout rate for regularization
        """
        super(RepresentationAutoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [max(input_dim // 2, latent_dim * 2), 
                          max(input_dim // 4, latent_dim * 2)]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (symmetric architecture)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation back to input space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


def prepare_representations_for_reduction(representations: Union[torch.Tensor, np.ndarray],
                                         flatten_mode: str = 'residue') -> np.ndarray:
    """
    Prepare representation tensors for dimensionality reduction.
    
    Args:
        representations: Tensor of shape (batch, seq_len, channels) or 
                        (batch, seq_len, seq_len, channels) for pair representations
        flatten_mode: How to flatten the data:
            - 'residue': Each residue becomes a data point
            - 'global': Aggregate over all dimensions to single point per layer
            - 'pairwise': For pair representations, flatten pairs
    
    Returns:
        2D numpy array ready for dimensionality reduction (n_samples, n_features)
    """
    if isinstance(representations, torch.Tensor):
        representations = representations.detach().cpu().numpy()
    
    original_shape = representations.shape
    
    if flatten_mode == 'residue':
        # Flatten to (n_residues, n_features)
        if len(original_shape) == 3:  # (batch, seq_len, channels)
            data = representations.reshape(-1, original_shape[-1])
        elif len(original_shape) == 4:  # (batch, seq1, seq2, channels) - pair rep
            # Take diagonal or flatten?
            # Let's flatten the pair dimensions
            data = representations.reshape(-1, original_shape[-1])
        else:
            raise ValueError(f"Unexpected shape: {original_shape}")
    
    elif flatten_mode == 'global':
        # Aggregate to single vector per sample in batch
        if len(original_shape) == 3:
            data = representations.mean(axis=1)  # Average over residues
        elif len(original_shape) == 4:
            data = representations.mean(axis=(1, 2))  # Average over both pair dimensions
        else:
            raise ValueError(f"Unexpected shape: {original_shape}")
    
    elif flatten_mode == 'pairwise':
        if len(original_shape) != 4:
            raise ValueError(f"Pairwise mode requires 4D tensor, got shape: {original_shape}")
        # Flatten completely
        batch, seq1, seq2, channels = original_shape
        data = representations.reshape(batch * seq1 * seq2, channels)
    
    else:
        raise ValueError(f"Unknown flatten_mode: {flatten_mode}")
    
    return data


def apply_tsne(data: np.ndarray, n_components: int = 2, perplexity: float = 30.0,
               learning_rate: float = 200.0, n_iter: int = 1000, 
               random_state: int = 42, **kwargs) -> np.ndarray:
    """
    Apply t-SNE dimensionality reduction.
    
    Args:
        data: Input data (n_samples, n_features)
        n_components: Target dimensionality (2 or 3)
        perplexity: Perplexity parameter (5-50 typical range)
        learning_rate: Learning rate for optimization
        n_iter: Number of iterations
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters for TSNE
    
    Returns:
        Reduced representation (n_samples, n_components)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for t-SNE")
    
    # Adjust perplexity if necessary
    if data.shape[0] <= perplexity * 3:
        perplexity = max(5, data.shape[0] // 3)
        warnings.warn(f"Adjusting perplexity to {perplexity} due to small sample size")
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                learning_rate=learning_rate, n_iter=n_iter,
                random_state=random_state, **kwargs)
    
    reduced = tsne.fit_transform(data)
    return reduced


def apply_pca(data: np.ndarray, n_components: int = 2, **kwargs) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA dimensionality reduction.
    
    Args:
        data: Input data (n_samples, n_features)
        n_components: Number of principal components
        **kwargs: Additional parameters for PCA
    
    Returns:
        Tuple of (reduced representation, fitted PCA model)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for PCA")
    
    pca = PCA(n_components=n_components, **kwargs)
    reduced = pca.fit_transform(data)
    return reduced, pca


def apply_umap(data: np.ndarray, n_components: int = 2, n_neighbors: int = 15,
               min_dist: float = 0.1, metric: str = 'euclidean',
               random_state: int = 42, **kwargs) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction.
    
    Args:
        data: Input data (n_samples, n_features)
        n_components: Target dimensionality (2 or 3)
        n_neighbors: Number of neighbors to consider
        min_dist: Minimum distance between points in embedding
        metric: Distance metric to use
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters for UMAP
    
    Returns:
        Reduced representation (n_samples, n_components)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP required. Install with: pip install umap-learn")
    
    # Adjust n_neighbors if necessary
    if data.shape[0] <= n_neighbors:
        n_neighbors = max(2, data.shape[0] - 1)
        warnings.warn(f"Adjusting n_neighbors to {n_neighbors} due to small sample size")
    
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, metric=metric, 
                        random_state=random_state, **kwargs)
    
    reduced = reducer.fit_transform(data)
    return reduced


def train_autoencoder(data: np.ndarray, latent_dim: int = 2,
                     hidden_dims: List[int] = None, n_epochs: int = 100,
                     batch_size: int = 32, learning_rate: float = 1e-3,
                     device: str = 'cpu', verbose: bool = True) -> Tuple[RepresentationAutoencoder, np.ndarray]:
    """
    Train an autoencoder for dimensionality reduction.
    
    Args:
        data: Input data (n_samples, n_features)
        latent_dim: Dimension of latent space
        hidden_dims: Hidden layer dimensions
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        verbose: Whether to print training progress
    
    Returns:
        Tuple of (trained model, latent representations)
    """
    input_dim = data.shape[1]
    
    # Create model
    model = RepresentationAutoencoder(input_dim, latent_dim, hidden_dims)
    model = model.to(device)
    
    # Prepare data
    data_tensor = torch.FloatTensor(data)
    dataset = TensorDataset(data_tensor, data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            
            # Forward pass
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}")
    
    # Extract latent representations
    model.eval()
    with torch.no_grad():
        data_tensor = data_tensor.to(device)
        _, latent_reps = model(data_tensor)
        latent_reps = latent_reps.cpu().numpy()
    
    return model, latent_reps


def plot_2d_embedding(embedding: np.ndarray, labels: Optional[np.ndarray] = None,
                     title: str = "2D Embedding", save_path: str = None,
                     cmap: str = 'viridis', figsize: Tuple[int, int] = (10, 8),
                     alpha: float = 0.7, s: int = 50) -> None:
    """
    Plot 2D embedding with optional labels.
    
    Args:
        embedding: 2D embedding (n_samples, 2)
        labels: Optional labels for coloring points
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap for scatter plot
        figsize: Figure size
        alpha: Point transparency
        s: Point size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           c=labels, cmap=cmap, alpha=alpha, s=s)
        plt.colorbar(scatter, ax=ax, label='Label/Layer')
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=alpha, s=s)
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D embedding to {save_path}")
    
    plt.show()


def plot_3d_embedding(embedding: np.ndarray, labels: Optional[np.ndarray] = None,
                     title: str = "3D Embedding", save_path: str = None,
                     cmap: str = 'viridis', figsize: Tuple[int, int] = (12, 9),
                     alpha: float = 0.7, s: int = 50) -> None:
    """
    Plot 3D embedding with optional labels.
    
    Args:
        embedding: 3D embedding (n_samples, 3)
        labels: Optional labels for coloring points
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap for scatter plot
        figsize: Figure size
        alpha: Point transparency
        s: Point size
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                           c=labels, cmap=cmap, alpha=alpha, s=s)
        plt.colorbar(scatter, ax=ax, label='Label/Layer', shrink=0.8)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                  alpha=alpha, s=s)
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_zlabel('Component 3', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D embedding to {save_path}")
    
    plt.show()


def compare_reduction_methods(data: np.ndarray, labels: Optional[np.ndarray] = None,
                              methods: List[str] = ['pca', 'tsne', 'umap'],
                              n_components: int = 2, save_dir: str = None,
                              figsize: Tuple[int, int] = (18, 6)) -> Dict[str, np.ndarray]:
    """
    Compare multiple dimensionality reduction methods side by side.
    
    Args:
        data: Input data (n_samples, n_features)
        labels: Optional labels for coloring
        methods: List of methods to compare ('pca', 'tsne', 'umap', 'autoencoder')
        n_components: Target dimensionality (2 or 3)
        save_dir: Directory to save individual and combined plots
        figsize: Figure size for combined plot
    
    Returns:
        Dictionary mapping method name to reduced representation
    """
    results = {}
    
    # Apply each method
    for method in methods:
        print(f"Applying {method.upper()}...")
        
        if method.lower() == 'pca':
            reduced, _ = apply_pca(data, n_components=n_components)
        elif method.lower() == 'tsne':
            reduced = apply_tsne(data, n_components=n_components)
        elif method.lower() == 'umap':
            reduced = apply_umap(data, n_components=n_components)
        elif method.lower() == 'autoencoder':
            _, reduced = train_autoencoder(data, latent_dim=n_components, verbose=False)
        else:
            warnings.warn(f"Unknown method: {method}, skipping")
            continue
        
        results[method] = reduced
        print(f"{method.upper()} completed. Shape: {reduced.shape}")
    
    # Create comparison plot for 2D
    if n_components == 2 and results:
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        
        if n_methods == 1:
            axes = [axes]
        
        for ax, (method, reduced) in zip(axes, results.items()):
            if labels is not None:
                scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                                   c=labels, cmap='viridis', alpha=0.7, s=30)
            else:
                ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=30)
            
            ax.set_title(f'{method.upper()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.grid(True, alpha=0.3)
        
        if labels is not None:
            # Add colorbar to the last subplot
            plt.colorbar(scatter, ax=axes[-1], label='Layer/Label')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            comparison_path = os.path.join(save_dir, 'method_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {comparison_path}")
        
        plt.show()
    
    # Save individual plots if save_dir provided
    if save_dir:
        for method, reduced in results.items():
            individual_path = os.path.join(save_dir, f'{method}_embedding.png')
            if n_components == 2:
                plot_2d_embedding(reduced, labels, title=f'{method.upper()} Embedding',
                                save_path=individual_path)
            elif n_components == 3:
                plot_3d_embedding(reduced, labels, title=f'{method.upper()} Embedding',
                                save_path=individual_path)
    
    return results


def visualize_layer_progression(layer_representations: Dict[int, torch.Tensor],
                                method: str = 'tsne', n_components: int = 2,
                                save_dir: str = None, rep_type: str = 'msa',
                                flatten_mode: str = 'residue') -> Dict[int, np.ndarray]:
    """
    Visualize how representations evolve across layers using dimensionality reduction.
    
    Args:
        layer_representations: Dict mapping layer index to representation tensor
        method: Reduction method ('pca', 'tsne', 'umap', 'autoencoder')
        n_components: Target dimensionality
        save_dir: Directory to save plots
        rep_type: Type of representation ('msa', 'pair', 'single')
        flatten_mode: How to flatten the data
    
    Returns:
        Dictionary mapping layer index to reduced representation
    """
    # Prepare data
    layer_indices = sorted(layer_representations.keys())
    all_data = []
    all_labels = []
    
    for layer_idx in layer_indices:
        rep = layer_representations[layer_idx]
        data = prepare_representations_for_reduction(rep, flatten_mode=flatten_mode)
        all_data.append(data)
        all_labels.extend([layer_idx] * len(data))
    
    # Concatenate all data
    combined_data = np.vstack(all_data)
    labels = np.array(all_labels)
    
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Layer range: {min(layer_indices)} to {max(layer_indices)}")
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reduced, pca = apply_pca(combined_data, n_components=n_components)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    elif method.lower() == 'tsne':
        reduced = apply_tsne(combined_data, n_components=n_components)
    elif method.lower() == 'umap':
        reduced = apply_umap(combined_data, n_components=n_components)
    elif method.lower() == 'autoencoder':
        _, reduced = train_autoencoder(combined_data, latent_dim=n_components)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    title = f'{rep_type.upper()} Representation Evolution ({method.upper()})'
    save_path = os.path.join(save_dir, f'{rep_type}_evolution_{method}.png') if save_dir else None
    
    if n_components == 2:
        plot_2d_embedding(reduced, labels, title=title, save_path=save_path)
    elif n_components == 3:
        plot_3d_embedding(reduced, labels, title=title, save_path=save_path)
    
    # Split back into per-layer results
    results = {}
    start_idx = 0
    for layer_idx, data in zip(layer_indices, all_data):
        end_idx = start_idx + len(data)
        results[layer_idx] = reduced[start_idx:end_idx]
        start_idx = end_idx
    
    return results


def plot_pca_variance_explained(data: np.ndarray, max_components: int = 50,
                                save_path: str = None, figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Plot cumulative and individual explained variance for PCA.
    
    Args:
        data: Input data (n_samples, n_features)
        max_components: Maximum number of components to consider
        save_path: Path to save figure
        figsize: Figure size
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for PCA")
    
    max_components = min(max_components, min(data.shape))
    
    pca = PCA(n_components=max_components)
    pca.fit(data)
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Individual variance
    ax1.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Variance Explained by Each Component', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
            'o-', linewidth=2, markersize=6)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax2.axhline(y=0.99, color='g', linestyle='--', label='99% variance')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved variance plot to {save_path}")
    
    plt.show()
    
    # Print some statistics
    for threshold in [0.9, 0.95, 0.99]:
        n_components = np.argmax(cumulative_var >= threshold) + 1
        print(f"Components needed for {threshold*100}% variance: {n_components}")


def plot_tsne_perplexity_comparison(data: np.ndarray, labels: Optional[np.ndarray] = None,
                                   perplexities: List[float] = [5, 30, 50, 100],
                                   save_path: str = None,
                                   figsize: Tuple[int, int] = (16, 4)) -> None:
    """
    Compare t-SNE results with different perplexity values.
    
    Args:
        data: Input data (n_samples, n_features)
        labels: Optional labels for coloring
        perplexities: List of perplexity values to try
        save_path: Path to save figure
        figsize: Figure size
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for t-SNE")
    
    n_perplexities = len(perplexities)
    fig, axes = plt.subplots(1, n_perplexities, figsize=figsize)
    
    if n_perplexities == 1:
        axes = [axes]
    
    for ax, perplexity in zip(axes, perplexities):
        # Check if perplexity is valid
        if data.shape[0] <= perplexity * 3:
            ax.text(0.5, 0.5, f'Perplexity {perplexity}\ntoo large for\n{data.shape[0]} samples',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        print(f"Computing t-SNE with perplexity={perplexity}...")
        reduced = apply_tsne(data, n_components=2, perplexity=perplexity)
        
        if labels is not None:
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                               c=labels, cmap='viridis', alpha=0.7, s=30)
        else:
            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=30)
        
        ax.set_title(f'Perplexity = {perplexity}', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
    
    if labels is not None:
        plt.colorbar(scatter, ax=axes[-1], label='Layer/Label')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved perplexity comparison to {save_path}")
    
    plt.show()


# Convenience function to run complete analysis
def run_complete_dimensionality_reduction_analysis(
    representations: Dict[int, torch.Tensor],
    output_dir: str,
    rep_type: str = 'msa',
    methods: List[str] = ['pca', 'tsne', 'umap'],
    flatten_mode: str = 'residue',
    layer_subset: Optional[List[int]] = None
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Run dimensionality reduction analysis on intermediate representations.
    
    Args:
        representations: Dictionary mapping layer indices to representation tensors
        output_dir: Directory to save all outputs
        rep_type: Type of representation ('msa', 'pair', 'single')
        methods: List of reduction methods to apply
        flatten_mode: How to flatten the data
        layer_subset: Optional subset of layers to analyze
    
    Returns:
        Nested dictionary: method -> layer_idx -> reduced_representation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if layer_subset:
        representations = {k: v for k, v in representations.items() if k in layer_subset}
    
    print(f"\n{'='*60}")
    print(f"Running Dimensionality Reduction Analysis")
    print(f"Representation type: {rep_type}")
    print(f"Number of layers: {len(representations)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"{'='*60}\n")
    
    # Prepare data from all layers
    layer_indices = sorted(representations.keys())
    sample_rep = representations[layer_indices[0]]
    sample_data = prepare_representations_for_reduction(sample_rep, flatten_mode=flatten_mode)
    
    print(f"Sample data shape per layer: {sample_data.shape}")
    
    # Run PCA variance analysis
    print("\n--- PCA Variance Analysis ---")
    variance_plot_path = os.path.join(output_dir, f'{rep_type}_pca_variance.png')
    plot_pca_variance_explained(sample_data, max_components=min(50, sample_data.shape[1]),
                                save_path=variance_plot_path)
    
    # Run layer progression analysis for each method
    all_results = {}
    for method in methods:
        print(f"\n--- {method.upper()} Layer Progression ---")
        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        
        results = visualize_layer_progression(
            representations, 
            method=method, 
            n_components=2,
            save_dir=method_dir,
            rep_type=rep_type,
            flatten_mode=flatten_mode
        )
        all_results[method] = results
    
    # Compare methods on single layer (e.g., final layer)
    print("\n--- Method Comparison (Final Layer) ---")
    final_layer_idx = max(layer_indices)
    final_rep = representations[final_layer_idx]
    final_data = prepare_representations_for_reduction(final_rep, flatten_mode=flatten_mode)
    
    comparison_dir = os.path.join(output_dir, 'method_comparison')
    compare_reduction_methods(final_data, methods=methods, n_components=2, 
                              save_dir=comparison_dir)
    
    # Save results
    results_file = os.path.join(output_dir, f'{rep_type}_reduction_results.npz')
    np.savez(results_file, **{f'{method}_layer_{k}': v 
                              for method, layer_dict in all_results.items() 
                              for k, v in layer_dict.items()})
    print(f"\nSaved all results to {results_file}")
    
    print(f"\n{'='*60}")
    print(f"Analysis Complete! Results saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return all_results


if __name__ == "__main__":
    print("Dimensionality Reduction Utilities for OpenFold Representations")
    print("="*70)
    print("\nThis module provides:")
    print("  - t-SNE, PCA, UMAP dimensionality reduction")
    print("  - Autoencoder-based representation learning")
    print("  - Comparative visualization tools")
    print("  - Layer progression analysis")
    print("\nSee demo notebooks for usage examples.")
    print("="*70)


