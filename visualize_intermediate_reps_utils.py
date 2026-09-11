"""
Intermediate representation visualization for OpenFold.

Extract and visualize MSA, Pair, and Structure representations from OpenFold's 
48-layer network. Supports layer-by-layer analysis and various visualization modes.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os


class IntermediateRepresentations:
    def __init__(self):
        self.msa_reps = {}
        self.pair_reps = {}
        self.single_reps = {}
        self.enabled = False
    
    def clear(self):
        self.msa_reps = {}
        self.pair_reps = {}
        self.single_reps = {}
    
    def enable(self):
        self.enabled = True
        self.clear()
    
    def disable(self):
        self.enabled = False
        self.clear()


INTERMEDIATE_REPS = IntermediateRepresentations()


def extract_msa_representations(model_output, layer_indices=None):
    """Extract MSA representations from model output or stored layers."""
    if model_output is not None and "msa" in model_output:
        return {-1: model_output["msa"]}
    
    if not INTERMEDIATE_REPS.msa_reps:
        raise ValueError("No MSA representations found. Enable hooks first.")
    
    if layer_indices is None:
        return INTERMEDIATE_REPS.msa_reps.copy()
    
    return {idx: INTERMEDIATE_REPS.msa_reps[idx] 
            for idx in layer_indices 
            if idx in INTERMEDIATE_REPS.msa_reps}


def extract_pair_representations(model_output, layer_indices=None):
    """Extract Pair representations from model output or stored layers."""
    if model_output is not None and "pair" in model_output:
        return {-1: model_output["pair"]}
    
    if not INTERMEDIATE_REPS.pair_reps:
        raise ValueError("No Pair representations found. Enable hooks first.")
    
    if layer_indices is None:
        return INTERMEDIATE_REPS.pair_reps.copy()
    
    return {idx: INTERMEDIATE_REPS.pair_reps[idx] 
            for idx in layer_indices 
            if idx in INTERMEDIATE_REPS.pair_reps}


def extract_structure_representations(model_output):
    """Extract structure module outputs from model."""
    if model_output is None or "sm" not in model_output:
        raise ValueError("No structure module outputs found.")
    
    sm_output = model_output["sm"]
    result = {
        'backbone_frames': sm_output['frames'],
        'angles': sm_output['angles'],
        'positions': sm_output['positions'],
        'single': sm_output.get('single', None),
    }
    
    if "final_atom_positions" in model_output:
        result['final_atom_positions'] = model_output['final_atom_positions']
    
    return result


def plot_msa_representation_heatmap(msa_tensor, layer_idx, save_path, 
                                   aggregate_method='mean', cmap='viridis',
                                   highlight_residue=None, custom_ticks=None):
    msa_2d = aggregate_channels(msa_tensor, method=aggregate_method, axis=-1)
    n_seq, n_res = msa_2d.shape
    
    fig, ax = plt.subplots(figsize=(max(10, n_res // 10), max(6, n_seq // 20)))
    
    im = ax.imshow(msa_2d, aspect='auto', cmap=cmap, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{aggregate_method.capitalize()} activation', 
                   rotation=270, labelpad=20)
    
    # Set custom tick labels for select residues
    if custom_ticks is None:
        custom_ticks = [0, 25, 50, 75]
    
    # Filter ticks that are within the sequence length
    valid_ticks = [tick for tick in custom_ticks if tick < n_res]
    ax.set_xticks(valid_ticks)
    ax.set_xticklabels([str(tick) for tick in valid_ticks])
    
    # Highlight specific residue position
    if highlight_residue is not None and highlight_residue < n_res:
        ax.axvline(x=highlight_residue, color='red', linewidth=2, alpha=0.8, 
                   linestyle='--', label=f'Residue {highlight_residue}')
        ax.legend()
    
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('MSA Sequence', fontsize=12)
    ax.set_title(f'MSA Representation - Layer {layer_idx}\n'
                 f'Aggregation: {aggregate_method}',
                 fontsize=14, fontweight='bold')
    ax.grid(False)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved MSA heatmap to {save_path}")
    
    return fig


def plot_pair_representation_heatmap(pair_tensor, layer_idx, save_path,
                                    aggregate_method='mean', cmap='RdBu_r',
                                    contact_map=None, show_correlation=False):
    pair_2d = aggregate_channels(pair_tensor, method=aggregate_method, axis=-1)
    n_res = pair_2d.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 9))
    
    if 'RdBu' in cmap:
        vmax = np.max(np.abs(pair_2d))
        vmin = -vmax
    else:
        vmin, vmax = None, None
    
    im = ax.imshow(pair_2d, aspect='equal', cmap=cmap, 
                   interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.plot([0, n_res-1], [0, n_res-1], 'k--', alpha=0.3, linewidth=0.5)
    
    # Overlay contact map if provided
    if contact_map is not None:
        if contact_map.shape != pair_2d.shape:
            print(f"Warning: Contact map shape {contact_map.shape} doesn't match pair representation {pair_2d.shape}")
        else:
            # Create a binary mask for contacts
            contact_mask = contact_map > 0.5  # Threshold for contacts
            # Overlay contacts as black circles
            contact_positions = np.where(contact_mask)
            for i, j in zip(contact_positions[0], contact_positions[1]):
                ax.plot(j, i, 'ko', markersize=2, alpha=0.7)
    
    # Compute and display correlation if requested
    correlation_text = ""
    if show_correlation and contact_map is not None:
        if contact_map.shape == pair_2d.shape:
            # Flatten both arrays for correlation computation
            pair_flat = pair_2d.flatten()
            contact_flat = contact_map.flatten()
            
            # Compute Pearson correlation
            correlation = np.corrcoef(pair_flat, contact_flat)[0, 1]
            correlation_text = f"Correlation: {correlation:.3f}"
            
            # Add correlation text to the plot
            ax.text(0.02, 0.98, correlation_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{aggregate_method.capitalize()} activation', 
                   rotation=270, labelpad=20)
    
    ax.set_xlabel('Residue i', fontsize=12)
    ax.set_ylabel('Residue j', fontsize=12)
    
    title = f'Pair Representation - Layer {layer_idx}\nAggregation: {aggregate_method}'
    if contact_map is not None:
        title += '\n(Black dots: contacts)'
    if correlation_text:
        title += f'\n{correlation_text}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(False)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Pair heatmap to {save_path}")
    
    return fig


def plot_representation_evolution(tensors_across_layers, residue_idx, save_path, 
                                 rep_type='msa', multiple_residues=None, show_confidence=True,
                                 show_differences=False):
    layer_indices = sorted(tensors_across_layers.keys())
    
    if multiple_residues is None:
        multiple_residues = [residue_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(multiple_residues)))
    
    for i, res_idx in enumerate(multiple_residues):
        magnitudes = []
        confidence_intervals = []
    
    for layer_idx in layer_indices:
        tensor = tensors_across_layers[layer_idx]
        
        if rep_type == 'msa':
                # Get representation for this residue across all sequences
                residue_reps = tensor[:, res_idx, :]  # (n_seq, c_m)
                
                # Compute L2 norm for each sequence
                if isinstance(residue_reps, torch.Tensor):
                    seq_magnitudes = torch.norm(residue_reps, dim=1).numpy()
                else:
                    seq_magnitudes = np.linalg.norm(residue_reps, axis=1)
                
                mean_mag = np.mean(seq_magnitudes)
                std_mag = np.std(seq_magnitudes)
                
        elif rep_type == 'pair':
            # Average over all pairings for this residue
                residue_rep = tensor[res_idx, :, :].mean(dim=0)
                
                if isinstance(residue_rep, torch.Tensor):
                    mean_mag = torch.norm(residue_rep).item()
                    std_mag = 0.0  # No confidence for single value
                else:
                    mean_mag = np.linalg.norm(residue_rep)
                    std_mag = 0.0
                
                seq_magnitudes = [mean_mag]
        else:
            raise ValueError(f"Unknown rep_type: {rep_type}. Must be 'msa' or 'pair'")
            
            magnitudes.append(mean_mag)
            confidence_intervals.append(std_mag)
        
        # Plot main line
        ax.plot(layer_indices, magnitudes, marker='o', linewidth=2, markersize=8,
                color=colors[i], label=f'Residue {res_idx}')
        
        # Add confidence intervals if requested and available
        if show_confidence and rep_type == 'msa' and np.any(confidence_intervals):
            lower_bound = np.array(magnitudes) - np.array(confidence_intervals)
            upper_bound = np.array(magnitudes) + np.array(confidence_intervals)
            ax.fill_between(layer_indices, lower_bound, upper_bound, 
                           alpha=0.2, color=colors[i])
    
    # Add layer difference plot if requested
    if show_differences and len(layer_indices) > 1:
        ax2 = ax.twinx()
        diff_magnitudes = np.diff(magnitudes)
        ax2.plot(layer_indices[1:], diff_magnitudes, '--', alpha=0.7, 
                color='red', linewidth=1, label='Layer Difference')
        ax2.set_ylabel('Magnitude Change Between Layers', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Representation Magnitude (L2 Norm)', fontsize=12)
    
    if len(multiple_residues) == 1:
        ax.set_title(f'{rep_type.upper()} Representation Evolution - Residue {residue_idx}',
                     fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'{rep_type.upper()} Representation Evolution - Multiple Residues',
                     fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved evolution plot to {save_path}")
    
    return fig


def plot_channel_specific_heatmap(tensor, layer_idx, channel_idx, save_path, 
                                  rep_type='msa', cmap='viridis'):
    """
    Visualize a specific channel of a representation.
    
    Args:
        tensor: MSA tensor (n_seq, n_res, c_m) or Pair tensor (n_res, n_res, c_z)
        layer_idx: Which layer this is from
        channel_idx: Which channel to visualize
        save_path: Where to save the figure
        rep_type: 'msa' or 'pair'
        cmap: Matplotlib colormap name
        
    Returns:
        Matplotlib figure object
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    channel_data = tensor[..., channel_idx]
    
    if rep_type == 'msa':
        n_seq, n_res = channel_data.shape
        fig, ax = plt.subplots(figsize=(max(10, n_res // 10), max(6, n_seq // 20)))
        ylabel = 'MSA Sequence'
    elif rep_type == 'pair':
        n_res = channel_data.shape[0]
        fig, ax = plt.subplots(figsize=(10, 9))
        ylabel = 'Residue j'
    else:
        raise ValueError(f"Unknown rep_type: {rep_type}. Must be 'msa' or 'pair'")
    
    im = ax.imshow(channel_data, aspect='auto' if rep_type == 'msa' else 'equal',
                   cmap=cmap, interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activation', rotation=270, labelpad=20)
    
    ax.set_xlabel('Residue Position' if rep_type == 'msa' else 'Residue i', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{rep_type.upper()} Representation - Layer {layer_idx}, Channel {channel_idx}',
                 fontsize=14, fontweight='bold')
    
    ax.grid(False)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved channel-specific heatmap to {save_path}")
    
    return fig


def aggregate_channels(tensor, method='mean', axis=-1):
    """
    Aggregate over channel dimension of a tensor.
    
    Args:
        tensor: Input tensor with channel dimension
        method: 'mean', 'max', 'norm', or 'sum'
        axis: Which axis is the channel dimension (default: -1)
        
    Returns:
        Aggregated tensor as numpy array
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    if method == 'mean':
        return np.mean(tensor, axis=axis)
    elif method == 'max':
        return np.max(tensor, axis=axis)
    elif method == 'norm':
        return np.linalg.norm(tensor, axis=axis)
    elif method == 'sum':
        return np.sum(tensor, axis=axis)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def save_intermediate_reps_to_disk(intermediate_reps, output_dir, protein_name):
    """
    Save extracted intermediate representations to disk for later analysis.
    
    Args:
        intermediate_reps: Dictionary of extracted representations
                          Keys should be 'msa', 'pair', 'structure', etc.
                          Values are either tensors or dicts of tensors
        output_dir: Directory to save to
        protein_name: Name of the protein (for filename)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, f"{protein_name}_intermediate_reps.pt")
    
    # Convert tensors to CPU before saving
    def to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        elif isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_cpu(item) for item in obj]
        else:
            return obj
    
    intermediate_reps_cpu = to_cpu(intermediate_reps)
    
    torch.save(intermediate_reps_cpu, save_path)
    print(f"Saved intermediate representations to {save_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"{protein_name}_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Protein: {protein_name}\n")
        f.write(f"Representations saved: {list(intermediate_reps.keys())}\n")
        for key, value in intermediate_reps.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for layer_idx, tensor in value.items():
                    if isinstance(tensor, torch.Tensor):
                        f.write(f"  Layer {layer_idx}: shape {tuple(tensor.shape)}\n")
            elif isinstance(value, torch.Tensor):
                f.write(f"{key}: shape {tuple(value.shape)}\n")
    
    print(f"Saved metadata to {metadata_path}")


def load_intermediate_reps_from_disk(input_path):
    """
    Load previously saved intermediate representations.
    
    Args:
        input_path: Path to saved representations (.pt file)
        
    Returns:
        Dictionary of representations
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    intermediate_reps = torch.load(input_path, map_location='cpu')
    
    print(f"Loaded intermediate representations from {input_path}")
    print(f"Available keys: {list(intermediate_reps.keys())}")
    
    return intermediate_reps


def register_evoformer_hooks(model):
    """
    Register forward hooks on Evoformer blocks to capture intermediate representations.
    
    This function attaches hooks to the model's Evoformer blocks to store
    MSA and Pair representations after each layer.
    
    Args:
        model: OpenFold AlphaFold model instance
        
    Usage:
        INTERMEDIATE_REPS.enable()
        register_evoformer_hooks(model)
        output = model(batch)
        msa_reps = extract_msa_representations(None)
        pair_reps = extract_pair_representations(None)
    """
    hooks = []
    
    def make_hook(layer_idx):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input_tuple, output_tuple):
            if INTERMEDIATE_REPS.enabled:
                m, z = output_tuple
                # Store copies to avoid issues with in-place operations
                INTERMEDIATE_REPS.msa_reps[layer_idx] = m.detach().clone()
                INTERMEDIATE_REPS.pair_reps[layer_idx] = z.detach().clone()
        return hook_fn
    
    # Register hooks on each Evoformer block
    if hasattr(model, 'evoformer') and hasattr(model.evoformer, 'blocks'):
        for idx, block in enumerate(model.evoformer.blocks):
            hook = block.register_forward_hook(make_hook(idx))
            hooks.append(hook)
        print(f"Registered {len(hooks)} forward hooks on Evoformer blocks")
    else:
        raise AttributeError(
            "Model does not have expected structure. "
            "Expected model.evoformer.blocks"
        )
    
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
    print(f"Removed {len(hooks)} hooks")


def generate_mock_contact_map(n_res, contact_probability=0.1, seed=None):
    """Generate a mock contact map for testing pair representation visualizations."""
    if seed is not None:
        np.random.seed(seed)
    
    contact_map = np.zeros((n_res, n_res))
    
    # Add some random contacts
    for i in range(n_res):
        for j in range(i + 5, n_res):  # Skip nearby residues (sequence distance > 4)
            if np.random.random() < contact_probability:
                contact_map[i, j] = 1.0
                contact_map[j, i] = 1.0  # Make symmetric
    
    return contact_map


def stratified_layer_sampling(n_layers=48, strategy='uniform', n_samples=None, seed=None):
    """Generate stratified sampling of layers for multi-layer analysis."""
    if seed is not None:
        np.random.seed(seed)
    
    if strategy == 'uniform':
        if n_samples is None:
            n_samples = 8
        step = n_layers // (n_samples - 1) if n_samples > 1 else n_layers
        layers = list(range(0, n_layers, step))
        if layers[-1] != n_layers - 1:
            layers.append(n_layers - 1)
        return layers
    
    elif strategy == 'grouped':
        # Sample from early, middle, and late layers
        early = [0, 4, 8, 12]
        middle = [16, 20, 24, 28, 32, 36]
        late = [40, 44, 47]
        return early + middle + late
    
    elif strategy == 'logarithmic':
        # More samples in early layers, fewer later
        layers = [0, 1, 2, 4, 8, 12, 16, 24, 32, 40, 47]
        return layers
    
    elif strategy == 'dense':
        # Max out sampling: ~75% of layers (36 out of 48)
        if n_samples is None:
            n_samples = int(n_layers * 0.75)
        
        # Always include first and last
        layers = [0, n_layers - 1]
        
        # Sample the rest uniformly
        remaining = n_samples - 2
        step = (n_layers - 2) // remaining if remaining > 0 else 1
        middle_layers = list(range(1, n_layers - 1, step))[:remaining]
        
        return sorted(list(set(layers + middle_layers)))
    
    elif strategy == 'random':
        # Randomized sampling with high density
        if n_samples is None:
            n_samples = int(n_layers * 0.70)  # Sample 70% randomly
        
        # Always include first and last
        layers = [0, n_layers - 1]
        
        # Randomly sample the rest
        remaining = n_samples - 2
        available = list(range(1, n_layers - 1))
        sampled = np.random.choice(available, size=min(remaining, len(available)), replace=False)
        
        return sorted(layers + list(sampled))
    
    elif strategy == 'all':
        return list(range(n_layers))
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def plot_multilayer_evolution(layer_representations, residue_indices, save_path, 
                               rep_type='msa', layer_sampling='uniform'):
    """Visualize how representations evolve across all 48 layers."""
    
    if isinstance(layer_sampling, str):
        sampled_layers = stratified_layer_sampling(
            n_layers=len(layer_representations), 
            strategy=layer_sampling
        )
    else:
        sampled_layers = layer_sampling
    
    # Filter to only sampled layers
    layer_dict = {k: v for k, v in layer_representations.items() if k in sampled_layers}
    layer_indices = sorted(layer_dict.keys())
    
    if not isinstance(residue_indices, list):
        residue_indices = [residue_indices]
    
    # Adaptive figure sizing based on number of tracked residues
    figsize = (20, 14) if len(residue_indices) > 5 else (14, 10)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Select colormap based on number of residues
    if len(residue_indices) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(residue_indices)))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, len(residue_indices)))
    
    # Top plot: Magnitude evolution
    for idx, res_idx in enumerate(residue_indices):
        magnitudes = []
        
        for layer_idx in layer_indices:
            tensor = layer_dict[layer_idx]
            
            if rep_type == 'msa':
                residue_reps = tensor[:, res_idx, :]
                if isinstance(residue_reps, torch.Tensor):
                    seq_magnitudes = torch.norm(residue_reps, dim=1).numpy()
                else:
                    seq_magnitudes = np.linalg.norm(residue_reps, axis=1)
                mean_mag = np.mean(seq_magnitudes)
            elif rep_type == 'pair':
                residue_rep = tensor[res_idx, :, :].mean(dim=0)
                if isinstance(residue_rep, torch.Tensor):
                    mean_mag = torch.norm(residue_rep).item()
                else:
                    mean_mag = np.linalg.norm(residue_rep)
            
            magnitudes.append(mean_mag)
        
        ax1.plot(layer_indices, magnitudes, marker='o', linewidth=2, markersize=6,
                color=colors[idx], label=f'Residue {res_idx}')
    
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Representation Magnitude (L2 Norm)', fontsize=12)
    ax1.set_title(f'{rep_type.upper()} Representation Evolution Across All Layers',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Bottom plot: Layer-to-layer changes
    if len(layer_indices) > 1:
        for idx, res_idx in enumerate(residue_indices):
            magnitudes = []
            
            for layer_idx in layer_indices:
                tensor = layer_dict[layer_idx]
                
                if rep_type == 'msa':
                    residue_reps = tensor[:, res_idx, :]
                    if isinstance(residue_reps, torch.Tensor):
                        seq_magnitudes = torch.norm(residue_reps, dim=1).numpy()
                    else:
                        seq_magnitudes = np.linalg.norm(residue_reps, axis=1)
                    mean_mag = np.mean(seq_magnitudes)
                elif rep_type == 'pair':
                    residue_rep = tensor[res_idx, :, :].mean(dim=0)
                    if isinstance(residue_rep, torch.Tensor):
                        mean_mag = torch.norm(residue_rep).item()
                    else:
                        mean_mag = np.linalg.norm(residue_rep)
                
                magnitudes.append(mean_mag)
            
            changes = np.diff(magnitudes)
            ax2.plot(layer_indices[1:], changes, marker='s', linewidth=2, markersize=5,
                    color=colors[idx], label=f'Residue {res_idx}', alpha=0.7)
    
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Magnitude Change', fontsize=12)
    ax2.set_title('Layer-to-Layer Representation Changes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.legend(loc='best')
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved multi-layer evolution plot to {save_path}")
    
    return fig


def plot_stratified_layer_comparison(layer_representations, layer_indices, save_path,
                                     rep_type='msa', aggregate_method='mean'):
    """Create side-by-side comparison of representations at different layers."""
    
    n_layers = len(layer_indices)
    
    # Adaptive grid sizing based on layer count
    if n_layers <= 9:
        cols = 3
        cell_size = 6
    elif n_layers <= 25:
        cols = 5
        cell_size = 5
    else:  # For complete 48-layer analysis
        cols = 8
        cell_size = 4
    
    rows = (n_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cell_size*cols, cell_size*rows))
    if n_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    for idx, layer_idx in enumerate(layer_indices):
        ax = axes[idx]
        tensor = layer_representations[layer_idx]
        
        if rep_type == 'msa':
            data_2d = aggregate_channels(tensor, method=aggregate_method, axis=-1)
            im = ax.imshow(data_2d, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_ylabel('MSA Sequence', fontsize=10)
            ax.set_xlabel('Residue Position', fontsize=10)
        elif rep_type == 'pair':
            data_2d = aggregate_channels(tensor, method=aggregate_method, axis=-1)
            vmax = np.max(np.abs(data_2d))
            im = ax.imshow(data_2d, aspect='equal', cmap='RdBu_r', 
                          interpolation='nearest', vmin=-vmax, vmax=vmax)
            ax.set_ylabel('Residue j', fontsize=10)
            ax.set_xlabel('Residue i', fontsize=10)
        
        ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide extra subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Stratified {rep_type.upper()} Representation Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved stratified layer comparison to {save_path}")
    
    return fig


def plot_layer_convergence_analysis(layer_representations, save_path, rep_type='msa'):
    """Analyze when representations converge across layers."""
    
    layer_indices = sorted(layer_representations.keys())
    
    # Compute layer-to-layer similarity
    similarities = []
    for i in range(len(layer_indices) - 1):
        curr_layer = layer_representations[layer_indices[i]]
        next_layer = layer_representations[layer_indices[i + 1]]
        
        # Flatten and compute correlation
        if isinstance(curr_layer, torch.Tensor):
            curr_flat = curr_layer.detach().cpu().numpy().flatten()
            next_flat = next_layer.detach().cpu().numpy().flatten()
        else:
            curr_flat = curr_layer.flatten()
            next_flat = next_layer.flatten()
        
        correlation = np.corrcoef(curr_flat, next_flat)[0, 1]
        similarities.append(correlation)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot similarities
    ax1.plot(layer_indices[1:], similarities, marker='o', linewidth=2, markersize=6, color='blue')
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Layer-to-Layer Correlation', fontsize=12)
    ax1.set_title(f'{rep_type.upper()} Representation Convergence Analysis',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='High similarity threshold')
    ax1.legend()
    
    # Plot rate of change
    if len(similarities) > 1:
        rate_of_change = np.diff(similarities)
        ax2.plot(layer_indices[2:], rate_of_change, marker='s', linewidth=2, 
                markersize=5, color='green')
        ax2.set_xlabel('Layer Index', fontsize=12)
        ax2.set_ylabel('Rate of Change in Similarity', fontsize=12)
        ax2.set_title('Convergence Rate', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence analysis to {save_path}")
    
    return fig


def plot_structure_module_evolution(structure_outputs, save_path):
    """Visualize structure module evolution - backbone frames, angles, positions."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract structure data
    backbone_frames = structure_outputs['backbone_frames']  # (n_recycles, n_res, 7)
    angles = structure_outputs['angles']  # (n_recycles, n_res, 7, 2)
    positions = structure_outputs['positions']  # (n_recycles, n_res, 14, 3)
    
    n_recycles = backbone_frames.shape[0]
    n_res = backbone_frames.shape[1]
    
    # Convert to numpy
    if isinstance(backbone_frames, torch.Tensor):
        backbone_frames = backbone_frames.detach().cpu().numpy()
        angles = angles.detach().cpu().numpy()
        positions = positions.detach().cpu().numpy()
    
    # Plot 1: Backbone frame norms across recycles
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(n_recycles):
        frame_norms = np.linalg.norm(backbone_frames[i], axis=-1)
        ax1.plot(range(n_res), frame_norms, label=f'Recycle {i}', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Residue', fontsize=12)
    ax1.set_ylabel('Frame Norm', fontsize=12)
    ax1.set_title('Backbone Frame Evolution Across Recycles', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Angle distributions (torsion angles)
    ax2 = fig.add_subplot(gs[1, 0])
    final_angles = angles[-1, :, :, 0]  # Last recycle, first angle component
    im2 = ax2.imshow(final_angles.T, aspect='auto', cmap='twilight', interpolation='nearest')
    ax2.set_xlabel('Residue', fontsize=10)
    ax2.set_ylabel('Angle Type', fontsize=10)
    ax2.set_title('Torsion Angles (Final Recycle)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Plot 3: Position RMSD across recycles
    ax3 = fig.add_subplot(gs[1, 1])
    rmsd_values = []
    for i in range(1, n_recycles):
        diff = positions[i] - positions[i-1]
        rmsd = np.sqrt(np.mean(diff**2))
        rmsd_values.append(rmsd)
    ax3.plot(range(1, n_recycles), rmsd_values, marker='o', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Recycle Step', fontsize=12)
    ax3.set_ylabel('RMSD from Previous', fontsize=12)
    ax3.set_title('Position Convergence', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: CA atom positions (3D trajectory)
    ax4 = fig.add_subplot(gs[1, 2], projection='3d')
    ca_positions = positions[:, :, 1, :]  # CA is typically atom 1
    colors_3d = plt.cm.viridis(np.linspace(0, 1, n_recycles))
    for i in range(n_recycles):
        ax4.plot(ca_positions[i, :, 0], ca_positions[i, :, 1], ca_positions[i, :, 2], 
                alpha=0.6, linewidth=1, color=colors_3d[i], label=f'Recycle {i}')
    ax4.set_xlabel('X', fontsize=10)
    ax4.set_ylabel('Y', fontsize=10)
    ax4.set_zlabel('Z', fontsize=10)
    ax4.set_title('CA Atom Trajectory', fontsize=12, fontweight='bold')
    
    # Plot 5: Per-residue displacement
    ax5 = fig.add_subplot(gs[2, :2])
    final_pos = positions[-1, :, 1, :]
    initial_pos = positions[0, :, 1, :]
    displacement = np.linalg.norm(final_pos - initial_pos, axis=-1)
    ax5.bar(range(n_res), displacement, color='steelblue', alpha=0.7)
    ax5.set_xlabel('Residue', fontsize=12)
    ax5.set_ylabel('Displacement (Å)', fontsize=12)
    ax5.set_title('Per-Residue Displacement (Initial to Final)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Angle change heatmap
    ax6 = fig.add_subplot(gs[2, 2])
    angle_changes = np.abs(angles[-1] - angles[0])[:, :, 0]
    im6 = ax6.imshow(angle_changes.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax6.set_xlabel('Residue', fontsize=10)
    ax6.set_ylabel('Angle Type', fontsize=10)
    ax6.set_title('Angle Changes', fontsize=12, fontweight='bold')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    fig.suptitle('Structure Module Evolution Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved structure module evolution to {save_path}")
    
    return fig


def compute_layer_importance_metrics(layer_representations, metric='variance'):
    """Compute importance metrics for each layer."""
    
    layer_indices = sorted(layer_representations.keys())
    importance_scores = {}
    
    for layer_idx in layer_indices:
        tensor = layer_representations[layer_idx]
        
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        if metric == 'variance':
            # Higher variance = more information
            score = np.var(tensor)
        elif metric == 'entropy':
            # Information entropy
            flat = tensor.flatten()
            hist, _ = np.histogram(flat, bins=50, density=True)
            hist = hist[hist > 0]
            score = -np.sum(hist * np.log(hist + 1e-10))
        elif metric == 'sparsity':
            # Sparsity (lower = more sparse)
            score = np.mean(np.abs(tensor) > 0.01)
        elif metric == 'norm':
            # L2 norm
            score = np.linalg.norm(tensor)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        importance_scores[layer_idx] = score
    
    return importance_scores


def plot_layer_importance_ranking(layer_representations, save_path, metrics=['variance', 'entropy', 'norm']):
    """Rank and visualize layer importance using multiple metrics."""
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        importance = compute_layer_importance_metrics(layer_representations, metric=metric)
        layers = list(importance.keys())
        scores = list(importance.values())
        
        # Normalize scores for comparison
        scores_norm = (np.array(scores) - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
        
        ax = axes[idx]
        bars = ax.bar(layers, scores_norm, color=plt.cm.viridis(scores_norm), alpha=0.8)
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel(f'{metric.capitalize()} Score (Normalized)', fontsize=12)
        ax.set_title(f'Layer Importance Ranking - {metric.capitalize()} Metric', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight top 5 layers
        top_5_indices = np.argsort(scores)[-5:]
        for i in top_5_indices:
            ax.text(layers[i], scores_norm[i], f'#{len(scores)-list(np.argsort(scores)).index(i)}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    fig.suptitle('Multi-Metric Layer Importance Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved layer importance ranking to {save_path}")
    
    return fig


def plot_residue_feature_analysis(tensor, residue_idx, save_path, rep_type='msa'):
    """Detailed per-residue feature analysis."""
    
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    if rep_type == 'msa':
        # MSA: (n_seq, n_res, c_m)
        residue_data = tensor[:, residue_idx, :]  # (n_seq, c_m)
        
        # Plot 1: Feature heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        im1 = ax1.imshow(residue_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_xlabel('MSA Sequence', fontsize=12)
        ax1.set_ylabel('Feature Channel', fontsize=12)
        ax1.set_title(f'Residue {residue_idx} Feature Map', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Plot 2: Feature distribution
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(residue_data.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Feature Value', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Feature Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Channel-wise statistics
        ax3 = fig.add_subplot(gs[1, :])
        channel_means = np.mean(residue_data, axis=0)
        channel_stds = np.std(residue_data, axis=0)
        channels = np.arange(len(channel_means))
        ax3.plot(channels, channel_means, linewidth=2, label='Mean', color='blue')
        ax3.fill_between(channels, channel_means - channel_stds, channel_means + channel_stds,
                        alpha=0.3, color='blue', label='±1 Std Dev')
        ax3.set_xlabel('Channel Index', fontsize=12)
        ax3.set_ylabel('Feature Value', fontsize=12)
        ax3.set_title('Per-Channel Statistics', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Top channels by variance
        ax4 = fig.add_subplot(gs[2, 0])
        channel_vars = np.var(residue_data, axis=0)
        top_10 = np.argsort(channel_vars)[-10:]
        ax4.barh(range(10), channel_vars[top_10], color='coral', alpha=0.7)
        ax4.set_yticks(range(10))
        ax4.set_yticklabels([f'Ch {i}' for i in top_10])
        ax4.set_xlabel('Variance', fontsize=11)
        ax4.set_title('Top 10 Channels by Variance', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Plot 5: Sequence correlation
        ax5 = fig.add_subplot(gs[2, 1])
        seq_corr = np.corrcoef(residue_data)
        im5 = ax5.imshow(seq_corr, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        ax5.set_xlabel('MSA Sequence', fontsize=11)
        ax5.set_ylabel('MSA Sequence', fontsize=11)
        ax5.set_title('Sequence Correlation', fontsize=12, fontweight='bold')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        # Plot 6: Channel activation patterns
        ax6 = fig.add_subplot(gs[2, 2])
        active_channels = np.mean(np.abs(residue_data) > 0.1, axis=0)
        ax6.plot(channels, active_channels, linewidth=2, color='green')
        ax6.fill_between(channels, 0, active_channels, alpha=0.3, color='green')
        ax6.set_xlabel('Channel Index', fontsize=11)
        ax6.set_ylabel('Activation Rate', fontsize=11)
        ax6.set_title('Channel Activation Patterns', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    elif rep_type == 'pair':
        # Pair: (n_res, n_res, c_z)
        residue_pairs = tensor[residue_idx, :, :]  # (n_res, c_z)
        
        # Similar plots adapted for pair representations
        ax1 = fig.add_subplot(gs[0, :2])
        im1 = ax1.imshow(residue_pairs.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax1.set_xlabel('Partner Residue', fontsize=12)
        ax1.set_ylabel('Feature Channel', fontsize=12)
        ax1.set_title(f'Residue {residue_idx} Pair Feature Map', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Additional pair-specific plots...
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(residue_pairs.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Feature Value', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Feature Distribution', fontsize=12, fontweight='bold')
        
    fig.suptitle(f'Residue {residue_idx} Feature Analysis ({rep_type.upper()})',
                fontsize=16, fontweight='bold', y=0.998)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved residue feature analysis to {save_path}")
    
    return fig


def plot_layer_clustering_dendrogram(layer_representations, save_path, method='ward'):
    """Hierarchical clustering of layers based on representation similarity."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    
    layer_indices = sorted(layer_representations.keys())
    
    # Flatten each layer for comparison
    layer_vectors = []
    for layer_idx in layer_indices:
        tensor = layer_representations[layer_idx]
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        # Sample subset for computational efficiency
        flat = tensor.flatten()[:10000]  # Use first 10k elements
        layer_vectors.append(flat)
    
    layer_vectors = np.array(layer_vectors)
    
    # Compute linkage
    linkage_matrix = linkage(layer_vectors, method=method)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Dendrogram
    dendrogram(linkage_matrix, labels=layer_indices, ax=ax1)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Distance', fontsize=12)
    ax1.set_title(f'Hierarchical Clustering of Layers ({method.capitalize()})',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Distance matrix
    dist_matrix = pdist(layer_vectors, metric='euclidean')
    from scipy.spatial.distance import squareform
    dist_square = squareform(dist_matrix)
    
    im = ax2.imshow(dist_square, cmap='viridis', interpolation='nearest')
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Layer Index', fontsize=12)
    ax2.set_title('Layer Distance Matrix', fontsize=14, fontweight='bold')
    
    # Set ticks
    tick_positions = np.linspace(0, len(layer_indices)-1, min(10, len(layer_indices)), dtype=int)
    ax2.set_xticks(tick_positions)
    ax2.set_yticks(tick_positions)
    ax2.set_xticklabels([layer_indices[i] for i in tick_positions])
    ax2.set_yticklabels([layer_indices[i] for i in tick_positions])
    
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved clustering dendrogram to {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Intermediate Representation Visualization Utilities")
    print("=" * 60)
    print("\nThis module provides functions for:")
    print("  - Extracting MSA, Pair, and Structure representations")
    print("  - Visualizing representations as heatmaps and line plots")
    print("\nStatus: ✓ IMPLEMENTED")
    print("\nFeatures:")
    print("  [✓] Extract MSA representations")
    print("  [✓] Extract Pair representations")
    print("  [✓] Extract Structure module outputs")
    print("  [✓] Visualize MSA as heatmaps")
    print("  [✓] Visualize Pair representations")
    print("  [✓] Track representation evolution across layers")
    print("  [✓] Channel-specific visualization")
    print("  [✓] Save/load representations to disk")
    print("\nNext steps:")
    print("  - Run unit tests: pytest tests/test_intermediate_extraction.py")
    print("  - Create demo notebook")
    print("  - Integrate with web interface")
    print("\nExample usage:")
    print("  from visualize_intermediate_reps_utils import *")
    print("  INTERMEDIATE_REPS.enable()")
    print("  hooks = register_evoformer_hooks(model)")
    print("  output = model(batch)")
    print("  msa_reps = extract_msa_representations(None)")
    print("  plot_msa_representation_heatmap(msa_reps[0], 0, 'msa_layer0.png')")