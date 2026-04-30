import zarr
import numpy as np

def create_vit_archive(archive_path, model_config, num_layers=12, num_heads=12, hidden_dim=768, num_tokens=197):
    """
    Create a standardized ViT archive with formal metadata, layer shapes, and head counts.
    
    Parameters:
        archive_path (str): Path to the Zarr archive.
        model_config (transformers.PretrainedConfig): ViT model config.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden dimension size.
        num_tokens (int): Number of tokens (patches + CLS).
        
    Returns:
        zarr.hierarchy.Group: Root Zarr archive.
    """
    
    # Root archive
    archive = zarr.open(archive_path, mode="w")
    
    # Metadata group
    meta = archive.create_group("metadata")
    meta.attrs["model_name"] = getattr(model_config, "model_type", "vit-base")
    meta.attrs["num_layers"] = num_layers
    meta.attrs["num_heads"] = num_heads
    meta.attrs["hidden_dim"] = hidden_dim
    meta.attrs["num_tokens"] = num_tokens
    meta.attrs["input_shape"] = (3, 224, 224)
    
    # Placeholder for layer-wise metadata
    layers_meta = meta.require_group("layers")
    for i in range(num_layers):
        layer_group = layers_meta.require_group(f"layer_{i}")
        layer_group.attrs["hidden_shape"] = (num_tokens, hidden_dim)
        layer_group.attrs["num_heads"] = num_heads
        layer_group.attrs["attention_shape"] = (num_heads, num_tokens, num_tokens)
    
    # Inputs
    archive.create_group("inputs")
    # Processed images will be stored later as datasets
    # shape example: (batch, 3, 224, 224)
    
    # Activations
    archive.create_group("activations")
    # Each layer will have a "hidden_states" dataset
    # shape example: (batch, num_tokens, hidden_dim)
    
    # Attention
    archive.create_group("attention")
    # Each layer will have a dataset of shape (batch, num_heads, tokens, tokens)
    
    # Outputs
    archive.create_group("outputs")
    # Logits and predicted class will be stored here
    
    return archive