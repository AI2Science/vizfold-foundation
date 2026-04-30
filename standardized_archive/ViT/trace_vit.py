import torch
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from archive_schema import create_vit_archive

def trace_vit(image, archive_path):
    """
    Trace a ViT image through the model and store inputs, activations, attention,
    outputs, and metadata into a standardized Zarr archive.
    """
    
    # Load model & processor
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="eager")
    model.eval()
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    
    # Create archive with metadata
    archive = create_vit_archive(archive_path, model.config)
    
    # Store inputs
    archive["inputs"].create_dataset(
        "processed_image",
        data=inputs["pixel_values"].numpy(),
        chunks=True
    )
    
    # Forward pass with gradients off
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True
        )
    
    # Store activations
    for i, hidden in enumerate(outputs.hidden_states):
        layer_group = archive["activations"].require_group(f"layer_{i}")
        layer_group.create_dataset(
            "hidden_states",
            data=hidden.detach().cpu().numpy(),
            chunks=True
        )

    # Store CLS token output from the last layer
    cls_output = outputs.hidden_states[-1][:, 0, :].detach().cpu().numpy()
    archive["activations"].create_dataset("cls_output", data=cls_output, chunks=True)
    
    # Store attention
    for i, attn in enumerate(outputs.attentions):
        archive["attention"].create_dataset(
            f"layer_{i}",
            data=attn.detach().cpu().numpy(),
            chunks=True
        )
    
    # Store outputs
    logits = outputs.logits.detach().cpu().numpy()
    archive["outputs"].create_dataset("logits", data=logits)
    archive["outputs"].attrs["predicted_class"] = int(logits.argmax(-1))
    
    return archive