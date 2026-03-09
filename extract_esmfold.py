import torch
from transformers import AutoTokenizer, EsmForProteinFolding

def run_esmfold(sequence: str, output_path: str):
    print("Loading ESMFold model and tokenizer...")
    model_name = "facebook/esmfold_v1"
    cache_path = "/home/hice1/rsinghal49/scratch/huggingface_cache"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=cache_path
    )
    
    # We must enforce output_attentions=True so the layers actually compute them
    model = EsmForProteinFolding.from_pretrained(
        model_name, 
        output_hidden_states=True, 
        output_attentions=True,
        cache_dir=cache_path,
        use_safetensors=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # ---------------------------------------------------------
    # PROJECT TASK: "Identify hooks to extract... Attention maps"
    # ---------------------------------------------------------
    print("Registering PyTorch forward hooks to intercept attention maps...")
    extracted_attentions = []

    def attention_hook(module, input, output):
        # Hugging Face self-attention outputs a tuple when output_attentions=True:
        # (context_layer, attention_probs)
        # We want to grab the attention_probs (index 1) and save it.
        if len(output) > 1:
            extracted_attentions.append(output[1].detach().cpu())
        else:
            print("Warning: Hook fired but attention probabilities were not found.")

    # Attach the hook to the self-attention module of every Transformer block in the ESM stem
    for layer in model.esm.encoder.layer:
        layer.attention.self.register_forward_hook(attention_hook)
    # ---------------------------------------------------------

    print(f"Tokenizing sequence: {sequence}")
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False).to(device)
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("Extracting tensors for VizFold...")
    
    # 1. Predicted Structures
    positions = outputs.positions.cpu()
    
    # 2. Layer-wise activations 
    # (s_s represents the per-residue embeddings derived from the ESM-2 LM stem)
    activations = outputs.s_s.cpu() 
    
    # 3. Attention Maps (Pulled from our custom hooks!)
    attentions = extracted_attentions if extracted_attentions else None

    extracted_data = {
        "sequence": sequence,
        "positions": positions,
        "activations": activations,
        "attentions": attentions
    }
    
    torch.save(extracted_data, output_path)
    print(f"Data successfully saved to {output_path}")
    
    # Print a sanity check to prove the hooks worked
    if attentions:
        print(f"\nSuccess! Extracted {len(attentions)} layers of attention maps.")
        print(f"Shape of layer 0 attention map: {attentions[0].shape}")

if __name__ == "__main__":
    test_sequence = "MKTVRQIGV"
    run_esmfold(test_sequence, "esmfold_trace.pt")