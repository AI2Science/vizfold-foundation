import torch
import numpy as np
import pickle

def format_for_vizfold(input_pt_path: str, output_archive_path: str):
    print(f"Loading raw trace from {input_pt_path}...")
    raw_data = torch.load(input_pt_path, weights_only=True)
    
    sequence = raw_data["sequence"]
    seq_len = len(sequence)
    
    # 1. Clean up Attention Maps (Intercepted from LM Stem: HAS special tokens)
    stacked_attentions = torch.stack(raw_data["attentions"])
    stacked_attentions = stacked_attentions.squeeze(1) # Remove batch -> [layers, heads, seq+2, seq+2]
    clean_attentions = stacked_attentions[:, :, 1:-1, 1:-1] # Slice out <cls> and <eos> -> [layers, heads, seq, seq]
    
    # 2. Clean up Activations (Output from Folding Trunk: NO special tokens)
    activations = raw_data["activations"]
    clean_activations = activations.squeeze(0) # Remove batch -> [seq, hidden_dim]
    
    # 3. Clean up Positions (Output from Folding Trunk)
    # Shape is [8_iterations, 1_batch, seq, 14_atoms, 3_coords]
    # We grab the final iteration [-1] and squeeze out the batch dimension
    clean_positions = raw_data["positions"][-1].squeeze(0) # -> [seq, 14, 3]
    
    # Assertions to guarantee everything perfectly matches the sequence length of 9
    assert clean_attentions.shape[2] == seq_len, f"Attention shape mismatch: {clean_attentions.shape}"
    assert clean_activations.shape[0] == seq_len, f"Activation shape mismatch: {clean_activations.shape}"
    assert clean_positions.shape[0] == seq_len, f"Position shape mismatch: {clean_positions.shape}"
    
    print("All tensor dimensions perfectly align with the amino acid sequence!")

    # 4. Package into standard VizFold schema
    vizfold_archive = {
        "sequence": sequence,
        "model_name": "ESMFold_v1",
        "representations": {
            "single": clean_activations.numpy(),
            "attention": clean_attentions.numpy() 
        },
        "structure": {
            "positions": clean_positions.numpy()
        }
    }
    
    with open(output_archive_path, 'wb') as f:
        pickle.dump(vizfold_archive, f)
        
    print(f"Successfully formatted and saved VizFold archive to {output_archive_path}")

if __name__ == "__main__":
    format_for_vizfold("esmfold_trace.pt", "esmfold_viz_archive.pkl")