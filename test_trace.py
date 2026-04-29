import torch
import sys

def validate_trace(file_path):
    print(f"Loading trace from: {file_path}")
    
    # Load the output dictionary
    if file_path.endswith('.pt'):
        data = torch.load(file_path, weights_only=True)
    else:
        import pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
    seq = data.get('sequence', None)
    if not seq:
        print("Could not find 'sequence' string in the trace output.")
        return
        
    N = len(seq)
    print(f"\nSequence: {seq}")
    print(f"Target Sequence Length (N): {N}")
    print("-" * 40)
    
    # 1. Check Attention Slicing
    attentions = data.get('attention', data.get('attentions'))
    if attentions is not None:
        # Check if the lead saved it as a list of tensors instead of one big tensor
        if isinstance(attentions, list):
            print(f"Structure: List of {len(attentions)} attention tensors.")
            shape = attentions[0].shape  # Look at the first layer's shape
        else:
            print("Structure: Single stacked tensor.")
            shape = attentions.shape
            
        print(f"Attention Shape: {shape}")
        
        if shape[-1] == N and shape[-2] == N:
            print("Token slicing is PERFECT! The <cls> and <eos> tokens were successfully removed.")
        elif shape[-1] == N + 2:
            print("Token slicing failed. The padding tokens are still in the tensor.")
        else:
            print(f"Unexpected attention dimensions. Expected {N}x{N}, got {shape[-2]}x{shape[-1]}")
    else:
        print("No attention trace found in the dictionary.")

    print("-" * 40)
    
    # 2. Check Activations (Single Representations)
    activations = data.get('activations')
    if activations is not None:
        if isinstance(activations, list):
             act_shape = activations[0].shape
        else:
             act_shape = activations.shape
             
        print(f"Activations Shape: {act_shape}")
    else:
        print("No 'activations' found. (Expected since we are waiting to push your s_s extraction!)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_trace.py <path_to_trace_file>")
    else:
        validate_trace(sys.argv[1])