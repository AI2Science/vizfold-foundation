import os
import numpy as np
import matplotlib.pyplot as plt

# ---- Config ----
PROT = "6KWC"
TRI_RESIDUE_IDX = 18
ATTN_MAP_DIR = f"./outputs/attention_files_{PROT}_demo_tri_{TRI_RESIDUE_IDX}"
IMAGE_OUTPUT_DIR = f"./outputs/attention_images_{PROT}_demo_tri_{TRI_RESIDUE_IDX}"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# ---- Helper: parse attention txt file ----
def parse_attention_file(filepath):
    scores = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    score = float(parts[2])
                    i = int(parts[0])
                    j = int(parts[1])
                    scores.append((i, j, score))
                except:
                    continue
    return scores

# ---- Helper: build matrix from scores ----
def build_matrix(scores, size=200):
    matrix = np.zeros((size, size))
    for i, j, score in scores:
        if i < size and j < size:
            matrix[i, j] = score
    return matrix

# ---- Plot 1: Heatmap for a single layer ----
def plot_heatmap(layer_idx, attention_type="msa_row"):
    if attention_type == "msa_row":
        fname = f"msa_row_attn_layer{layer_idx}.txt"
    else:
        fname = f"triangle_start_attn_layer{layer_idx}_residue_idx_{TRI_RESIDUE_IDX}.txt"
    
    filepath = os.path.join(ATTN_MAP_DIR, fname)
    scores = parse_attention_file(filepath)
    matrix = build_matrix(scores)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Score')
    plt.title(f'{attention_type} Attention - Layer {layer_idx}')
    plt.xlabel('Residue j')
    plt.ylabel('Residue i')
    out_path = os.path.join(IMAGE_OUTPUT_DIR, f'heatmap_{attention_type}_layer{layer_idx}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {out_path}")

# ---- Plot 2: Line plot of top attention scores across layers ----
def plot_scores_across_layers(attention_type="msa_row", num_layers=48):
    top_scores = []
    for layer_idx in range(num_layers):
        if attention_type == "msa_row":
            fname = f"msa_row_attn_layer{layer_idx}.txt"
        else:
            fname = f"triangle_start_attn_layer{layer_idx}_residue_idx_{TRI_RESIDUE_IDX}.txt"
        filepath = os.path.join(ATTN_MAP_DIR, fname)
        if os.path.exists(filepath):
            scores = parse_attention_file(filepath)
            if scores:
                top_scores.append(max(s[2] for s in scores))
            else:
                top_scores.append(0)
        else:
            top_scores.append(0)
    
    plt.figure(figsize=(12, 5))
    plt.plot(range(num_layers), top_scores, marker='o', linewidth=2)
    plt.title(f'Top Attention Score Across Layers - {attention_type}')
    plt.xlabel('Layer')
    plt.ylabel('Top Attention Score')
    plt.grid(True)
    out_path = os.path.join(IMAGE_OUTPUT_DIR, f'lineplot_{attention_type}_across_layers.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved line plot to {out_path}")

# ---- Run everything ----
if __name__ == "__main__":
    # Generate heatmaps for a few layers
    for layer in range(48):
        plot_heatmap(layer, attention_type="msa_row")
        plot_heatmap(layer, attention_type="triangle_start")
    
    # Generate line plots across all 48 layers
    plot_scores_across_layers(attention_type="msa_row")
    plot_scores_across_layers(attention_type="triangle_start")
    
    print("All done! Check", IMAGE_OUTPUT_DIR)