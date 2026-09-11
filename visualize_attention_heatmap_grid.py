import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_all_heads(connections_file):
    heads = {}
    current_head = None
    with open(connections_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('layer'):
                parts = line.replace(',', '').split()
                head_idx = int(parts[-1])
                current_head = head_idx
                heads[current_head] = []
            else:
                res1, res2, weight = map(float, line.split())
                heads[current_head].append((int(res1), int(res2), weight))
    return heads


def reconstruct_matrix(connections, seq_len):
    matrix = np.zeros((seq_len, seq_len))
    for res1, res2, weight in connections:
        if res1 < seq_len and res2 < seq_len:
            matrix[res1, res2] = weight
    return matrix


def create_heatmap_grid(attention_file, seq_len, layer_idx=47, attention_type="msa_row", output_html="heatmap_grid.html", threshold=None):
    heads = load_all_heads(attention_file)
    num_heads = len(heads)

    if num_heads == 0:
        print(f"No heads found in {attention_file}")
        return

    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    # Calculate global and per-head min/max values, respecting the threshold
    all_weights = [w for head_idx in sorted(heads.keys()) for _, _, w in heads[head_idx]]
    if threshold is not None:
        all_weights = [w for w in all_weights if w >= threshold]
    
    global_min = min(all_weights) if all_weights else 0
    global_max = max(all_weights) if all_weights else 1

    per_head_mins = []
    per_head_maxs = []

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Head {i}" for i in sorted(heads.keys())],
        horizontal_spacing=0.05,
        vertical_spacing=0.15
    )

    for idx, head_idx in enumerate(sorted(heads.keys())):
        row = idx // cols + 1
        col = idx % cols + 1

        matrix = reconstruct_matrix(heads[head_idx], seq_len)
        if threshold is not None:
            matrix[matrix < threshold] = np.nan  # Use nan to hide values below threshold

        head_connections = heads[head_idx]
        head_weights = [w for _, _, w in head_connections]
        if threshold is not None:
            head_weights = [w for w in head_weights if w >= threshold]

        head_min = min(head_weights) if head_weights else 0
        head_max = max(head_weights) if head_weights else 1
        per_head_mins.append(head_min)
        per_head_maxs.append(head_max)

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale='Blues',
                zmin=global_min,
                zmax=global_max,
                showscale=(idx == 0),
                colorbar=dict(x=1.02, len=0.3, title="Weight") if idx == 0 else None
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Residue", row=row, col=col, showticklabels=False)
        fig.update_yaxes(title_text="Residue", row=row, col=col, showticklabels=False)

    title_text = f"{attention_type.upper()} Layer {layer_idx} - All Heads"
    if threshold is not None:
        title_text += f" (Threshold > {threshold})"

    fig.update_layout(
        title_text=title_text,
        title_x=0.5,
        height=350 * rows,
        width=1200,
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.6,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=list([
                    dict(
                        label="Global Norm",
                        method="restyle",
                        args=[{"zmin": [global_min], "zmax": [global_max], "showscale": [True] + [False] * (num_heads - 1)}],
                    ),
                    dict(
                        label="Per-Head Norm",
                        method="restyle",
                        args=[{"zmin": per_head_mins, "zmax": per_head_maxs, "showscale": [False] * num_heads}],
                    ),
                ]),
            )
        ]
    )

    fig.write_html(output_html)
    print(f"Saved: {output_html}")
    return fig


def visualize_layer_attention(attention_dir, seq_len, layer_idx=47, attention_type="msa_row", residue_idx=None, output_dir="./outputs/attention_heatmaps", threshold=None):
    os.makedirs(output_dir, exist_ok=True)

    if attention_type == "msa_row":
        attention_file = os.path.join(attention_dir, f"msa_row_attn_layer{layer_idx}.txt")
        output_html = os.path.join(output_dir, f"msa_row_layer{layer_idx}_heatmap_grid.html")
    elif attention_type == "triangle_start":
        if residue_idx is None:
            raise ValueError("residue_idx required for triangle_start")
        attention_file = os.path.join(attention_dir, f"triangle_start_attn_layer{layer_idx}_residue_idx_{residue_idx}.txt")
        output_html = os.path.join(output_dir, f"triangle_start_layer{layer_idx}_res{residue_idx}_heatmap_grid.html")
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")

    if not os.path.exists(attention_file):
        print(f"File not found: {attention_file}")
        return None

    print(f"Processing: {attention_file}")
    return create_heatmap_grid(attention_file, seq_len, layer_idx, attention_type, output_html, threshold)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize attention heatmap grids for OpenFold.")
    
    parser.add_argument("--attention_dir", type=str, required=True, 
                        help="Directory containing attention files.")
    parser.add_argument("--output_dir", type=str, default="./outputs/attention_heatmaps", 
                        help="Directory to save the output HTML files.")
    parser.add_argument("--seq_len", type=int, required=True, 
                        help="Sequence length.")
    parser.add_argument("--layer_idx", type=int, required=True, 
                        help="Layer index to visualize.")
    parser.add_argument("--attention_type", type=str, required=True, choices=["msa_row", "triangle_start"],
                        help="Type of attention to visualize.")
    parser.add_argument("--residue_idx", type=int, default=None,
                        help="Residue index, required for 'triangle_start' attention type.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Attention weight threshold. Weights below this value will not be displayed.")

    args = parser.parse_args()

    if args.attention_type == "triangle_start" and args.residue_idx is None:
        parser.error("--residue_idx is required when attention_type is 'triangle_start'")

    visualize_layer_attention(
        attention_dir=args.attention_dir,
        seq_len=args.seq_len,
        layer_idx=args.layer_idx,
        attention_type=args.attention_type,
        residue_idx=args.residue_idx,
        output_dir=args.output_dir,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()
