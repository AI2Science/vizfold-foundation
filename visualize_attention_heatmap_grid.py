import json
import os
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import torch
except ImportError:
    torch = None


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


def load_dense_attention(attention_file):
    if not os.path.exists(attention_file):
        raise FileNotFoundError(attention_file)

    data = np.load(attention_file)
    if isinstance(data, np.ndarray):
        return data

    if "attn" in data:
        return data["attn"]

    if len(data.files) == 1:
        return data[data.files[0]]

    raise ValueError(f"Unable to parse attention contents from {attention_file}")


def load_attention_from_text(attention_file, seq_len):
    heads = load_all_heads(attention_file)
    matrices = [reconstruct_matrix(heads[head_idx], seq_len) for head_idx in sorted(heads.keys())]
    if len(matrices) == 0:
        raise ValueError(f"No attention heads found in {attention_file}")
    return np.stack(matrices, axis=0)


def load_attention_array(attention_dir, seq_len, layer_idx, attention_type="msa_row", residue_idx=None):
    if attention_type == "msa_row":
        base_name = f"msa_row_attn_layer{layer_idx}"
    elif attention_type == "triangle_start":
        if residue_idx is None:
            raise ValueError("residue_idx required for triangle_start attention")
        base_name = f"triangle_start_attn_layer{layer_idx}_residue_idx_{residue_idx}"
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")

    npz_file = os.path.join(attention_dir, f"{base_name}.npz")
    txt_file = os.path.join(attention_dir, f"{base_name}.txt")

    if os.path.exists(npz_file):
        return load_dense_attention(npz_file)

    if os.path.exists(txt_file):
        return load_attention_from_text(txt_file, seq_len)

    raise FileNotFoundError(f"Neither {npz_file} nor {txt_file} exists")


def discover_attention_layers(attention_dir, attention_type="msa_row", residue_idx=None):
    layer_indices = set()
    if not os.path.isdir(attention_dir):
        return []

    if attention_type == "msa_row":
        pattern = re.compile(r"^msa_row_attn_layer(\d+)\.(?:npz|txt)$")
    else:
        if residue_idx is None:
            return []
        pattern = re.compile(rf"^triangle_start_attn_layer(\d+)_residue_idx_{residue_idx}\.(?:npz|txt)$")

    for fname in os.listdir(attention_dir):
        match = pattern.match(fname)
        if match:
            layer_indices.add(int(match.group(1)))

    return sorted(layer_indices)


def compute_attention_rollout(attention_arrays, add_identity=True, eps=1e-12, use_gpu=True):
    if len(attention_arrays) == 0:
        raise ValueError("No attention arrays available for rollout computation")

    if use_gpu and torch is not None and torch.cuda.is_available():
        device = torch.device("cuda")
        layers = []
        for attn in attention_arrays:
            if isinstance(attn, np.ndarray):
                attn = torch.from_numpy(attn)
            elif not isinstance(attn, torch.Tensor):
                attn = torch.tensor(attn)
            layers.append(attn.to(device=device, dtype=torch.float32))

        averaged_layers = [layer.mean(dim=0) for layer in layers]

        if add_identity:
            averaged_layers = [layer + torch.eye(layer.shape[-1], device=device) for layer in averaged_layers]

        normalized_layers = [layer / (layer.sum(dim=-1, keepdim=True) + eps) for layer in averaged_layers]

        rollout = normalized_layers[0]
        for layer in normalized_layers[1:]:
            rollout = layer.matmul(rollout)

        rollout = rollout.detach().cpu()
        return rollout.numpy()

    # Fallback to NumPy if PyTorch/CUDA is unavailable.
    averaged_layers = [np.mean(attn, axis=0) for attn in attention_arrays]

    if add_identity:
        averaged_layers = [layer + np.eye(layer.shape[-1]) for layer in averaged_layers]

    normalized_layers = [layer / (layer.sum(axis=-1, keepdims=True) + eps) for layer in averaged_layers]

    rollout = normalized_layers[0]
    for layer in normalized_layers[1:]:
        rollout = layer @ rollout

    return rollout


def create_rollout_heatmap(rollout_matrix, seq_len, attention_type="msa_row", layer_start=0, layer_end=None, output_html="rollout_heatmap.html", threshold=None):
    if hasattr(rollout_matrix, "cpu") and not isinstance(rollout_matrix, np.ndarray):
        rollout_matrix = rollout_matrix.detach().cpu().numpy()

    matrix = rollout_matrix.copy()
    if threshold is not None:
        matrix[matrix < threshold] = np.nan

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            colorscale='Blues',
            colorbar=dict(title='Rollout'),
        )
    )

    title_range = f"layers {layer_start}-{layer_end}" if layer_end is not None else f"layers {layer_start}-end"
    title_text = f"{attention_type.upper()} Attention Rollout ({title_range})"
    if threshold is not None:
        title_text += f" (threshold > {threshold})"

    fig.update_layout(
        title_text=title_text,
        title_x=0.5,
        xaxis_title="Residue",
        yaxis_title="Residue",
        width=900,
        height=900,
    )

    fig.write_html(output_html)
    print(f"Saved rollout heatmap to: {output_html}")
    return fig


def visualize_attention_rollout(attention_dir, seq_len, attention_type="msa_row", residue_idx=None, output_dir="./outputs/attention_heatmaps", layer_start=None, layer_end=None, threshold=None, add_identity=True, use_gpu=True):
    os.makedirs(output_dir, exist_ok=True)

    available_layers = discover_attention_layers(attention_dir, attention_type, residue_idx)
    if len(available_layers) == 0:
        raise ValueError(f"No attention files found in {attention_dir} for {attention_type}")

    if layer_start is None:
        layer_start = available_layers[0]
    if layer_end is None:
        layer_end = available_layers[-1]

    layer_indices = [L for L in available_layers if layer_start <= L <= layer_end]
    if len(layer_indices) == 0:
        raise ValueError(f"No layers found in range {layer_start}-{layer_end}")

    attention_arrays = [
        load_attention_array(attention_dir, seq_len, layer_idx, attention_type, residue_idx)
        for layer_idx in layer_indices
    ]

    rollout_matrix = compute_attention_rollout(attention_arrays, add_identity=add_identity, use_gpu=use_gpu)

    output_suffix = f"layers{layer_start}-{layer_end}"
    if attention_type == "msa_row":
        output_html = os.path.join(output_dir, f"msa_row_rollout_{output_suffix}_heatmap.html")
    else:
        output_html = os.path.join(output_dir, f"triangle_start_rollout_{output_suffix}_res{residue_idx}_heatmap.html")

    return create_rollout_heatmap(
        rollout_matrix,
        seq_len,
        attention_type,
        layer_start,
        layer_end,
        output_html,
        threshold,
    )


def create_interactive_rollout_html(averaged_layers, seq_len, attention_type="msa_row", layer_indices=None, output_html="interactive_rollout.html", residue_idx=None, threshold=None, default_start=None, default_end=None):
    if layer_indices is None:
        layer_indices = [layer_data["layer"] for layer_data in averaged_layers]

    if default_start is None:
        default_start = min(layer_indices)
    if default_end is None:
        default_end = max(layer_indices)

    attention_label = attention_type.upper()
    residue_part = f" (residue {residue_idx})" if residue_idx is not None else ""
    layer_data_json = json.dumps([
        {"layer": layer_data["layer"], "matrix": layer_data["matrix"]}
        for layer_data in averaged_layers
    ])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <title>{attention_label} Attention Rollout</title>
    <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        .controls {{ margin-bottom: 16px; }}
        label {{ margin-right: 12px; }}
        input {{ width: 80px; margin-left: 4px; }}
        button {{ margin-left: 12px; padding: 8px 14px; }}
        #status {{ margin-top: 12px; color: #333; }}
    </style>
</head>
<body>
    <h1>{attention_label} Attention Rollout</h1>
    <p>Interactive layer-range selector for {attention_label} rollout{residue_part}.</p>
    <div class=\"controls\">
        <label>Start layer <input id=\"layer_start\" type=\"number\" min=\"{min(layer_indices)}\" max=\"{max(layer_indices)}\" value=\"{default_start}\" /></label>
        <label>End layer <input id=\"layer_end\" type=\"number\" min=\"{min(layer_indices)}\" max=\"{max(layer_indices)}\" value=\"{default_end}\" /></label>
        <label>Threshold <input id=\"threshold\" type=\"number\" step=\"0.0001\" value=\"{threshold if threshold is not None else ''}\" placeholder=\"none\" /></label>
        <button id=\"compute_button\">Compute Rollout</button>
    </div>
    <div id=\"status\">Loaded {len(layer_indices)} layers.</div>
    <div id=\"heatmap\"></div>

    <script>
        const attentionLayers = {layer_data_json};

        function addIdentity(matrix) {{
            const n = matrix.length;
            const out = new Array(n);
            for (let i = 0; i < n; i++) {{
                out[i] = new Array(n);
                for (let j = 0; j < n; j++) {{
                    out[i][j] = matrix[i][j] + (i === j ? 1.0 : 0.0);
                }}
            }}
            return out;
        }}

        function normalize(matrix) {{
            const n = matrix.length;
            const out = new Array(n);
            for (let i = 0; i < n; i++) {{
                out[i] = new Array(n);
                let rowSum = 0.0;
                for (let j = 0; j < n; j++) {{
                    rowSum += matrix[i][j];
                }}
                const denom = rowSum === 0 ? 1.0 : rowSum;
                for (let j = 0; j < n; j++) {{
                    out[i][j] = matrix[i][j] / denom;
                }}
            }}
            return out;
        }}

        function matMul(A, B) {{
            const n = A.length;
            const out = new Array(n);
            for (let i = 0; i < n; i++) {{
                out[i] = new Array(n).fill(0.0);
            }}
            for (let i = 0; i < n; i++) {{
                for (let k = 0; k < n; k++) {{
                    const a = A[i][k];
                    if (a === 0) continue;
                    const rowB = B[k];
                    const outRow = out[i];
                    for (let j = 0; j < n; j++) {{
                        outRow[j] += a * rowB[j];
                    }}
                }}
            }}
            return out;
        }}

        function applyThreshold(matrix, threshold) {{
            if (isNaN(threshold)) {{
                return matrix;
            }}
            const n = matrix.length;
            const out = new Array(n);
            for (let i = 0; i < n; i++) {{
                out[i] = new Array(n);
                for (let j = 0; j < n; j++) {{
                    out[i][j] = matrix[i][j] >= threshold ? matrix[i][j] : NaN;
                }}
            }}
            return out;
        }}

        function computeRollout() {{
            const start = parseInt(document.getElementById('layer_start').value, 10);
            const end = parseInt(document.getElementById('layer_end').value, 10);
            const threshold = parseFloat(document.getElementById('threshold').value);
            const selected = attentionLayers.filter(layerData => layerData.layer >= start && layerData.layer <= end);

            const status = document.getElementById('status');
            if (selected.length === 0) {{
                status.textContent = `No layers found in range ${start}-${end}.`;
                return;
            }}

            status.textContent = `Computing rollout for layers ${start}-${end} ...`;
            let rollout = normalize(addIdentity(selected[0].matrix));
            for (let i = 1; i < selected.length; i++) {{
                const layer = normalize(addIdentity(selected[i].matrix));
                rollout = matMul(layer, rollout);
            }}

            const thresholded = applyThreshold(rollout, threshold);
            const title = `{attention_label} Attention Rollout (layers ${start}-${end})`;
            const data = [{{
                z: thresholded,
                type: 'heatmap',
                colorscale: 'Blues',
                colorbar: {{title: 'Rollout'}},
                hoverongaps: false
            }}];
            const layout = {{
                title: title,
                xaxis: {{title: 'Residue'}},
                yaxis: {{title: 'Residue'}},
                width: 900,
                height: 900,
            }};

            Plotly.react('heatmap', data, layout);
            status.textContent = `Rendered rollout for layers ${start}-${end}.`;
        }}

        document.getElementById('compute_button').addEventListener('click', computeRollout);
        window.addEventListener('load', computeRollout);
    </script>
</body>
</html>
"""

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Saved interactive rollout HTML to: {output_html}")
    return output_html


def visualize_attention_rollout_interactive(attention_dir, seq_len, attention_type="msa_row", residue_idx=None, output_dir="./outputs/attention_heatmaps", layer_start=None, layer_end=None, threshold=None):
    os.makedirs(output_dir, exist_ok=True)

    available_layers = discover_attention_layers(attention_dir, attention_type, residue_idx)
    if len(available_layers) == 0:
        raise ValueError(f"No attention files found in {attention_dir} for {attention_type}")

    if layer_start is None:
        layer_start = available_layers[0]
    if layer_end is None:
        layer_end = available_layers[-1]

    layer_indices = [L for L in available_layers if layer_start <= L <= layer_end]
    if len(layer_indices) == 0:
        raise ValueError(f"No layers found in range {layer_start}-{layer_end}")

    averaged_layers = []
    for layer_idx in layer_indices:
        attn = load_attention_array(attention_dir, seq_len, layer_idx, attention_type, residue_idx)
        averaged = np.mean(attn, axis=0)
        averaged_layers.append({"layer": layer_idx, "matrix": averaged.tolist()})

    output_suffix = f"layers{layer_start}-{layer_end}"
    if attention_type == "msa_row":
        output_html = os.path.join(output_dir, f"msa_row_interactive_rollout_{output_suffix}_heatmap.html")
    else:
        output_html = os.path.join(output_dir, f"triangle_start_interactive_rollout_{output_suffix}_res{residue_idx}_heatmap.html")

    return create_interactive_rollout_html(
        averaged_layers,
        seq_len,
        attention_type,
        layer_indices=layer_indices,
        output_html=output_html,
        residue_idx=residue_idx,
        threshold=threshold,
        default_start=layer_start,
        default_end=layer_end,
    )


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
    parser.add_argument("--layer_idx", type=int, default=None, 
                        help="Layer index to visualize for per-head heatmaps.")
    parser.add_argument("--rollout", action="store_true", default=False,
                        help="Compute attention rollout instead of per-head heatmap grids.")
    parser.add_argument("--interactive_rollout", action="store_true", default=False,
                        help="Generate standalone HTML with layer-range selection UI for rollout.")
    parser.add_argument("--rollout_layer_start", type=int, default=None,
                        help="Start layer index for rollout aggregation.")
    parser.add_argument("--rollout_layer_end", type=int, default=None,
                        help="End layer index for rollout aggregation.")
    parser.add_argument("--attention_type", type=str, required=True, choices=["msa_row", "triangle_start"],
                        help="Type of attention to visualize.")
    parser.add_argument("--residue_idx", type=int, default=None,
                        help="Residue index, required for 'triangle_start' attention type.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Attention weight threshold. Weights below this value will not be displayed.")

    args = parser.parse_args()

    if args.attention_type == "triangle_start" and args.residue_idx is None:
        parser.error("--residue_idx is required when attention_type is 'triangle_start'")

    if args.rollout:
        if args.interactive_rollout:
            visualize_attention_rollout_interactive(
                attention_dir=args.attention_dir,
                seq_len=args.seq_len,
                attention_type=args.attention_type,
                residue_idx=args.residue_idx,
                output_dir=args.output_dir,
                layer_start=args.rollout_layer_start,
                layer_end=args.rollout_layer_end,
                threshold=args.threshold,
            )
        else:
            visualize_attention_rollout(
                attention_dir=args.attention_dir,
                seq_len=args.seq_len,
                attention_type=args.attention_type,
                residue_idx=args.residue_idx,
                output_dir=args.output_dir,
                layer_start=args.rollout_layer_start,
                layer_end=args.rollout_layer_end,
                threshold=args.threshold,
            )
    else:
        if args.layer_idx is None:
            parser.error("--layer_idx is required when not using --rollout")

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
