import os
import numpy as np
import matplotlib.pyplot as plt


# ========== Input Parsing ==========
def load_all_heads(connections_file, top_k=None):
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

    for head_idx, conns in heads.items():
        conns.sort(key=lambda x: x[2], reverse=True)
        if top_k is not None:
            heads[head_idx] = conns[:top_k]

    return heads


def parse_fasta_sequence(fasta_path):
    """
    Parse a single-entry FASTA file and return the sequence string.
    """
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
    
    seq_lines = [line.strip() for line in lines if not line.startswith('>')]
    sequence = ''.join(seq_lines)
    return sequence


# ========== Arc Plotting ==========
def plot_arc_diagram_with_labels(connections, residue_sequence, output_file='arc.png',
                                 highlight_residue_index=None, save_to_png=True,
                                 plt_title=None):
    if not connections:
        print("No connections to plot.")
        return

    n_residues = len(residue_sequence)
    weights = [w for _, _, w in connections]
    w_min, w_max = min(weights), max(weights)

    fig, ax = plt.subplots(figsize=(max(12, n_residues // 10), 5))

    plotted = 0
    for res1, res2, weight in connections:
        res1 += 0.5
        res2 += 0.5
        height = abs(res2 - res1) / 2
        norm_weight = (weight - w_min) / (w_max - w_min) if w_max != w_min else 0.5
        linewidth = 0.5 + norm_weight * 3
        color = (0.0, 0.0, 0.5 + 0.5 * norm_weight)

        arc = np.linspace(0, np.pi, 100)
        arc_x = np.linspace(res1, res2, 100)
        arc_y = height * np.sin(arc)

        ax.plot(arc_x, arc_y, color=color, linewidth=linewidth, alpha=0.9)
        plotted += 1

    x_locs = np.arange(len(residue_sequence)) + 0.5
    x_labels = list(residue_sequence)

    ax.set_xticks(x_locs)
    tick_labels = ax.set_xticklabels(x_labels, fontsize=8, rotation=0, ha='center')

    # Highlight the specific residue
    if highlight_residue_index is not None and 0 <= highlight_residue_index < len(tick_labels):
        tick_labels[highlight_residue_index].set_color('blue')
        tick_labels[highlight_residue_index].set_fontweight('bold')

    ax.set_ylim(0, None)
    ax.set_ylabel('Attention Strength')

    if plt_title is not None:
        ax.set_title(plt_title)
    else:
        ax.set_title(f'Residue Attention (n={plotted})')

    ax.tick_params(axis='x', which='both', length=0)
    ax.set_yticks([])

    plt.tight_layout()
    # plt.show()
    if save_to_png:
        plt.savefig(output_file, dpi=300)
        print(f"[Saved] {output_file}")
    plt.close()


# ========== Main Function ==========
def generate_arc_diagrams(
    attention_dir,
    residue_sequence,
    output_dir,
    protein,
    attention_type="msa_row",  # or "triangle_start"
    residue_indices=None,      # only for triangle
    top_k=50,
    layer_idx=47,
    save_to_png=True,
):
    os.makedirs(output_dir, exist_ok=True)

    if attention_type == "msa_row":
        file_path = os.path.join(attention_dir, f"msa_row_attn_layer{layer_idx}.txt")
        heads = load_all_heads(file_path, top_k=top_k)
        pngs = []

        for head_idx, connections in heads.items():
            out_png = os.path.join(output_dir, f"msa_row_head_{head_idx}_layer_{layer_idx}_{protein}_arc.png")
            plt_title = f"Residue Attention: {protein} MSA Row (Head {head_idx} Layer {layer_idx})"
            plot_arc_diagram_with_labels(connections, residue_sequence, output_file=out_png,
                                         save_to_png=save_to_png, plt_title=plt_title)
            pngs.append(out_png)

    elif attention_type == "triangle_start":
        assert residue_indices is not None, "residue_indices required for triangle_start attention"

        for res_idx in residue_indices:
            file_path = os.path.join(attention_dir, f"triangle_start_attn_layer{layer_idx}_residue_idx_{res_idx}.txt")
            if not os.path.exists(file_path):
                print(f"[Warning] Missing file for residue {res_idx}")
                continue

            heads = load_all_heads(file_path, top_k=top_k)
            pngs = []

            for head_idx, connections in heads.items():
                out_png = os.path.join(output_dir, f"tri_start_res_{res_idx}_head_{head_idx}_layer_{layer_idx}_{protein}_arc.png")
                plt_title = f"Residue Attention: {protein} Tri Start (Head {head_idx} Layer {layer_idx})"
                plot_arc_diagram_with_labels(connections, residue_sequence, output_file=out_png,
                             highlight_residue_index=res_idx, save_to_png=save_to_png, plt_title=plt_title)
                pngs.append(out_png)


if __name__ == "__main__":
    topk = 50
    layer_idx = 47
    attention_dir = "/projects/bekh/thayes/demo_attn_saves/6KWC_demo"
    msa_output_dir = "/u/thayes/vizfold/demo_plots_msa_row"
    tri_output_dir = "/u/thayes/vizfold/demo_plots_tri_start"
    fasta_path = "./examples/monomer/fasta_dir/6kwc.fasta"
    protein = '6KWC'

    # Load sequence
    residue_seq = parse_fasta_sequence(fasta_path)

    # For MSA row
    print('Making visuals for MSA Row Attention...')
    generate_arc_diagrams(
        attention_dir=attention_dir,
        residue_sequence=residue_seq,
        output_dir=msa_output_dir,
        protein=protein,
        attention_type="msa_row",
        top_k=topk,
        layer_idx=layer_idx
    )

    # For Triangle Start
    print('Making visuals for Triangle Start Attention...')
    generate_arc_diagrams(
        attention_dir=attention_dir,
        residue_sequence=residue_seq,
        output_dir=tri_output_dir,
        protein=protein,
        attention_type="triangle_start",
        residue_indices=[18, 39, 51, 79, 138, 159],
        top_k=topk,
        layer_idx=layer_idx
    )

    print()
