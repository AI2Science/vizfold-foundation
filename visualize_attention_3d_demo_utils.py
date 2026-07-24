import csv
import numpy as np
from pymol import cmd
from pymol.cgo import CYLINDER, SPHERE

# Initialize cmd.stored manually
cmd.stored = type('stored', (object,), {})()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from visualize_attention_data import get_attention_file_path, load_attention_map


# ========== Attention File I/O ==========
load_all_heads = load_attention_map


def load_connections(connections_file, top_k=None):
    """
    Loads connections (res1, res2, weight) from a text file.
    Sorts by descending weight and selects top_k if specified.
    """
    connections = []

    with open(connections_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            res1 = int(row[0])
            res2 = int(row[1])
            weight = float(row[2])
            connections.append((res1, res2, weight))

    # Sort by weight descending
    connections.sort(key=lambda x: x[2], reverse=True)

    if top_k is not None:
        connections = connections[:top_k]

    return connections


def extract_head_number(filename):
    parts = filename.replace('.', '_').replace('-', '_').split('_')
    for i, part in enumerate(parts):
        if part.lower() == 'head' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
        if part.lower().startswith('head'):
            try:
                return int(part.lower().replace('head', ''))
            except ValueError:
                pass
    return -1


# ========== Residue Indexing and Geometry ==========
def check_residue_numbering(selection='all'):
    """
    Automatically checks if residue numbering matches array indices.
    """
    model = cmd.get_model(f"({selection}) and name CA")
    stored_residues = sorted(list(set(int(atom.resi) for atom in model.atom)))

    if not stored_residues:
        print("No residues found in selection!")
        return None

    # print(f"Detected residues: {stored_residues[:10]} ... (total {len(stored_residues)})")

    expected = list(range(1, len(stored_residues) + 1))
    if stored_residues == expected:
        # print("Residues are sequential starting at 1 — simple +1 mapping.")
        return 'plus_one'
    elif stored_residues[0] != 0 and stored_residues[0] != 1:
        # print(f"Residues start at {stored_residues[0]} — building index-to-residue mapping.")
        index_to_resi = {i: resi for i, resi in enumerate(stored_residues)}
        return index_to_resi
    else:
        print("Residue numbering unexpected — manual check recommended.")
        return None


def get_backbone_center(resi, selection='all'):
    coords = []
    for atom in ['N', 'CA', 'C']:
        try:
            coord = cmd.get_atom_coords(f"({selection}) and resi {resi} and name {atom}")
            coords.append(coord)
        except:
            continue
    if coords:
        return [sum(x)/len(x) for x in zip(*coords)]
    return None


def offset_point_pair(p1, p2, offset=0.1):
    v = np.array(p2) - np.array(p1)
    norm = np.linalg.norm(v)
    if norm == 0:
        return p1, p2
    unit = v / norm
    return (list(np.array(p1) + offset * unit), list(np.array(p2) - offset * unit))


# ========== Drawing Utilities ==========
def normalize_weight(w, w_min, w_max, r_min=0.15, r_max=0.7):
    if w_max != w_min:
        return r_min + ((w - w_min) / (w_max - w_min)) * (r_max - r_min)
    return (r_min + r_max) / 2


def color_from_weight_monochrome(w, w_min, w_max, base_color=(0.0, 0.0, 1.0), invert=False):
    if w_max != w_min:
        norm_w = (w - w_min) / (w_max - w_min)
    else:
        norm_w = 0.5
    factor = norm_w if invert else (1.0 - norm_w)
    return [c * factor for c in base_color]


def draw_connections(connections, mapping, selection='all', base_color=(0.0, 0.0, 1.0)):
    obj = []
    weights = [w for _, _, w in connections]
    w_min, w_max = min(weights), max(weights)

    for res1_index, res2_index, weight in connections:
        if mapping == 'plus_one':
            res1 = res1_index + 1
            res2 = res2_index + 1
        elif isinstance(mapping, dict):
            res1 = mapping.get(res1_index)
            res2 = mapping.get(res2_index)
        else:
            res1, res2 = res1_index, res2_index

        coord1 = get_backbone_center(res1, selection)
        coord2 = get_backbone_center(res2, selection)

        if not coord1 or not coord2:
            continue

        coord1, coord2 = offset_point_pair(coord1, coord2, offset=0.2)
        radius = normalize_weight(weight, w_min, w_max)
        color = color_from_weight_monochrome(weight, w_min, w_max, base_color=base_color)

        obj.extend([
            CYLINDER,
            *coord1,
            *coord2,
            radius,
            *color,
            *color
        ])

    cmd.load_cgo(obj, 'connections')


def draw_highlight_residue(res_index, mapping, selection='all', radius=1.0, color=(1.0, 0.0, 0.0)):
    """
    Draw a sphere at the specified residue index.
    """
    if mapping == 'plus_one':
        resi = res_index + 1
    elif isinstance(mapping, dict):
        resi = mapping.get(res_index, res_index)
    else:
        resi = res_index

    coord = get_backbone_center(resi, selection)
    if coord:
        sphere_obj = [SPHERE, *coord, radius, *color]
        cmd.load_cgo(sphere_obj, f"highlight_residue_{resi}")
        print(f"Marked residue {resi} with sphere.")
    else:
        print(f"Could not find coords for residue {resi}")


# ========== Plotting Utilities ==========
def plot_attention_grid(image_paths, titles, rows, cols, figure_title, output_file):
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows), constrained_layout=True)
    axes = axes.flatten()

    for ax, img_path, title in zip(axes, image_paths, titles):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    for ax in axes[len(image_paths):]:
        ax.axis('off')

    fig.suptitle(figure_title, fontsize=14, weight='bold', y=1.02)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"Saved summary grid to {output_file}")


# ========== PyMOL Visualization ==========
def master_plot(pdb_file, connections, save_path,
                selection='all', base_color=(0.0, 0.0, 1.0),
                highlight_res_index=None):

    cmd.reinitialize()
    cmd.load(pdb_file, 'structure')

    # Check residue mapping for indexing
    mapping = check_residue_numbering(selection=selection)

    print(f"Drawing {len(connections)} connections")

    cmd.bg_color('white')
    cmd.show('cartoon', 'structure')
    cmd.color('gray80', 'structure')
    cmd.set('cartoon_transparency', 0.0)
    cmd.set('cartoon_side_chain_helper', 1)
    cmd.hide('lines', 'structure')

    cmd.set('ray_trace_mode', 1)
    cmd.set('ray_trace_gain', 0.1)
    cmd.set('specular', 0.3)
    cmd.set('ambient', 0.4)
    cmd.set('direct', 0.8)
    cmd.set('reflect', 0.1)

    draw_connections(connections, mapping, selection=selection,
                     base_color=base_color)
    
    if highlight_res_index is not None:
        draw_highlight_residue(highlight_res_index, mapping, selection=selection)

    cmd.orient()
    cmd.viewport(1920, 1080)
    cmd.ray(1920, 1080)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cmd.png(save_path, dpi=300)

    print(f"Saved snapshot to {save_path}")


def plot_pymol_attention_heads(
    pdb_file,
    attention_dir,
    output_dir,
    protein,
    attention_type="msa_row",
    residue_indices=None,
    top_k=50,
    layer_idx=47,
    ):
    """
    Generate PyMOL visualizations of MSA row or triangle start attention.
    
    Args:
        pdb_file (str): Path to input PDB.
        attention_dir (str): Directory with attention text files.
        output_dir (str): Where to save output PNGs.
        attention_type (str): 'msa_row' or 'triangle_start'.
        residue_indices (List[int]): Only for triangle_start.
        top_k (int): Max number of attention edges to show.
        layer_idx (int): Which layer's attention to visualize.
    """

    os.makedirs(output_dir, exist_ok=True)

    if attention_type == "msa_row":
        msa_file = get_attention_file_path(attention_dir, attention_type, layer_idx)
        msa_heads = load_attention_map(msa_file, top_k=top_k)

        image_paths = []
        for head_idx, connections in msa_heads.items():
            output_png = os.path.join(output_dir, f"msa_row_head_{head_idx}_layer_{layer_idx}_{protein}.png")
            master_plot(pdb_file, connections, output_png, base_color=(0.0, 0.0, 1.0))
            image_paths.append(output_png)

        # Subplot
        titles = [f"MSA Row Head {extract_head_number(p)}" for p in image_paths]
        subplot_path = os.path.join(output_dir, f"msa_row_heads_layer_{layer_idx}_{protein}_subplot.png")
        plot_attention_grid(image_paths, titles, rows=2, cols=4,
                            figure_title=f"{protein} MSA Row Attention Heads Layer {layer_idx}", output_file=subplot_path)

    elif attention_type == "triangle_start":
        assert residue_indices is not None, "Must supply residue_indices for triangle attention"

        for res_idx in residue_indices:
            tri_file = get_attention_file_path(
                attention_dir, attention_type, layer_idx, residue_idx=res_idx
            )
            if not os.path.exists(tri_file):
                print(f"[Warning] Missing attention file for residue {res_idx}")
                continue

            tri_heads = load_attention_map(tri_file, top_k=top_k)
            res_pngs = []
            for head_idx, connections in tri_heads.items():
                output_png = os.path.join(output_dir, f"tri_start_residue_{res_idx}_head_{head_idx}_layer_{layer_idx}_{protein}.png")
                master_plot(pdb_file, connections, output_png,
                            base_color=(0.0, 0.0, 1.0),
                            highlight_res_index=res_idx)
                res_pngs.append(output_png)

            # Subplot for this residue
            subplot_path = os.path.join(output_dir, f"triangle_start_residue_{res_idx}_layer_{layer_idx}_{protein}_subplot.png")
            titles = [f"Head {extract_head_number(p)}" for p in res_pngs]
            plot_attention_grid(res_pngs, titles, rows=1, cols=len(res_pngs),
                                figure_title=f"Triangle Start Attention Heads Layer {layer_idx} — Residue {res_idx}",
                                output_file=subplot_path)


if __name__ == "__main__":
    # pdb_file = '/projects/bekh/thayes/output_dirs/my_outputs_align_6KWC_layer47_test/predictions/6KWC_1_model_1_ptm_relaxed.pdb'
    pdb_file = '/projects/bekh/thayes/output_dirs/my_outputs_align_6KWC_demo_tri_18/predictions/6KWC_1_model_1_ptm_relaxed.pdb'
    attention_dir = '/projects/bekh/thayes/demo_attn_saves/6KWC_demo'
    output_msa = '/u/thayes/vizfold/demo_plots_msa_row'
    output_tri = '/u/thayes/vizfold/demo_plots_tri_start'
    residue_indices = [18, 39, 51, 79, 138, 159]
    layer_idx = 47
    topk = 50
    protein = '6KWC'

    # Run for MSA Row Attention
    print('Making visuals for MSA Row Attention...')
    plot_pymol_attention_heads(
        pdb_file=pdb_file,
        attention_dir=attention_dir,
        output_dir=output_msa,
        protein=protein,
        attention_type="msa_row",
        top_k=topk,
        layer_idx=layer_idx
    )

    # Run for Triangle Start Attention
    print('Making visuals for Triangle Start Attention...')
    plot_pymol_attention_heads(
        pdb_file=pdb_file,
        attention_dir=attention_dir,
        output_dir=output_tri,
        protein=protein,
        attention_type="triangle_start",
        residue_indices=residue_indices,
        top_k=topk,
        layer_idx=layer_idx
    )

    print()
