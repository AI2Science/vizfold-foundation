import os
import numpy as np
from pymol import cmd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from visualize_attention_3d_demo_utils import extract_head_number


def make_fasta_file(FASTA_PATH, FASTA_DIR, FASTA_SEQUENCE):
    # Make sure output directory exists
    os.makedirs(FASTA_DIR, exist_ok=True)

    # Write FASTA to disk
    with open(FASTA_PATH, "w") as f:
        f.write(FASTA_SEQUENCE)

    print(f"Saved FASTA to {FASTA_PATH}")


def render_pdb_to_image(pdb_file, output_image_path, fname):
    os.makedirs(output_image_path, exist_ok=True)
    output_image_path = os.path.join(output_image_path, fname)

    cmd.reinitialize()
    cmd.load(pdb_file, 'protein')

    cmd.bg_color('white')
    cmd.show('cartoon', 'protein')
    cmd.color('gray80', 'protein')
    cmd.hide('lines', 'protein')
    cmd.orient()

    cmd.viewport(800, 600)
    cmd.ray(800, 600)
    cmd.png(output_image_path, dpi=300)
    print(f"Saved image to {output_image_path}")
    
    show_image(output_image_path, 'Predicted Structure')


def show_image(image_path, title=None):
    img = mpimg.imread(image_path)
    plt.figure(figsize=(6, 5))
    plt.imshow(img)
    if title:
        plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()


def combine_3d_and_arc_images(
    structure_img_path,
    arc_img_path,
    output_path,
    title_top="3D Attention",
    title_bottom="Arc Diagram",
    fig_title=None,
    show_plot=True
):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)

    for ax, img_path, title in zip(axes, [structure_img_path, arc_img_path], [title_top, title_bottom]):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12, weight='bold')
        ax.axis('off')

    if fig_title:
        fig.suptitle(fig_title, fontsize=16, weight='bold', y=1.03)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"[Saved] Combined panel to {output_path}")


def generate_combined_attention_panels(
    attention_type,  # "msa_row" or "triangle_start"
    protein,
    layer_idx,
    output_dir_3d,
    output_dir_arc,
    combined_output_dir,
    residue_indices=None  # for triangle_start
):
    os.makedirs(combined_output_dir, exist_ok=True)

    if attention_type == "msa_row":
        for fname in os.listdir(output_dir_3d):
            if fname.startswith("msa_row_head_") and fname.endswith(f"_{protein}.png") and f"layer_{layer_idx}_" in fname:
                head = extract_head_number(fname)
                prefix = f"msa_row_head_{head}_layer_{layer_idx}_{protein}"
                struct_path = os.path.join(output_dir_3d, f"{prefix}.png")
                arc_path = os.path.join(output_dir_arc, f"{prefix}_arc.png")
                out_path = os.path.join(combined_output_dir, f"{prefix}_combo.png")
                print('\n')
                print(struct_path)
                print(arc_path)
                print(out_path)
                if os.path.exists(struct_path) and os.path.exists(arc_path):
                    generate_title = f"MSA Row — Head {head}, Layer {layer_idx}, {protein}"
                    combine_3d_and_arc_images(struct_path, arc_path, out_path, fig_title=generate_title)
                else:
                    print(f"[Skipped] Missing image for {prefix}")

    elif attention_type == "triangle_start":
        assert residue_indices is not None
        for res_idx in residue_indices:
            for fname in os.listdir(output_dir_3d):
                if fname.startswith(f"tri_start_residue_{res_idx}_head_") and fname.endswith(f"_{protein}.png") and f"layer_{layer_idx}_" in fname:
                    head = extract_head_number(fname)
                    prefix = f"tri_start_residue_{res_idx}_head_{head}_layer_{layer_idx}_{protein}"
                    struct_path = os.path.join(output_dir_3d, f"{prefix}.png")
                    arc_path = os.path.join(output_dir_arc, f"tri_start_res_{res_idx}_head_{head}_layer_{layer_idx}_{protein}_arc.png")
                    out_path = os.path.join(combined_output_dir, f"{prefix}_combo.png")
                    print('\n')
                    print(struct_path)
                    print(arc_path)
                    print(out_path)
                    if os.path.exists(struct_path) and os.path.exists(arc_path):
                        generate_title = f"Triangle Start — Head {head}, Res {res_idx}, Layer {layer_idx}, {protein}"
                        combine_3d_and_arc_images(struct_path, arc_path, out_path, fig_title=generate_title)
                    else:
                        print(f"[Skipped] Missing image for {prefix}")


def compute_ca_distance_matrix(pdb_file, chain_id="A"):
    """Pairwise Cα–Cα distance matrix (Å) for one chain of a predicted structure."""
    from Bio import PDB

    structure = PDB.PDBParser(QUIET=True).get_structure("protein", pdb_file)
    coords = np.array([
        residue["CA"].get_vector().get_array()
        for model_ in structure
        for chain in model_ if chain.id == chain_id
        for residue in chain
        if PDB.is_aa(residue, standard=True) and "CA" in residue
    ])
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def render_contact_attention_panel(
    pdb_file,
    output_path,
    attention_map=None,
    chain_id="A",
    contact_threshold=8.0,
    region=None,
    title=None,
    show_plot=True,
):
    """Cα distance map, contact map, and — when an NxN ``attention_map`` is supplied —
    an attention overlay with contacts marked, saved as one figure.

    The caller passes an already-aggregated attention map (mean over heads/layers)
    rather than re-running inference here. ``region`` is an optional ``(start, end)``
    residue window. Lifted from the contact-map work in closed PR #75.
    """
    dist = compute_ca_distance_matrix(pdb_file, chain_id)
    n = dist.shape[0]
    r0, r1 = region if region else (0, n)
    extent = [r0 - 0.5, r1 - 0.5, r0 - 0.5, r1 - 0.5]
    dist_r = dist[r0:r1, r0:r1]

    n_panels = 3 if attention_map is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))

    im0 = axes[0].imshow(dist_r, cmap="viridis_r", origin="lower", extent=extent)
    axes[0].set_title("Cα distance map")
    plt.colorbar(im0, ax=axes[0], label="Distance (Å)")

    axes[1].imshow(dist_r < contact_threshold, cmap="Greys", origin="lower", extent=extent)
    axes[1].set_title(f"Contact map (Cα < {contact_threshold} Å)")

    if attention_map is not None:
        attn = np.asarray(attention_map)
        if attn.shape != (n, n):
            raise ValueError(f"attention_map is {attn.shape}, expected ({n}, {n}) to match the structure")
        attn_r = ((attn + attn.T) / 2)[r0:r1, r0:r1]
        im2 = axes[2].imshow(attn_r, cmap="hot_r", origin="lower", extent=extent)
        ci, cj = np.where(dist_r < contact_threshold)
        axes[2].scatter(cj + r0, ci + r0, s=0.3, c="cyan", alpha=0.5, label=f"Cα < {contact_threshold} Å")
        axes[2].set_title("Attention vs. contacts")
        axes[2].legend(markerscale=10, fontsize=8)
        plt.colorbar(im2, ax=axes[2], label="Attention weight")

    for ax in axes:
        ax.set_xlabel("Residue index")
        ax.set_ylabel("Residue index")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    print(f"[Saved] Contact/attention panel to {output_path}")