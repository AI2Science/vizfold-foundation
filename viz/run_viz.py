import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from visualize_attention_general_utils import render_pdb_to_image, generate_combined_attention_panels
from visualize_attention_3d_demo_utils import plot_pymol_attention_heads
from visualize_attention_arc_diagram_demo_utils import generate_arc_diagrams, parse_fasta_sequence

PROT = "6KWC"
TRI_RESIDUE_IDX = 18
LAYER_IDX = 47
TOP_K = 50

ATTN_MAP_DIR = f"./outputs/attention_files_{PROT}_demo_tri_{TRI_RESIDUE_IDX}"
OUTPUT_DIR = f"./outputs/my_outputs_align_{PROT}_demo_tri_{TRI_RESIDUE_IDX}"
IMAGE_OUTPUT_DIR = f"./outputs/attention_images_{PROT}_demo_tri_{TRI_RESIDUE_IDX}"
PDB_FILE = os.path.join(OUTPUT_DIR, f"predictions/{PROT}_1_model_1_ptm_relaxed.pdb")
FASTA_PATH = f"./examples/monomer/fasta_dir_{PROT}/{PROT}.fasta"

output_dir_msa = os.path.join(IMAGE_OUTPUT_DIR, 'msa_row_attention_plots')
output_dir_tri = os.path.join(IMAGE_OUTPUT_DIR, 'tri_start_attention_plots')

print("Step 1: Rendering 3D structure...")
render_pdb_to_image(PDB_FILE, IMAGE_OUTPUT_DIR, f"predicted_structure_{PROT}_tri_{TRI_RESIDUE_IDX}.png")

print("Step 2: Generating MSA 3D attention plots...")
plot_pymol_attention_heads(
    pdb_file=PDB_FILE, attention_dir=ATTN_MAP_DIR, output_dir=output_dir_msa,
    protein=PROT, attention_type="msa_row", top_k=TOP_K, layer_idx=LAYER_IDX
)

print("Step 3: Generating triangle 3D attention plots...")
plot_pymol_attention_heads(
    pdb_file=PDB_FILE, attention_dir=ATTN_MAP_DIR, output_dir=output_dir_tri,
    protein=PROT, attention_type="triangle_start", residue_indices=[TRI_RESIDUE_IDX],
    top_k=TOP_K, layer_idx=LAYER_IDX
)

print("Step 4: Parsing FASTA for arc diagrams...")
residue_seq = parse_fasta_sequence(FASTA_PATH)

print("Step 5: Generating MSA arc diagrams...")
generate_arc_diagrams(
    attention_dir=ATTN_MAP_DIR, residue_sequence=residue_seq, output_dir=output_dir_msa,
    protein=PROT, attention_type="msa_row", top_k=TOP_K, layer_idx=LAYER_IDX
)

print("Step 6: Generating triangle arc diagrams...")
generate_arc_diagrams(
    attention_dir=ATTN_MAP_DIR, residue_sequence=residue_seq, output_dir=output_dir_tri,
    protein=PROT, attention_type="triangle_start", residue_indices=[TRI_RESIDUE_IDX],
    top_k=TOP_K, layer_idx=LAYER_IDX
)

print("Step 7: Combining panels...")
generate_combined_attention_panels(
    attention_type="msa_row", protein=PROT, layer_idx=LAYER_IDX,
    output_dir_3d=output_dir_msa, output_dir_arc=output_dir_msa,
    combined_output_dir=IMAGE_OUTPUT_DIR
)
generate_combined_attention_panels(
    attention_type="triangle_start", protein=PROT, layer_idx=LAYER_IDX,
    output_dir_3d=output_dir_tri, output_dir_arc=output_dir_tri,
    combined_output_dir=IMAGE_OUTPUT_DIR, residue_indices=[TRI_RESIDUE_IDX]
)

print("Done! Images saved to", IMAGE_OUTPUT_DIR)
