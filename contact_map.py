import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from Bio import PDB

# Repo root — all relative paths are anchored here
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.data.data_pipeline import DataPipeline
from openfold.data.feature_pipeline import FeaturePipeline
from openfold.model.triangular_attention import TriangleAttention
from openfold.model.primitives import Attention, softmax_no_cast, permute_final_dims
import openfold.model.primitives as primitives

# Config is passed via environment variables by generate_contact_map.py.
def _require(var):
    val = os.environ.get(var)
    if not val:
        sys.exit(f"error: {var} not set — run this script via generate_contact_map.py")
    return val

CONFIG_PRESET     = _require("CM_CONFIG_PRESET")
FASTA_PATH        = _require("CM_FASTA_PATH")
ALIGNMENT_DIR     = _require("CM_ALIGNMENT_DIR")
OUTPUT_DIR        = os.environ.get("CM_OUTPUT_DIR",         os.path.join(SCRIPT_DIR, "outputs"))
PARAMS_PATH       = os.environ.get("CM_PARAMS_PATH",        os.path.join(SCRIPT_DIR, "openfold", "resources", "params", f"params_{CONFIG_PRESET}.npz"))
CHAIN_ID          = os.environ.get("CM_CHAIN_ID",           "A")
LAYER_START       = int(os.environ.get("CM_LAYER_START",    "38"))
LAYER_END         = int(os.environ.get("CM_LAYER_END",      "47"))
CONTACT_THRESHOLD = float(os.environ.get("CM_CONTACT_THRESHOLD", "8.0"))
REGION_START      = int(os.environ["CM_REGION_START"]) if "CM_REGION_START" in os.environ else None
REGION_END        = int(os.environ["CM_REGION_END"])   if "CM_REGION_END"   in os.environ else None

protein_tag = os.path.splitext(os.path.basename(FASTA_PATH))[0].upper()
PDB_PATH    = os.environ.get("CM_PDB_PATH", os.path.join(OUTPUT_DIR, "predictions", f"{protein_tag}_1_{CONFIG_PRESET}_relaxed.pdb"))


# 1  Contact map from PDB
def get_distance_matrix(pdb_path, chain_id="A"):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    coords = []
    for model_ in structure:
        for chain in model_:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if PDB.is_aa(residue, standard=True) and "CA" in residue:
                    coords.append(residue["CA"].get_vector().get_array())
    coords = np.array(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

dist_matrix = get_distance_matrix(PDB_PATH, CHAIN_ID)
N = dist_matrix.shape[0]
print(f"Protein: {protein_tag}  |  Length: {N} residues")


# 2  Patch _attention to capture softmax weights
_capture = [False]
_attn_weights_store = []

_orig_attention = primitives._attention

def _patched_attention(query, key, value, biases):
    key_t = permute_final_dims(key, (1, 0))
    a = torch.matmul(query, key_t)
    for b in biases:
        a += b
    a = softmax_no_cast(a, -1)
    if _capture[0]:
        _attn_weights_store.append(a.detach().cpu())
    return torch.matmul(a, value)

primitives._attention = _patched_attention
Attention.forward.__globals__["_attention"] = _patched_attention


# 3  Load model
config = model_config(CONFIG_PRESET)
model  = AlphaFold(config).eval().cuda()
import_jax_weights_(model, PARAMS_PATH, version=CONFIG_PRESET)
print("Model loaded.")


# 4  Register hooks to toggle capture for target layers only
def get_layer_idx(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "blocks":
            try:
                return int(parts[i + 1])
            except (ValueError, IndexError):
                pass
    return -1

hooks = []

for name, module in model.named_modules():
    if not isinstance(module, TriangleAttention):
        continue
    if not name.startswith("evoformer"):
        continue
    if not (LAYER_START <= get_layer_idx(name) <= LAYER_END):
        continue

    def make_hooks():
        def pre_hook(_module, _input):
            _capture[0] = True
        def post_hook(_module, _input, _output):
            _capture[0] = False
        return pre_hook, post_hook

    pre, post = make_hooks()
    hooks.append(module.register_forward_pre_hook(pre))
    hooks.append(module.register_forward_hook(post))

print(f"Hooks registered on {len(hooks) // 2} TriangleAttention modules "
      f"(layers {LAYER_START}–{LAYER_END}).")


# 5  Run inference
data_proc    = DataPipeline(template_featurizer=None)
feature_proc = FeaturePipeline(config.data)
features     = data_proc.process_fasta(fasta_path=FASTA_PATH, alignment_dir=ALIGNMENT_DIR)
processed    = feature_proc.process_features(features, mode="predict")
batch        = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
                for k, v in processed.items()}

print("Running inference...")
with torch.no_grad():
    model(batch)

for h in hooks:
    h.remove()

primitives._attention = _orig_attention
Attention.forward.__globals__["_attention"] = _orig_attention

print(f"Captured {len(_attn_weights_store)} attention weight tensors.")


# 6  Aggregate: mean over heads and recycles, then over layers
if not _attn_weights_store:
    raise RuntimeError("No attention weights captured — check LAYER_START/LAYER_END.")

tensors = []
for t in _attn_weights_store:
    t = t.squeeze(0)
    if t.shape[-1] != N or t.shape[-2] != N:
        raise ValueError(f"Unexpected shape {list(t.shape)}, expected last two dims {N}")
    tensors.append(t.reshape(-1, N, N).float().mean(dim=0).numpy())

attn_map = np.stack(tensors).mean(axis=0)
attn_map = (attn_map + attn_map.T) / 2
print(f"Final attention map shape: {attn_map.shape}")


# 7  Plot
r0 = REGION_START if REGION_START is not None else 0
r1 = REGION_END   if REGION_END   is not None else N

dist_plot = dist_matrix[r0:r1, r0:r1]
attn_plot = attn_map[r0:r1, r0:r1]

# extent keeps axis labels in actual residue-index space
extent = [r0 - 0.5, r1 - 0.5, r0 - 0.5, r1 - 0.5]
region_label = f"residues {r0}–{r1 - 1}" if (REGION_START is not None or REGION_END is not None) else "all residues"

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

im1 = axes[0].imshow(dist_plot, cmap="viridis_r", origin="lower", extent=extent)
axes[0].set_title("Cα Distance Map")
axes[0].set_xlabel("Residue index")
axes[0].set_ylabel("Residue index")
plt.colorbar(im1, ax=axes[0], label="Distance (Å)")

axes[1].imshow(dist_plot < CONTACT_THRESHOLD, cmap="Greys", origin="lower", extent=extent)
axes[1].set_title(f"Contact Map (Cα < {CONTACT_THRESHOLD} Å)")
axes[1].set_xlabel("Residue index")
axes[1].set_ylabel("Residue index")

contacts_i, contacts_j = np.where(dist_plot < CONTACT_THRESHOLD)
im3 = axes[2].imshow(attn_plot, cmap="hot_r", origin="lower", extent=extent)
axes[2].scatter(contacts_j + r0, contacts_i + r0, s=0.3, c="cyan", alpha=0.5,
                label=f"Cα < {CONTACT_THRESHOLD} Å")
axes[2].set_title(f"Triangle Attention (layers {LAYER_START}–{LAYER_END})\n"
                  "Mean over heads + recycles + layers")
axes[2].set_xlabel("Residue index")
axes[2].set_ylabel("Residue index")
plt.colorbar(im3, ax=axes[2], label="Attention weight")
axes[2].legend(markerscale=10, fontsize=8)

plt.suptitle(f"{protein_tag} — Contact Map + Triangle Attention Weights ({region_label})",
             fontsize=13, fontweight="bold")
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
region_suffix = f"_r{r0}-{r1 - 1}" if (REGION_START is not None or REGION_END is not None) else ""
out_name = os.path.join(OUTPUT_DIR, f"contact_attention_{protein_tag}_{LAYER_START}-{LAYER_END}{region_suffix}.png")
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"Saved to {out_name}")
plt.show()
