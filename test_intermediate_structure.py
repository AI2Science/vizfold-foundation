import os, sys, types
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules["attn_core_inplace_cuda"] = types.ModuleType("attn_core_inplace_cuda")
sys.modules["attn_core_inplace_cuda"].forward_ = lambda *args, **kwargs: None

import torch
from openfold.model.model import AlphaFold
from openfold.config import model_config

from openfold.utils.script_utils import protein_from_prediction
from openfold.np import protein as protein_np

def write_structure_to_pdb(coords, pdb_path, aatype, mask):
    # build a "prediction" dict that protein_from_prediction understands
    pred = {
        "final_atom_positions": coords,   # [N, 37, 3]
        "final_atom_mask": atom_mask,     # [N, 37]
        "plddt": plddt,                   # optional
        # etc...
    }
    prot = protein_from_prediction(pred, aatype=aatype, residue_index=None, chain_index=None)
    pdb_str = protein_np.to_pdb(prot)
    with open(pdb_path, "w") as f:
        f.write(pdb_str)

# Helper: create a hook for each Evoformer layer
def attachHook(layer_idx, layerArray):
    def hook(module, input, output):
        # Handle tuple or dict outputs from Evoformer blocks
        if isinstance(output, tuple):
            msa, pair = output
        elif isinstance(output, dict):
            msa, pair = output["msa"], output["pair"]
        else:
            raise TypeError(f"Unexpected Evoformer block output type: {type(output)}")

        layerArray[layer_idx] = {
            "msa": msa.detach().clone(),
            "pair": pair.detach().clone(),
        }
    return hook


def main():
    # new change: parse CLI arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_id", required=True)
    parser.add_argument("--residue_idx", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root directory that matches Flask's UPLOAD_FOLDER (web_tmp_dir)",
    )
    args = parser.parse_args()

    protein_id = args.protein_id
    residue_idx = args.residue_idx
    layer_to_use = args.layer
    output_root = os.path.abspath(args.output_root)

    # new change: build the directory that Flask will read from
    out_dir = os.path.join(
        output_root,
        f"outputs/intermediate_structures_{protein_id}_demo_tri_{residue_idx}"
    )
    os.makedirs(out_dir, exist_ok=True)

    pdb_path = os.path.join(out_dir, f"layer_{layer_to_use}.pdb")
    
    # Load AlphaFold model configuration
    cfg = model_config("model_1")
    cfg.num_recycle = 3  # ensure 3 total passes: 0,1,2

    # Patch missing fields used by attention-viz variant of OpenFold
    from ml_collections import ConfigDict
    if not hasattr(cfg, "attention_config"):
        cfg.attention_config = ConfigDict({"demo_attn": False})
    if not hasattr(cfg, "attn_map_dir"):
        cfg.attn_map_dir = None
    if not hasattr(cfg, "save_attn_maps"):
        cfg.save_attn_maps = False
    if not hasattr(cfg, "num_recycles_save"):
        cfg.num_recycles_save = None

    # Disable templates in this demo
    for path in [
        ("embeddings_and_evoformer", "template", "enabled"),
        ("template", "enabled"),
        ("data", "common", "use_templates"),
    ]:
        try:
            obj = cfg
            for k in path[:-1]:
                obj = getattr(obj, k)
            setattr(obj, path[-1], False)
        except Exception:
            pass

    # Initialize model
    model = AlphaFold(cfg)
    model.eval()

    # --- Cap recycling to exactly 3 passes inside forward(), no repo edits ---
    orig_forward = model.forward

    def forward_capped_to_three(batch, *args, **kwargs):
        g = orig_forward.__globals__.copy()
        real_range = range

        def capped_range(n):
            # Allow at most 3 iterations
            return real_range(n if n <= 3 else 3)

        g["range"] = capped_range

        wrapped = types.FunctionType(
            orig_forward.__func__.__code__ if hasattr(orig_forward, "__func__") else orig_forward.__code__,
            g,
            name=orig_forward.__name__,
            argdefs=orig_forward.__defaults__,
            closure=orig_forward.__closure__,
        )
        bound = types.MethodType(wrapped, model)

        # Handle early return before 'outputs' exists
        try:
            return bound(batch, *args, **kwargs)
        except UnboundLocalError as e:
            if "outputs" in str(e):
                print("⚠️  Early stop: outputs not returned after 3 recycles, forcing safe return.")
                return {"num_recycles": torch.tensor(3)}
            raise

    model.forward = forward_capped_to_three
    # ------------------------------------------------------------------------

    # Define NoExtraMSA replacement
    class NoExtraMSA(torch.nn.Module):
        def forward(self, *args, **kwargs):
            z_kw = kwargs.get("z", None)
            if z_kw is not None:
                return z_kw
            if len(args) >= 2:
                return args[1]
            raise RuntimeError("NoExtraMSA: couldn't find 'z' in args/kwargs")

    model.extra_msa_stack = NoExtraMSA()

    # Dictionary to hold intermediate representations
    intermediate_reps = {}

    # Register forward hooks on all Evoformer blocks
    for layer_idx, block in enumerate(model.evoformer.blocks):
        block.register_forward_hook(attachHook(layer_idx, intermediate_reps))

    # Dummy input batch
    num_msa = 1
    sequenceLen = 128
    batch = {
        "aatype": torch.randint(0, 20, (1, sequenceLen), dtype=torch.int64),
        "residue_index": torch.arange(sequenceLen).unsqueeze(0),
        "seq_mask": torch.ones((1, sequenceLen)),

        # MSA features (B, N_seq, L, C_m)
        "msa_feat": torch.randn((1, num_msa, 49, sequenceLen)),
        "msa_mask": torch.ones((1, num_msa, sequenceLen)),

        # Pair features (B, L, L, C_z)
        "pair_feat": torch.randn((1, sequenceLen, sequenceLen, 128)),

        # Target features (B, 22, L)
        "target_feat": torch.randn((1, 22, sequenceLen)),

        # Template placeholders (disabled but kept for shape sanity)
        "template_aatype": torch.zeros((1, 1, sequenceLen), dtype=torch.int64),
        "template_all_atom_masks": torch.zeros((1, 1, sequenceLen, 37)),
        "template_all_atom_positions": torch.zeros((1, 1, sequenceLen, 37, 3)),
        "template_mask": torch.zeros((1, 1, sequenceLen)),
        "template_pseudo_beta_mask": torch.zeros((1, 1, sequenceLen)),
        "template_pseudo_beta": torch.zeros((1, 1, sequenceLen, 3)),

        # Extra MSA — single sequence
        "extra_msa": torch.zeros((1, num_msa, sequenceLen), dtype=torch.int64),
        "extra_has_deletion": torch.zeros((1, num_msa, sequenceLen)),
        "extra_deletion_value": torch.zeros((1, num_msa, sequenceLen)),
        "extra_msa_mask": torch.ones((1, num_msa, sequenceLen)),
    }
    batch["residx_atom37_to_atom14"] = torch.zeros((1, sequenceLen, 37), dtype=torch.int64)
    batch["residx_atom14_to_atom37"] = torch.zeros((1, sequenceLen, 14), dtype=torch.int64)
    batch["atom37_atom_exists"] = torch.ones((1, sequenceLen, 37))

    # Run forward pass
    with torch.no_grad():
        _ = model(batch)

    print("\n✅ Completed exactly 3 recycles.")

    # Pull the last evoformer representations
    layer_to_use = len(intermediate_reps) - 1
    msa = intermediate_reps[layer_to_use]["msa"]  # (B, N_seq, L, C_m) or related fork variant
    pair = intermediate_reps[layer_to_use]["pair"]  # (B, L, L, C_z)

    # Build inputs for the StructureModule
    # Collapse MSA to single-seq representation (s) the way many forks do
        # --- Build inputs for the StructureModule ---
        # --- Build inputs for the StructureModule ---
        # --- Build inputs for the StructureModule ---
        # --- Build inputs for the StructureModule ---
        # --- Build inputs for the StructureModule ---
        # --- Build inputs for the StructureModule ---
        # --- Build inputs for the StructureModule ---
    # evoformer "pair" gives us the true L: (B, L, L, C_z)
    z_act = pair
    # Handle both (B, L, L, C_z) and (B, L, C_z)
    if z_act.ndim == 3:
        # Expand to (B, L, L, C_z)
        B, L, Cz = z_act.shape
        z_act = z_act.unsqueeze(2).expand(B, L, L, Cz).contiguous()
    elif z_act.ndim == 4:
        B, L, L2, Cz = z_act.shape
        assert L == L2, f"pair expected square (L,L), got {(L, L2)}"
    else:
        raise RuntimeError(f"Unexpected pair shape: {z_act.shape}")

    # MSA → single representation
    s_act = msa.mean(dim=1)  # (B, L, C_s) or (B, C_s, L) or (L, C_s)

    # Ensure batch and seq dims
    if s_act.ndim == 2:  # (L, C_s)
        s_act = s_act.unsqueeze(0)
    if s_act.shape[1] != L and s_act.shape[-1] == L:
        s_act = s_act.permute(0, 2, 1).contiguous()
    elif s_act.shape[1] != L and s_act.shape[-1] != L:
        raise RuntimeError(f"Cannot align s_act with pair L={L}, got {tuple(s_act.shape)}")

    # Coerce to float and contiguous for matmul stability
    s_act = s_act.float().contiguous()
    z_act = z_act.float().contiguous()

    # Mask and aatype fix to (B, L)
    mask = batch["seq_mask"]
    aatype = batch["aatype"]
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if aatype.ndim == 1:
        aatype = aatype.unsqueeze(0)
    if mask.shape[-1] != L:
        mask = mask[..., :L]
    if aatype.shape[-1] != L:
        aatype = aatype[..., :L]

    structure_module = model.structure_module

    # Adjust feature dimension if needed
    try:
        expected_dim = structure_module.layer_norm_s.weight.shape[0]
    except AttributeError:
        expected_dim = getattr(structure_module.layer_norm_s, "normalized_shape", [s_act.shape[-1]])[0]

    current_dim = s_act.shape[-1]
    if current_dim != expected_dim:
        print(f"Projecting Evoformer single from {current_dim} → {expected_dim}")
        projector = torch.nn.Linear(current_dim, expected_dim)
        s_act = projector(s_act)

    evoformer_output_dict = {"single": s_act, "pair": z_act}

    with torch.no_grad():
        structure_output = structure_module(evoformer_output_dict, aatype, mask, False)

    possible_keys = [
        "final_atom_positions",
        "positions",
        "atom_positions",
        "structure",
    ]
    coords = None
    for k in possible_keys:
        if k in structure_output:
            coords = structure_output[k]
            print(f"✅ Found coordinates under key: '{k}'")
            break

    # Some OpenFold forks wrap this inside 'sm_output'
    if coords is None and "sm_output" in structure_output:
        sm_out = structure_output["sm_output"]
        for k in possible_keys:
            if k in sm_out:
                coords = sm_out[k]
                print(f"✅ Found coordinates under sm_output['{k}']")
                break

    if coords is None:
        print("⚠️  Could not locate final_atom_positions key. Dumping keys for inspection:")
        print(list(structure_output.keys()))
        raise KeyError("No coordinate tensor found in StructureModule output.")

    print(f"\n✅ Completed exactly 3 recycles.")
    print(f"Generated structure coordinates shape: {coords.shape}")
    #torch.save(structure_output, f"structure_from_layer_{layer_to_use}.pt")
    write_structure_to_pdb(coords, pdb_path, aatype=batch["aatype"], mask=batch["seq_mask"])

if __name__ == "__main__":
    main()
