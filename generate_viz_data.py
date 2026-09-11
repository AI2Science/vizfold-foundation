import argparse
import logging
import math
import numpy as np
import os
import pickle
import random
import time
import json
import torch
import torch.nn.functional as F

# OpenFold Imports
from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.data.tools import hhsearch, hmmsearch
from openfold.np import protein
from openfold.utils.script_utils import (
    load_models_from_command_line,
    parse_fasta,
    run_model,
    prep_output,
    relax_protein
)
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.model.primitives import Attention
from scripts.precompute_embeddings import EmbeddingGenerator
from scripts.utils import add_data_args

ATTENTION_STORE = {}

def get_custom_attention_forward(original_forward):
    """
    Wrapper for OpenFold's Attention.forward.
    Captures the attention matrix (Softmax(QK^T)) before it is consumed.
    """
    def custom_forward(self, q, k, v, mask=None, bias=None):
        if hasattr(self, 'viz_config'):
            layer_idx = self.viz_config['layer_idx']
            attn_type = self.viz_config['type']

            with torch.no_grad():
                scaling = 1.0 / (q.shape[-1] ** 0.5)
                
                # Q * K^T
                logits = torch.matmul(q, k.transpose(-2, -1)) * scaling
                
                if bias is not None: logits += bias
                if mask is not None: logits += mask
                
                # Softmax to get probabilities
                attn_weights = torch.softmax(logits, dim=-1)
                
                # Store as FP16 to save memory. 
                # Shape: [Heads, Seq, Seq] (remove batch dim if present)
                if attn_weights.dim() == 4:
                    weights_cpu = attn_weights[0].detach().cpu().half()
                else:
                    weights_cpu = attn_weights.detach().cpu().half()
                
                ATTENTION_STORE[(layer_idx, attn_type)] = weights_cpu

        # Continue with original forward pass
        return original_forward(self, q, k, v, mask=mask, bias=bias)
    
    return custom_forward

def register_viz_patches(model):
    """
    Tags model layers so we know which attention map belongs to which block.
    """
    print("Applying visualization patches to OpenFold model...")
    
    tagged_count = 0
    for name, module in model.named_modules():
        # Look for Evoformer blocks
        if "evoformer.blocks" in name and isinstance(module, Attention):
            parts = name.split('.')
            try:
                # e.g. 'evoformer.blocks.0.msa_att_row'
                block_idx = int(parts[2])
                
                attn_type = None
                if "msa_att_row" in name: attn_type = "msa_row_attention"
                elif "msa_att_col" in name: attn_type = "msa_col_attention"
                elif "pair_att" in name: attn_type = "pair_attention"
                
                if attn_type:
                    module.viz_config = {'layer_idx': block_idx, 'type': attn_type}
                    tagged_count += 1
            except (IndexError, ValueError):
                pass
    
    print(f"Tagged {tagged_count} attention modules.")
    
    # Monkey patch the class
    Attention.original_forward = Attention.forward
    Attention.forward = get_custom_attention_forward(Attention.original_forward)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

torch_versions = torch.__version__.split(".")
if int(torch_versions[0]) > 1 or (int(torch_versions[0]) == 1 and int(torch_versions[1]) >= 12):
    torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)
TRACING_INTERVAL = 50

def precompute_alignments(tags, seqs, alignment_dir, args):
    # Standard OpenFold alignment logic
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
    for tag, seq in zip(tags, seqs):
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")
        local_alignment_dir = os.path.join(alignment_dir, tag)

        if args.use_precomputed_alignments is None:
            logger.info(f"Generating alignments for {tag}...")
            os.makedirs(local_alignment_dir, exist_ok=True)
            
            # Simplified for brevity - assumes standard HHSearch/Jackhmmer setup
            # (If you need the complex DB selection logic, copy it from original script)
            # This block just calls the runners if they exist.
            pass 
            
        os.remove(tmp_fasta_path)

def generate_feature_dict(tags, seqs, alignment_dir, data_processor, args):
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
    with open(tmp_fasta_path, "w") as fp:
        fp.write(f">{tags[0]}\n{seqs[0]}")
    
    local_alignment_dir = os.path.join(alignment_dir, tags[0])
    feature_dict = data_processor.process_fasta(
        fasta_path=tmp_fasta_path,
        alignment_dir=local_alignment_dir,
        seqemb_mode=args.use_single_seq_mode,
    )
    os.remove(tmp_fasta_path)
    return feature_dict

def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]

def main(args):
    if args.use_deepspeed_evoformer_attention:
        logger.warning("FORCING use_deepspeed_evoformer_attention=False.")
        args.use_deepspeed_evoformer_attention = False

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.attn_map_dir, exist_ok=True)

    config = model_config(args.config_preset)
    
    # Load Model
    model_generator = load_models_from_command_line(
        config, args.model_device, args.openfold_checkpoint_path, args.jax_param_path, args.output_dir
    )

    # Prepare Data Pipeline
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=config.data.predict.max_templates,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=args.release_dates_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path,
    )
    data_processor = data_pipeline.DataPipeline(template_featurizer)

    # Parse Input
    tag_list = []; seq_list = []
    for fasta_file in list_files_with_extensions(args.fasta_dir, (".fasta", ".fa")):
        with open(os.path.join(args.fasta_dir, fasta_file), "r") as fp:
            data = fp.read()
        tags, seqs = parse_fasta(data)
        tag_list.append(('-'.join(tags), tags))
        seq_list.append(seqs)

    for model, output_directory in model_generator:
        
        register_viz_patches(model)

        for (tag, tags), seqs in tag_list:
            ATTENTION_STORE.clear()

            # Alignments (Assuming precomputed for simplicity in this demo script)
            alignment_dir = os.path.join(args.output_dir, "alignments")
            feature_dict = generate_feature_dict(tags, seqs, alignment_dir, data_processor, args)
            
            processed_feature_dict = feature_pipeline.FeaturePipeline(config.data).process_features(feature_dict, mode="predict")
            processed_feature_dict = {k: torch.as_tensor(v, device=args.model_device) for k,v in processed_feature_dict.items()}

            logger.info(f"Running model for {tag}...")
            out = run_model(model, processed_feature_dict, tag, args.output_dir)

            # save the formatted data for vizfold
            viz_data = {}
            num_blocks = 48 

            def stack_attention(type_key):
                layers = []
                for i in range(num_blocks):
                    if (i, type_key) in ATTENTION_STORE:
                        layers.append(ATTENTION_STORE[(i, type_key)].numpy())
                    else:
                        return None
                return np.stack(layers) # Shape: [48, Heads, Seq, Seq]

            if (res := stack_attention("msa_row_attention")) is not None:
                viz_data["msa_row_attention"] = res
            if (res := stack_attention("msa_col_attention")) is not None:
                viz_data["msa_col_attention"] = res
            if (res := stack_attention("pair_attention")) is not None:
                viz_data["pair_attention"] = res

            attn_path = os.path.join(args.attn_map_dir, f"{tag}_attention.pkl")
            with open(attn_path, "wb") as f:
                pickle.dump(viz_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Visualization data saved to {attn_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_dir", type=str)
    parser.add_argument("template_mmcif_dir", type=str)
    parser.add_argument("--output_dir", type=str, default=os.getcwd())
    parser.add_argument("--model_device", type=str, default="cpu")
    parser.add_argument("--config_preset", type=str, default="model_1")
    parser.add_argument("--jax_param_path", type=str, default=None)
    parser.add_argument("--openfold_checkpoint_path", type=str, default=None)
    parser.add_argument("--attn_map_dir", type=str, default="attention_maps")
    parser.add_argument("--use_deepspeed_evoformer_attention", action="store_true", default=False)
    # TODO: Add more arguments when we need them
    add_data_args(parser)
    args = parser.parse_args()

    if args.jax_param_path is None and args.openfold_checkpoint_path is None:
        args.jax_param_path = os.path.join("openfold", "resources", "params", f"params_{args.config_preset}.npz")

    main(args)