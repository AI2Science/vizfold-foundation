# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Sequence, Optional
from functools import partial
from abc import ABC, abstractmethod

from openfold.model.primitives import Linear, LayerNorm, ATTENTION_METADATA
from openfold.model.dropout import DropoutRowwise, DropoutColumnwise
from openfold.model.msa import (
    MSAAttention,
    MSARowAttentionWithPairBias,
    MSAColumnAttention,
    MSAColumnGlobalAttention,
)
from openfold.model.outer_product_mean import OuterProductMean
from openfold.model.pair_transition import PairTransition
from openfold.model.triangular_attention import (
    TriangleAttention,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    FusedTriangleMultiplicationIncoming,
    FusedTriangleMultiplicationOutgoing
)
from openfold.utils.checkpointing import checkpoint_blocks, get_checkpoint_fn
from openfold.utils.chunk_utils import chunk_layer, ChunkSizeTuner
from openfold.utils.tensor_utils import add


class MSATransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.

    Implements Algorithm 9
    """
    def __init__(self, c_m, n):
        """
        Args:
            c_m:
                MSA channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super(MSATransition, self).__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init="final")

    def _transition(self, m, mask):
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    @torch.jit.ignore
    def _chunk(self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
         return chunk_layer(
             self._transition,
             {"m": m, "mask": mask},
             chunk_size=chunk_size,
             no_batch_dims=len(m.shape[:-2]),
         )

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA activation
            mask:
                [*, N_seq, N_res, C_m] MSA mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)

        return m


class PairStack(nn.Module):
    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
        attention_config=None,
    ):
        super(PairStack, self).__init__()

        if fuse_projection_weights:
            self.tri_mul_out = FusedTriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = FusedTriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )
        else:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )

        self.tri_att_start = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
            attention_config=attention_config,
        )
        self.tri_att_end = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
            attention_config=attention_config,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

    def forward(self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        pair_trans_mask = pair_mask if _mask_trans else None

        if (_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        tmu_update = self.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if (not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if (not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_start(
                        z,
                        mask=pair_mask,
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = z.transpose(-2, -3)
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_end(
                        z,
                        mask=pair_mask.transpose(-1, -2),
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = z.transpose(-2, -3)
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
                self.pair_transition(
                    z, mask=pair_trans_mask, chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
        )

        return z


class MSABlock(nn.Module, ABC):
    @abstractmethod
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
        attn_map_dir: str='',
        attention_config=None,
    ):
        super(MSABlock, self).__init__()

        self.opm_first = opm_first
        self.attn_map_dir = attn_map_dir

        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
            attention_config=attention_config,
        )

        self.msa_dropout_layer = DropoutRowwise(msa_dropout)

        self.msa_transition = MSATransition(
            c_m=c_m,
            n=transition_n,
        )

        self.outer_product_mean = OuterProductMean(
            c_m,
            c_z,
            c_hidden_opm,
        )

        self.pair_stack = PairStack(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf,
            eps=eps,
            attention_config=attention_config,
        )

    def _compute_opm(self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
        _offload_inference: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        m, z = input_tensors

        if (_offload_inference and inplace_safe):
            # m: GPU, z: CPU
            del m, z
            assert (sys.getrefcount(input_tensors[1]) == 2)
            input_tensors[1] = input_tensors[1].cpu()
            m, z = input_tensors

        opm = self.outer_product_mean(
            m, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe
        )

        if (_offload_inference and inplace_safe):
            # m: GPU, z: GPU
            del m, z
            assert (sys.getrefcount(input_tensors[0]) == 2)
            input_tensors[1] = input_tensors[1].to(opm.device)
            m, z = input_tensors

        z = add(z, opm, inplace=inplace_safe)
        del opm

        return m, z

    @abstractmethod
    def forward(self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

def parse_attention_metadata_key(key):
    """
    Parse attention metadata keys like:
    'msa_attention_block_47_attn_0_recycle_0' 
    or 
    'tri_attention_block_47_attn_0_recycle_0'
    
    Returns:
        attn_type: "msa_row_attn" or "triangle_start_attn"
        layer_idx: int
    """
    parts = key.split('_')

    # Early checks
    if not (key.startswith("msa_attention") or key.startswith("tri_attention")):
        return None, -1

    # Find block index and layer index
    try:
        block_idx = parts.index("block")
        layer_idx = int(parts[block_idx + 1])
    except (ValueError, IndexError):
        layer_idx = -1

    # Now check attn number
    try:
        attn_idx = parts.index("attn")
        attn_num = int(parts[attn_idx + 1])
    except (ValueError, IndexError):
        attn_num = -1

    if attn_num != 0:
        return None, -1  # Only process attn_0 types

    # Determine attention type
    if key.startswith("msa_attention"):
        attn_type = "msa_row_attn"
    elif key.startswith("tri_attention"):
        attn_type = "triangle_start_attn"
    else:
        attn_type = None

    return attn_type, layer_idx


def save_attention_topk(attention_dict, save_dir, layer_name, layer_idx, attn_type, triangle_residue_idx, top_k=500):
    """
    Save the top K attention values for MSA Row and Triangle Start attention maps.
    
    Args:
        attention_dict (dict): Dictionary holding recent attention maps.
        save_dir (str): Directory to save output text files.
        layer_name (str): Full name of the layer.
        layer_idx (int): Layer number.
        attn_type (str): Either "msa_row_attn" or "triangle_start_attn".
        top_k (int): Number of top values to save.
    """

    if layer_name not in attention_dict:
        print(f"[Warning] Layer {layer_name} not found in recent attention.")
        return

    attn_array = attention_dict[layer_name]

    if isinstance(attn_array, list):
        attn_array = attn_array[0] if len(attn_array) == 1 else np.concatenate(attn_array, axis=0)

    if attn_type == "msa_row_attn":
        # Shape: (M, N, S, S)
        attn_array = attn_array[0]  # Now: (N, S, S)

    elif attn_type == "triangle_start_attn":
        # Shape: (Q, R, S, S)
        if triangle_residue_idx is None:
            print(f"[Warning] Triangle residue index not provided. Using average of all residues.")
            attn_array = np.mean(attn_array, axis=0)  # Now: (R, S, S)
            triangle_residue_idx = 'avg'
        else:
            attn_array = attn_array[triangle_residue_idx]
            triangle_residue_idx = str(triangle_residue_idx)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")

    os.makedirs(save_dir, exist_ok=True)

    if attn_type == "msa_row_attn":
        output_path = os.path.join(save_dir, f"{attn_type}_layer{layer_idx}.txt")
    elif attn_type == "triangle_start_attn":
        output_path = os.path.join(save_dir, f"{attn_type}_layer{layer_idx}_residue_idx_{triangle_residue_idx}.txt")

    with open(output_path, "w") as f:
        num_heads = attn_array.shape[0]
        S = attn_array.shape[1]

        for head_idx in range(num_heads):
            attn_map = attn_array[head_idx]  # Shape: (S, S)

            total_entries = S * S
            k = min(top_k, total_entries)

            flat_indices = np.argsort(attn_map.flatten())[::-1][:k]
            row_indices, col_indices = np.unravel_index(flat_indices, (S, S))
            scores = attn_map[row_indices, col_indices]

            # Write header
            f.write(f"Layer {layer_idx}, Head {head_idx}\n")

            for r, c, s in zip(row_indices, col_indices, scores):
                f.write(f"{r} {c} {s:.6f}\n")

    print(f"[Done] Saved top {top_k} entries for {attn_type} to {output_path}")


def save_all_topk_from_recent_attention(save_dir, triangle_residue_idx, top_k=500):
    """
    Scan ATTENTION_METADATA.recent_attention and save top-K attention maps for MSA and Triangle.
    
    Args:
        save_dir (str): Where to store the txt files.
        triangle_residue_idx (int): Residue index for triangle start/end attention.
        top_k (int): How many top values to save per head.
    """
    os.makedirs(save_dir, exist_ok=True)

    for k, v in ATTENTION_METADATA.recent_attention.items():
        attn_type, layer_idx = parse_attention_metadata_key(k)

        if attn_type is not None and layer_idx >= 0:
            # Handle chunked attention
            if isinstance(v, list) and isinstance(v[0], np.ndarray):
                try:
                    v_concat = v[0] if len(v) == 1 else np.concatenate(v, axis=0)
                except Exception as e:
                    print(f"[WARNING] Could not concatenate attention for {k}: {e}")
                    continue
            else:
                v_concat = v

            # Temporary dictionary to pass into `save_attention_topk`
            temp_dict = {k: v_concat}

            save_attention_topk(
                attention_dict=temp_dict,
                save_dir=save_dir,
                layer_name=k,
                layer_idx=layer_idx,
                attn_type=attn_type,
                triangle_residue_idx=triangle_residue_idx,
                top_k=top_k
            )


class EvoformerBlock(MSABlock):
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        no_column_attention: bool,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
        attn_map_dir: str='',
        attention_config=None,
    ):
        super(EvoformerBlock, self).__init__(c_m=c_m,
                                             c_z=c_z,
                                             c_hidden_msa_att=c_hidden_msa_att,
                                             c_hidden_opm=c_hidden_opm,
                                             c_hidden_mul=c_hidden_mul,
                                             c_hidden_pair_att=c_hidden_pair_att,
                                             no_heads_msa=no_heads_msa,
                                             no_heads_pair=no_heads_pair,
                                             transition_n=transition_n,
                                             msa_dropout=msa_dropout,
                                             pair_dropout=pair_dropout,
                                             opm_first=opm_first,
                                             fuse_projection_weights=fuse_projection_weights,
                                             inf=inf,
                                             eps=eps,
                                             attn_map_dir=attn_map_dir,
                                             attention_config=attention_config,
                                             )

        # Specifically, seqemb mode does not use column attention
        self.no_column_attention = no_column_attention
        self.attention_config = attention_config

        if not self.no_column_attention:
            self.msa_att_col = MSAColumnAttention(
                c_m,
                c_hidden_msa_att,
                no_heads_msa,
                inf=inf,
                attention_config=attention_config,
            )

    def forward(self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        msa_trans_mask = msa_mask if _mask_trans else None

        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        if(_offload_inference and inplace_safe):
            input_tensors = _offloadable_inputs
            del _offloadable_inputs
        else:
            input_tensors = [m, z]

        m, z = input_tensors

        if self.opm_first:
            del m, z

            m, z = self._compute_opm(input_tensors=input_tensors,
                                     msa_mask=msa_mask,
                                     chunk_size=chunk_size,
                                     inplace_safe=inplace_safe,
                                     _offload_inference=_offload_inference)

        m = add(m,
                self.msa_dropout_layer(
                    self.msa_att_row(
                        m,
                        z=z,
                        mask=msa_mask,
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                    )
                ),
                inplace=inplace_safe,
                )

        if (_offload_inference and inplace_safe):
            # m: GPU, z: CPU
            del m, z
            assert (sys.getrefcount(input_tensors[1]) == 2)
            input_tensors[1] = input_tensors[1].cpu()
            torch.cuda.empty_cache()
            m, z = input_tensors

        # Specifically, column attention is not used in seqemb mode.
        if not self.no_column_attention:
            m = add(m,
                    self.msa_att_col(
                        m,
                        mask=msa_mask,
                        chunk_size=chunk_size,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                        use_flash=use_flash,
                    ),
                    inplace=inplace_safe,
                    )

        m = add(
            m,
            self.msa_transition(
                m, mask=msa_trans_mask, chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        )

        if not self.opm_first:
            if (not inplace_safe):
                input_tensors = [m, z]

            del m, z

            m, z = self._compute_opm(input_tensors=input_tensors,
                                     msa_mask=msa_mask,
                                     chunk_size=chunk_size,
                                     inplace_safe=inplace_safe,
                                     _offload_inference=_offload_inference)

        if (_offload_inference and inplace_safe):
            # m: CPU, z: GPU
            del m, z
            assert (sys.getrefcount(input_tensors[0]) == 2)
            device = input_tensors[0].device
            input_tensors[0] = input_tensors[0].cpu()
            input_tensors[1] = input_tensors[1].to(device)
            m, z = input_tensors

        if (not inplace_safe):
            input_tensors = [m, z]

        del m, z

        z = self.pair_stack(
            z=input_tensors[1],
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size
        )

        if (_offload_inference and inplace_safe):
            # m: GPU, z: GPU
            device = z.device
            assert (sys.getrefcount(input_tensors[0]) == 2)
            input_tensors[0] = input_tensors[0].to(device)
            m, _ = input_tensors
        else:
            m = input_tensors[0]

        loop_idx = getattr(self, "loop_idx", 0)
        filename = f"{self._layer_name}" + '_recycle_%d' % loop_idx
        print('\t' + filename)
        with torch.no_grad():

            # compute top-k and save to text file for demo
            if self.attention_config.get("demo_attn", False) and hasattr(ATTENTION_METADATA, "recent_attention"):
                triangle_residue_idx = self.attention_config.get("triangle_residue_idx", None)
                top_k = self.attention_config.get("top_k", 500)
                save_all_topk_from_recent_attention(self.attn_map_dir, triangle_residue_idx, top_k=top_k)
                # Clear after use to free memory
                ATTENTION_METADATA.recent_attention.clear()
        return m, z


class ExtraMSABlock(MSABlock):
    """ 
        Almost identical to the standard EvoformerBlock, except in that the
        ExtraMSABlock uses GlobalAttention for MSA column attention and
        requires more fine-grained control over checkpointing. Separated from
        its twin to preserve the TorchScript-ability of the latter.
    """
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
        ckpt: bool,
        attention_config=None,
    ):
        super(ExtraMSABlock, self).__init__(c_m=c_m,
                                            c_z=c_z,
                                            c_hidden_msa_att=c_hidden_msa_att,
                                            c_hidden_opm=c_hidden_opm,
                                            c_hidden_mul=c_hidden_mul,
                                            c_hidden_pair_att=c_hidden_pair_att,
                                            no_heads_msa=no_heads_msa,
                                            no_heads_pair=no_heads_pair,
                                            transition_n=transition_n,
                                            msa_dropout=msa_dropout,
                                            pair_dropout=pair_dropout,
                                            opm_first=opm_first,
                                            fuse_projection_weights=fuse_projection_weights,
                                            inf=inf,
                                            eps=eps,
                                            attention_config=attention_config,
                                            )

        self.ckpt = ckpt

        self.msa_att_col = MSAColumnGlobalAttention(
            c_in=c_m,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
            eps=eps,
        )

    def forward(self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size
       
        if(_offload_inference and inplace_safe):
            input_tensors = _offloadable_inputs
            del _offloadable_inputs
        else:
            input_tensors = [m, z]

        m, z = input_tensors

        if self.opm_first:
            del m, z

            m, z = self._compute_opm(input_tensors=input_tensors,
                                     msa_mask=msa_mask,
                                     chunk_size=chunk_size,
                                     inplace_safe=inplace_safe,
                                     _offload_inference=_offload_inference)

        m = add(m,
            self.msa_dropout_layer(
                self.msa_att_row(
                    m.clone() if torch.is_grad_enabled() else m, 
                    z=z.clone() if torch.is_grad_enabled() else z, 
                    mask=msa_mask, 
                    chunk_size=_attn_chunk_size,
                    use_lma=use_lma,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_memory_efficient_kernel=not (use_lma or use_deepspeed_evo_attention),
                    _checkpoint_chunks=
                        self.ckpt if torch.is_grad_enabled() else False,
                )
            ),
            inplace=inplace_safe,
        )

        if (not inplace_safe):
            input_tensors = [m, z]

        del m, z

        def fn(input_tensors):
            m, z = input_tensors

            if (_offload_inference and inplace_safe):
                # m: GPU, z: CPU
                del m, z
                assert (sys.getrefcount(input_tensors[1]) == 2)
                input_tensors[1] = input_tensors[1].cpu()
                torch.cuda.empty_cache()
                m, z = input_tensors

            m = add(m,
                    self.msa_att_col(
                        m,
                        mask=msa_mask,
                        chunk_size=chunk_size,
                        use_lma=use_lma,
                    ),
                    inplace=inplace_safe,
                    )

            m = add(
                m,
                self.msa_transition(
                    m, mask=msa_mask, chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
            )

            if not self.opm_first:
                if (not inplace_safe):
                    input_tensors = [m, z]

                del m, z

                m, z = self._compute_opm(input_tensors=input_tensors,
                                         msa_mask=msa_mask,
                                         chunk_size=chunk_size,
                                         inplace_safe=inplace_safe,
                                         _offload_inference=_offload_inference)

            if (_offload_inference and inplace_safe):
                # m: CPU, z: GPU
                del m, z
                assert (sys.getrefcount(input_tensors[0]) == 2)
                device = input_tensors[0].device
                input_tensors[0] = input_tensors[0].cpu()
                input_tensors[1] = input_tensors[1].to(device)
                m, z = input_tensors

            if (not inplace_safe):
                input_tensors = [m, z]

            del m, z

            z = self.pair_stack(
                input_tensors[1],
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
                _attn_chunk_size=_attn_chunk_size
            )

            m = input_tensors[0]
            if (_offload_inference and inplace_safe):
                # m: GPU, z: GPU
                device = z.device
                del m
                assert (sys.getrefcount(input_tensors[0]) == 2)
                input_tensors[0] = input_tensors[0].to(device)
                m, _ = input_tensors

            return m, z

        if (torch.is_grad_enabled() and self.ckpt):
            checkpoint_fn = get_checkpoint_fn()
            m, z = checkpoint_fn(fn, input_tensors)
        else:
            m, z = fn(input_tensors)

        return m, z


class EvoformerStack(nn.Module):
    """
    Main Evoformer trunk.

    Implements Algorithm 6.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        c_s: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        no_column_attention: bool,
        opm_first: bool,
        fuse_projection_weights: bool,
        blocks_per_ckpt: int,
        inf: float,
        eps: float,
        clear_cache_between_blocks: bool = False, 
        tune_chunk_size: bool = False,
        attn_map_dir: str='',
        attention_config=None,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair channel dimension
            c_hidden_msa_att:
                Hidden dimension in MSA attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            c_s:
                Channel dimension of the output "single" embedding
            no_heads_msa:
                Number of heads used for MSA attention
            no_heads_pair:
                Number of heads used for pair attention
            no_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
            no_column_attention:
                When True, doesn't use column attention. Required for running
                sequence embedding mode
            opm_first:
                When True, Outer Product Mean is performed at the beginning of
                the Evoformer block instead of after the MSA Stack.
                Used in Multimer pipeline.
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super(EvoformerStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks
        self.attn_map_dir = attn_map_dir

        self.blocks = nn.ModuleList()

        for _ in range(no_blocks):
            block = EvoformerBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                no_column_attention=no_column_attention,
                opm_first=opm_first,
                fuse_projection_weights=fuse_projection_weights,
                inf=inf,
                eps=eps,
                attn_map_dir=attn_map_dir,
                attention_config=attention_config,
            )
            self.blocks.append(block)

        self.linear = Linear(c_m, c_s)

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if(tune_chunk_size):
            self.chunk_size_tuner = ChunkSizeTuner()

        # naming for easily grabbing features and attention maps
        self.assign_layer_names()

        self.forward_counter = 0  # Track global forward passes

        # Register hook for all EvoformerBlocks and attention layers
        for block in self.blocks:
            block.register_forward_pre_hook(self.inject_counter)  # Hook for EvoformerBlock
            for submodule in block.modules():
                if isinstance(submodule, (MSAAttention, TriangleAttention)):
                    submodule.register_forward_pre_hook(self.inject_counter)

    def inject_counter(self, module, inputs):
        """Dynamically injects forward_counter into EvoformerBlocks and attention layers."""
        setattr(module, "loop_idx", self.forward_counter)

    def assign_layer_names(self):
        """Assigns unique names to each MSAAttention, TriangleAttention, EvoformerBlock layer"""
        msa_block = 0
        msa_counter = 0
        tri_block = 0
        tri_counter = 0
        block_counter = 0
        for name, module in self.named_modules():
            if isinstance(module, (MSAAttention)):
                block = name.split('blocks.')[1].split('.')[0]
                if block != msa_block:
                    msa_block = block
                    msa_counter = 0
                module._attention_name = f"msa_attention_block_{msa_block}_attn_{msa_counter}"  # Assign a unique name
                if hasattr(module, "mha"):
                    module.mha._attention_name = module._attention_name
                # print(f"Assigned: {module._attention_name} to {name}")
                msa_counter += 1
            elif isinstance(module, (TriangleAttention)):
                block = name.split('blocks.')[1].split('.')[0]
                if block != tri_block:
                    tri_block = block
                    tri_counter = 0
                module._attention_name = f"tri_attention_block_{tri_block}_attn_{tri_counter}"  # Assign a unique name
                if hasattr(module, "mha"):
                    module.mha._attention_name = module._attention_name
                # print(f"Assigned: {module._attention_name} to {name}")
                tri_counter += 1
            elif isinstance(module, (EvoformerBlock)):
                module._layer_name = f"evo_block_{block_counter}"  # Assign a unique name
                # print(f"Assigned: {module._layer_name} to {name}")
                block_counter += 1

    def _prep_blocks(self, 
        m: torch.Tensor, 
        z: torch.Tensor, 
        chunk_size: int,
        use_deepspeed_evo_attention: bool,
        use_lma: bool,
        use_flash: bool,
        msa_mask: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        inplace_safe: bool,
        _mask_trans: bool,
    ):
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if(self.clear_cache_between_blocks):
            def block_with_cache_clear(block, *args, **kwargs):
                torch.cuda.empty_cache()
                return block(*args, **kwargs)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        if(chunk_size is not None and self.chunk_size_tuner is not None):
            assert(not self.training)
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                # We don't want to write in-place during chunk tuning runs
                args=(m.clone(), z.clone(),),
                min_chunk_size=chunk_size,
            )
            blocks = [
                partial(b, 
                    chunk_size=tuned_chunk_size,
                    # A temporary measure to address torch's occasional
                    # inability to allocate large tensors
                    _attn_chunk_size=max(chunk_size, tuned_chunk_size // 4),
                ) for b in blocks
            ]

        return blocks

    def _forward_offload(self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert(not (self.training or torch.is_grad_enabled()))
        blocks = self._prep_blocks(
            # We are very careful not to create references to these tensors in
            # this function
            m=input_tensors[0],
            z=input_tensors[1],
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_flash=use_flash,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            inplace_safe=True,
            _mask_trans=_mask_trans,
        )

        for b in blocks:
            m, z = b(
                None, 
                None, 
                _offload_inference=True,
                _offloadable_inputs=input_tensors,
            )
            input_tensors[0] = m
            input_tensors[1] = z
            del m, z
        
        m, z = input_tensors
        
        s = self.linear(m[..., 0, :, :])
        
        return m, z, s

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
            use_flash: 
                Whether to use FlashAttention where possible. Mutually 
                exclusive with use_lma and use_deepspeed_evo_attention.
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """ 
        blocks = self._prep_blocks(
            m=m,
            z=z,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_flash=use_flash,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if(not torch.is_grad_enabled()):
            blocks_per_ckpt = None
        
        m, z = checkpoint_blocks(
            blocks,
            args=(m, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )

        s = self.linear(m[..., 0, :, :])
        self.forward_counter += 1  # Increment automatically on each forward pass
        
        return m, z, s


class ExtraMSAStack(nn.Module):
    """
    Implements Algorithm 18.
    """
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
        ckpt: bool,
        clear_cache_between_blocks: bool = False,
        tune_chunk_size: bool = False,
        attention_config=None,
        **kwargs,
    ):
        super(ExtraMSAStack, self).__init__()
 
        self.ckpt = ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks
        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = ExtraMSABlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                opm_first=opm_first,
                fuse_projection_weights=fuse_projection_weights,
                inf=inf,
                eps=eps,
                ckpt=False,
                attention_config=attention_config,
            )
            self.blocks.append(block)
            
        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if(tune_chunk_size):
            self.chunk_size_tuner = ChunkSizeTuner()

    def _prep_blocks(self, 
        m: torch.Tensor, 
        z: torch.Tensor, 
        chunk_size: int,
        use_deepspeed_evo_attention: bool,
        use_lma: bool,
        msa_mask: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        inplace_safe: bool,
        _mask_trans: bool,
    ):
        blocks = [
            partial(
                b, 
                msa_mask=msa_mask, 
                pair_mask=pair_mask, 
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            ) for b in self.blocks
        ]

        def clear_cache(b, *args, **kwargs):
            torch.cuda.empty_cache()
            return b(*args, **kwargs)

        if(self.clear_cache_between_blocks):
            blocks = [partial(clear_cache, b) for b in blocks]

        if(chunk_size is not None and self.chunk_size_tuner is not None):
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                # Tensors cloned to avoid getting written to in-place
                # A corollary is that chunk size tuning should be disabled for
                # large N, when z gets really big
                args=(m.clone(), z.clone(),),
                min_chunk_size=chunk_size,
            )
            blocks = [
                partial(b, 
                    chunk_size=tuned_chunk_size,
                    # A temporary measure to address torch's occasional
                    # inability to allocate large tensors
                    _attn_chunk_size=max(chunk_size, tuned_chunk_size // 4),
                ) for b in blocks
            ]

        return blocks

    def _forward_offload(self,
        input_tensors: Sequence[torch.Tensor],
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        assert(not (self.training or torch.is_grad_enabled()))
        blocks = self._prep_blocks(
            # We are very careful not to create references to these tensors in
            # this function
            m=input_tensors[0],
            z=input_tensors[1],
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            inplace_safe=True,
            _mask_trans=_mask_trans,
        )

        for b in blocks:
            m, z = b(
                None, 
                None, 
                _offload_inference=True,
                _offloadable_inputs=input_tensors,
            )
            input_tensors[0] = m
            input_tensors[1] = z
            del m, z

        return input_tensors[1]

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_extra, N_res, C_m] extra MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            chunk_size: Inference-time subbatch size for Evoformer modules
            use_deepspeed_evo_attention: Whether to use DeepSpeed memory-efficient kernel
            use_lma: Whether to use low-memory attention during inference
            msa_mask:
                Optional [*, N_extra, N_res] MSA mask
            pair_mask:
                Optional [*, N_res, N_res] pair mask
        Returns:
            [*, N_res, N_res, C_z] pair update
        """
        checkpoint_fn = get_checkpoint_fn()
        blocks = self._prep_blocks(
            m=m,
            z=z,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        for b in blocks:
            if(self.ckpt and torch.is_grad_enabled()):
                m, z = checkpoint_fn(b, m, z)
            else:
                m, z = b(m, z)

        return z
