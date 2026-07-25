"""
Hook-based extraction of attention weights and hidden states from HuggingFace ESMFold.

HF EsmForProteinFolding does NOT support output_attentions / output_hidden_states.
We capture traces by registering forward hooks on three stages:

  1. ESM-2 trunk (model.esm):
     Attention weights: hook on each EsmSelfAttention module, monkey-patching
                        the forward to force attn_weights to be returned.
     Activations:       hook on each EsmLayer (full transformer block output).

  2. Folding trunk (model.trunk):
     Per-block: hook on each EsmFoldTriangularSelfAttentionBlock to capture
                sequence_state [B, L, C_s] and pairwise_state [B, L, L, C_z].
     Final:     capture out.s_s and out.s_z from the model output.

  3. Structure module (model.trunk.structure_module):
     IPA attention: StructureModuleTraceCollector wraps ipa.softmax to capture
                    the attention matrix [H, N, N] at each block per recycle.
     Backbone:      hook on structure_module to capture per-recycle positions
                    and single representations.

The ESM-2 tokenizer adds <cls> and <eos> tokens, so attention maps are
(seq_len+2, seq_len+2). We slice out the special tokens so stored tensors
are (seq_len, seq_len), matching the FASTA sequence.

Structure module tracing:
  StructureModuleTraceCollector hooks into model.trunk.structure_module.ipa
  to capture IPA attention weights [H, N, N] at each block within each
  recycling iteration, and hooks on trunk.structure_module to capture
  per-recycle backbone positions and single representations.
"""
import inspect
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ESMFoldTraceCollector:
    """
    Collects attention weights, hidden states, and folding trunk intermediates
    from HuggingFace EsmForProteinFolding.
    """

    def __init__(
        self,
        want_attention: bool = True,
        want_activations: bool = True,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
        expected_seq_len: Optional[int] = None,
    ):
        self.want_attention = want_attention
        self.want_activations = want_activations
        self.layer_indices = layer_indices  # None => all
        self.head_indices = head_indices
        self.expected_seq_len = expected_seq_len
        # ESM-2 encoder outputs
        self.attention: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles: List[Any] = []
        self._patched_forwards: List[Tuple[nn.Module, Callable]] = []
        self.recycled_s_s: List[torch.Tensor] = []
        self.recycled_s_z: List[torch.Tensor] = []
        # Per-block evoformer intermediates from trunk.blocks[i]
        self.trunk_blocks: Dict[str, torch.Tensor] = {}
        self._slice_validated = False

    def clear(self) -> None:
        self.attention.clear()
        self.activations.clear()
        self.recycled_s_s.clear()
        self.recycled_s_z.clear()
        self.trunk_blocks.clear()
        self._slice_validated = False

    def _should_store_layer(self, layer_idx: int) -> bool:
        return self.layer_indices is None or layer_idx in self.layer_indices

    def register_hooks(self, esm_model: nn.Module) -> None:
        """
        Register hooks on the ESM-2 trunk (model.esm passed in).

        Targets:
          - encoder.layer[i].attention.self  -> attention weights [B, H, N, N]
          - encoder.layer[i]                 -> activations [B, N, D]

        For HF ESM, EsmSelfAttention.forward() only returns attn_weights when
        the framework's output capturing mechanism requests it. We monkey-patch
        the forward to always emit (output, attn_weights).
        """
        encoder_layers = self._find_encoder_layers(esm_model)
        if not encoder_layers:
            warnings.warn(
                "Could not find encoder.layer ModuleList in ESM trunk. "
                "Trace extraction may not work.",
                UserWarning,
            )
            return

        for layer_idx, layer_module in enumerate(encoder_layers):
            if not self._should_store_layer(layer_idx):
                continue

            if self.want_attention:
                self_attn = self._find_self_attention(layer_module)
                if self_attn is not None:
                    self._patch_and_hook_attention(self_attn, layer_idx)

            if self.want_activations:
                h = layer_module.register_forward_hook(self._make_activation_hook(layer_idx))
                self._handles.append(h)

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for module, orig_forward in self._patched_forwards:
            module.forward = orig_forward
        self._patched_forwards.clear()

    def _find_encoder_layers(self, esm_model: nn.Module) -> Optional[nn.ModuleList]:
        """Find the nn.ModuleList of transformer layers in the ESM encoder."""
        if hasattr(esm_model, "encoder"):
            enc = esm_model.encoder
            if hasattr(enc, "layer") and isinstance(enc.layer, nn.ModuleList):
                return enc.layer
        for name, module in esm_model.named_modules():
            if isinstance(module, nn.ModuleList) and re.match(r".*encoder.*layer$", name):
                return module
        return None

    def _find_self_attention(self, layer_module: nn.Module) -> Optional[nn.Module]:
        """Find the self-attention submodule inside a transformer layer."""
        if hasattr(layer_module, "attention"):
            attn = layer_module.attention
            if hasattr(attn, "self"):
                return attn.self
            return attn
        return None

    def _patch_and_hook_attention(self, self_attn: nn.Module, layer_idx: int) -> None:
        """
        Monkey-patch EsmSelfAttention.forward to always return attn_weights,
        then register a hook to capture them.

        HF EsmSelfAttention returns (attn_output, attn_weights) from its
        internal attention function. But the outer EsmAttention layer discards
        attn_weights with `attn_output, _ = self.self(...)`. We hook self_attn
        directly to get the full tuple.
        """
        orig_forward = self_attn.forward
        params = list(inspect.signature(orig_forward).parameters)
        # Find output_attentions position by name — robust to signature changes
        oa_pos = params.index("output_attentions") if "output_attentions" in params else -1

        def patched_forward(*args, **kwargs):
            if oa_pos >= 0 and oa_pos < len(args):
                args = args[:oa_pos] + (True,) + args[oa_pos + 1:]
            else:
                kwargs["output_attentions"] = True
            return orig_forward(*args, **kwargs)

        self_attn.forward = patched_forward
        self._patched_forwards.append((self_attn, orig_forward))

        h = self_attn.register_forward_hook(self._make_attention_hook(layer_idx))
        self._handles.append(h)

    def _make_attention_hook(self, layer_idx: int) -> Callable:
        """Hook that captures attn_weights from (attn_output, attn_weights) tuple."""
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            if not isinstance(out, tuple) or len(out) < 2:
                return
            attn_weights = out[1]
            if attn_weights is None:
                return
            # attn_weights shape: [B, H, N+2, N+2] (includes <cls> and <eos>)
            # Slice out special tokens -> [B, H, N, N]
            if attn_weights.dim() == 4 and attn_weights.shape[-1] >= 3:
                attn_weights = attn_weights[:, :, 1:-1, 1:-1]

            # Validate that the sliced shape matches expected sequence length
            if (self.expected_seq_len is not None
                    and not self._slice_validated
                    and attn_weights.dim() == 4):
                actual_n = attn_weights.shape[-1]
                if actual_n != self.expected_seq_len:
                    warnings.warn(
                        f"Attention slice mismatch: after removing special tokens, "
                        f"attention dimension is {actual_n} but expected sequence "
                        f"length is {self.expected_seq_len}. Attention maps may be "
                        f"misaligned with residue indices.",
                        UserWarning,
                    )
                self._slice_validated = True

            # Filter heads if requested, to save memory
            if self.head_indices is not None:
                head_dim = 1 if attn_weights.dim() == 4 else 0
                idx = torch.tensor(self.head_indices, device=attn_weights.device)
                attn_weights = attn_weights.index_select(head_dim, idx)

            key = f"layer_{layer_idx:03d}"
            self.attention[key] = attn_weights.detach().cpu()
        return hook

    def _make_activation_hook(self, layer_idx: int) -> Callable:
        """Hook that captures the transformer layer output (hidden state).

        Strips the leading <cls> and trailing <eos> special tokens so the
        activation shape [B, N, D] matches the attention shape [B, H, N, N].
        """
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            if h is not None and isinstance(h, torch.Tensor) and h.dim() >= 2:
                # Strip <cls> (index 0) and <eos> (index -1) to align with
                # attention maps which are already sliced to seq_len.
                if h.dim() == 3 and h.shape[1] >= 3:
                    h = h[:, 1:-1, :]
                key = f"layer_{layer_idx:03d}"
                self.activations[key] = h.detach().cpu()
        return hook

    def _make_trunk_hook(self) -> Callable:
        """Hook that captures s_s and s_z at every recycling iteration."""
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            s_s, s_z = None, None

            # Handle HF returning a tuple (usually s_s is index 0 and s_z is index 1)
            if isinstance(out, tuple) and len(out) >= 2:
                s_s, s_z = out[0], out[1]
            # Handle HF returning a dataclass or object
            elif hasattr(out, 's_s') and hasattr(out, 's_z'):
                s_s, s_z = out.s_s, out.s_z
            # Handle dictionaries
            elif isinstance(out, dict):
                s_s, s_z = out.get('s_s'), out.get('s_z')

            if s_s is not None and s_z is not None:
                # Squeeze out the batch dimension and move to CPU to prevent RAM crashes
                # s_s shape: [N, 1024] | s_z shape: [N, N, 128]
                self.recycled_s_s.append(s_s.squeeze(0).cpu().detach())
                self.recycled_s_z.append(s_z.squeeze(0).cpu().detach())
        return hook

    # -------------------------------------------------------------------
    # Per-block evoformer hooks
    # -------------------------------------------------------------------

    def register_trunk_hooks(self, model: nn.Module) -> None:
        """
        Register hooks on each EsmFoldTriangularSelfAttentionBlock inside
        model.trunk.blocks to capture per-block sequence_state [B, L, C_s]
        and pairwise_state [B, L, L, C_z].

        Note: these tensors can be large (pair state is L×L×C_z per block).
        """
        trunk = getattr(model, "trunk", None)
        if trunk is None:
            warnings.warn("model.trunk not found; trunk block hooks skipped.", UserWarning)
            return
        blocks = getattr(trunk, "blocks", None)
        if blocks is None or not isinstance(blocks, nn.ModuleList):
            warnings.warn("model.trunk.blocks not found; trunk block hooks skipped.", UserWarning)
            return

        for block_idx, block in enumerate(blocks):
            h = block.register_forward_hook(self._make_trunk_block_hook(block_idx))
            self._handles.append(h)

    def _make_trunk_block_hook(self, block_idx: int) -> Callable:
        """
        Hook for EsmFoldTriangularSelfAttentionBlock.
        Output is (sequence_state, pairwise_state).

        Note: trunk blocks are called once per recycle iteration. Since dict
        keys are the same across iterations, only the LAST recycle's data
        is retained. This matches the behavior of recycled_s_s/s_z where
        only the final values are used downstream. All recycle iterations
        are captured in recycled_s_s/recycled_s_z lists for analysis if needed.
        """
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            if not isinstance(out, tuple) or len(out) < 2:
                return
            seq_state, pair_state = out[0], out[1]
            if isinstance(seq_state, torch.Tensor):
                self.trunk_blocks[f"block_{block_idx:03d}_seq"] = seq_state.squeeze(0).detach().cpu()
            if isinstance(pair_state, torch.Tensor):
                self.trunk_blocks[f"block_{block_idx:03d}_pair"] = pair_state.squeeze(0).detach().cpu()
        return hook


class StructureModuleTraceCollector:
    """
    Captures IPA attention weights and per-recycling-iteration backbone
    outputs from the ESMFold structure module.

    Hook targets:
      - trunk.structure_module.ipa: monkey-patched to stash the softmax
        attention matrix a [*, H, N, N] before it's consumed internally.
        Fires num_blocks times per recycle (IPA is a single shared module
        reused across all structure module blocks).
      - trunk.structure_module: captures the full output per recycle,
        including stacked positions [num_blocks, B, N, 14, 3], frames,
        and the final single representation. Handles both dict and
        dataclass outputs from HuggingFace.
    """

    def __init__(self) -> None:
        self.ipa_attention: Dict[str, torch.Tensor] = {}
        self.backbone_positions: Dict[str, torch.Tensor] = {}
        self.backbone_frames: Dict[str, torch.Tensor] = {}
        self.sm_states: Dict[str, torch.Tensor] = {}
        self._handles: List[Any] = []
        self._patched_forwards: List[Tuple[nn.Module, Callable]] = []
        self._recycle_idx = 0
        self._block_idx = 0

    def clear(self) -> None:
        self.ipa_attention.clear()
        self.backbone_positions.clear()
        self.backbone_frames.clear()
        self.sm_states.clear()
        self._recycle_idx = 0
        self._block_idx = 0

    def register_hooks(self, model: nn.Module) -> None:
        """
        Register hooks on model.trunk.structure_module and its IPA submodule.

        Args:
            model: the full EsmForProteinFolding model (we navigate to trunk).
        """
        trunk = getattr(model, "trunk", None)
        if trunk is None:
            warnings.warn("model.trunk not found; structure module hooks skipped.", UserWarning)
            return
        sm = getattr(trunk, "structure_module", None)
        if sm is None:
            warnings.warn("trunk.structure_module not found; hooks skipped.", UserWarning)
            return

        ipa = getattr(sm, "ipa", None)
        if ipa is not None:
            self._patch_ipa(ipa)

        h = sm.register_forward_hook(self._sm_output_hook)
        self._handles.append(h)

    def _patch_ipa(self, ipa: nn.Module) -> None:
        """
        Monkey-patch IPA forward to stash the softmax attention matrix.

        IPA computes a = softmax(scalar_attn + point_attn + pair_bias + mask)
        but only returns the single-rep update. We intercept a after softmax
        and store it, keyed by recycle_idx and block_idx.

        ipa.softmax is an nn.Softmax module, so we can't replace it with a
        plain function (PyTorch __setattr__ rejects non-Module assignments).
        Instead we wrap it in an nn.Module subclass that captures the output.
        """
        orig_forward = ipa.forward
        orig_softmax_module = ipa.softmax
        collector = self

        class CapturingSoftmax(nn.Module):
            def __init__(self, wrapped: nn.Module):
                super().__init__()
                self.wrapped = wrapped
                self.last_a: Optional[torch.Tensor] = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                result = self.wrapped(x)
                self.last_a = result.detach()
                return result

        capturing = CapturingSoftmax(orig_softmax_module)
        ipa.softmax = capturing

        def patched_forward(*args, **kwargs):
            capturing.last_a = None
            out = orig_forward(*args, **kwargs)
            if capturing.last_a is not None:
                key = f"recycle_{collector._recycle_idx:02d}_block_{collector._block_idx:02d}"
                collector.ipa_attention[key] = capturing.last_a
            collector._block_idx += 1
            return out

        ipa.forward = patched_forward
        self._patched_forwards.append((ipa, orig_forward))
        self._orig_softmax = (ipa, orig_softmax_module)

    def _extract_from_output(self, out: Any) -> dict:
        """Extract fields from structure module output (dict or dataclass)."""
        result = {}
        # Try dict access first
        if isinstance(out, dict):
            for key in ("positions", "frames", "single"):
                if key in out:
                    result[key] = out[key]
        else:
            # Handle dataclass / namedtuple / object with attributes
            for key in ("positions", "frames", "single"):
                val = getattr(out, key, None)
                if val is not None:
                    result[key] = val
        return result

    def _sm_output_hook(self, module: nn.Module, inp: Any, out: Any) -> None:
        """
        Fires once per recycling iteration after the full structure module
        forward (all blocks). Captures backbone positions and states.
        Handles both dict and dataclass outputs from HuggingFace.
        """
        key = f"recycle_{self._recycle_idx:02d}"
        fields = self._extract_from_output(out)

        if "positions" in fields:
            self.backbone_positions[key] = fields["positions"].detach().cpu()
        if "frames" in fields:
            self.backbone_frames[key] = fields["frames"].detach().cpu()
        if "single" in fields:
            self.sm_states[key] = fields["single"].detach().cpu()

        # Always reset block counter and advance recycle counter,
        # regardless of whether we captured any data.
        self._recycle_idx += 1
        self._block_idx = 0

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for module, orig_forward in self._patched_forwards:
            module.forward = orig_forward
        self._patched_forwards.clear()
        if hasattr(self, "_orig_softmax"):
            ipa_mod, orig_sm = self._orig_softmax
            ipa_mod.softmax = orig_sm
            del self._orig_softmax
