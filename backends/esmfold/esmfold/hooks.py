"""Trace extraction from HuggingFace ESMFold, which supports neither output_attentions nor
output_hidden_states -- so everything here is forward hooks on the ESM-2 trunk, the folding trunk,
and the structure module. The tokenizer adds <cls>/<eos>, so attention arrives as (N+2, N+2) and is
sliced back to (N, N) to match the FASTA sequence.
"""
import inspect
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ESMFoldTraceCollector:
    """Attention, hidden states, and folding trunk intermediates from EsmForProteinFolding."""

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
        self.attention: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles: List[Any] = []
        self._patched_forwards: List[Tuple[nn.Module, Callable]] = []
        self.recycled_s_s: List[torch.Tensor] = []
        self.recycled_s_z: List[torch.Tensor] = []
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
        """Hook `encoder.layer[i].attention.self` for [B, H, N, N] and `encoder.layer[i]` for [B, N, D]."""
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
        """Hooked on self_attn, not the outer EsmAttention, which discards attn_weights."""
        orig_forward = self_attn.forward
        params = list(inspect.signature(orig_forward).parameters)
        # By name, so a signature change does not silently shift it.
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
            # [B, H, N+2, N+2] with <cls>/<eos>, sliced to [B, H, N, N].
            if attn_weights.dim() == 4 and attn_weights.shape[-1] >= 3:
                attn_weights = attn_weights[:, :, 1:-1, 1:-1]

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

            if self.head_indices is not None:
                head_dim = 1 if attn_weights.dim() == 4 else 0
                idx = torch.tensor(self.head_indices, device=attn_weights.device)
                attn_weights = attn_weights.index_select(head_dim, idx)

            key = f"layer_{layer_idx:03d}"
            self.attention[key] = attn_weights.detach().cpu()
        return hook

    def _make_activation_hook(self, layer_idx: int) -> Callable:
        """The layer output, <cls>/<eos> stripped so [B, N, D] matches the attention's [B, H, N, N]."""
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            if h is not None and isinstance(h, torch.Tensor) and h.dim() >= 2:
                # Strip <cls>/<eos> to align with the already-sliced attention maps.
                if h.dim() == 3 and h.shape[1] >= 3:
                    h = h[:, 1:-1, :]
                key = f"layer_{layer_idx:03d}"
                self.activations[key] = h.detach().cpu()
        return hook

    def _make_trunk_hook(self) -> Callable:
        """Hook that captures s_s and s_z at every recycling iteration."""
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            s_s, s_z = None, None

            # tuple
            if isinstance(out, tuple) and len(out) >= 2:
                s_s, s_z = out[0], out[1]
            # dataclass or object
            elif hasattr(out, 's_s') and hasattr(out, 's_z'):
                s_s, s_z = out.s_s, out.s_z
            # dict
            elif isinstance(out, dict):
                s_s, s_z = out.get('s_s'), out.get('s_z')

            if s_s is not None and s_z is not None:
                # Batch dim squeezed and moved to CPU: s_z is [N, N, 128].
                self.recycled_s_s.append(s_s.squeeze(0).cpu().detach())
                self.recycled_s_z.append(s_z.squeeze(0).cpu().detach())
        return hook


    def register_trunk_hooks(self, model: nn.Module) -> None:
        """Per-block sequence_state [B, L, C_s] and pairwise_state [B, L, L, C_z] -- the latter is large."""
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
        """Blocks fire once per recycle onto the same keys, so only the last recycle is retained here;
        every iteration is in recycled_s_s/recycled_s_z.
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
    """IPA attention and per-recycle backbone outputs from the structure module. IPA is one shared
    module reused across blocks, so its patched softmax fires num_blocks times per recycle.
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
        """Hook `model.trunk.structure_module` and its IPA submodule; `model` is the full model."""
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
        """IPA returns only the single-rep update, so its softmax output is stashed on the way past.
        Wrapped in an nn.Module: PyTorch __setattr__ rejects assigning a plain function.
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
        if isinstance(out, dict):
            for key in ("positions", "frames", "single"):
                if key in out:
                    result[key] = out[key]
        else:
            for key in ("positions", "frames", "single"):
                val = getattr(out, key, None)
                if val is not None:
                    result[key] = val
        return result

    def _sm_output_hook(self, module: nn.Module, inp: Any, out: Any) -> None:
        """Once per recycle, after all blocks: backbone positions and states, dict or dataclass."""
        key = f"recycle_{self._recycle_idx:02d}"
        fields = self._extract_from_output(out)

        if "positions" in fields:
            self.backbone_positions[key] = fields["positions"].detach().cpu()
        if "frames" in fields:
            self.backbone_frames[key] = fields["frames"].detach().cpu()
        if "single" in fields:
            self.sm_states[key] = fields["single"].detach().cpu()

        # Always advance, captured or not.
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
