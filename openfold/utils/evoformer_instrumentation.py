from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class InstrumentationConfig:
    layers: Optional[List[int]] = None
    capture_msa: bool = True
    capture_pair: bool = True
    clone: bool = True
    to_cpu: bool = True
    dtype: Optional[torch.dtype] = torch.float32


class EvoformerRecorder:
    def __init__(self, config: InstrumentationConfig):
        self.config = config
        self.records: Dict[str, Any] = {}
        self.enabled = True

    def clear(self):
        self.records.clear()

    def _process_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.clone:
            x = x.clone()
        x = x.detach()
        if self.config.to_cpu:
            x = x.cpu()
        if self.config.dtype is not None:
            x = x.to(self.config.dtype)
        return x

    def record(self, key: str, value: Any):
        if not self.enabled:
            return

        if isinstance(value, torch.Tensor):
            self.records[key] = self._process_tensor(value)
        elif isinstance(value, (tuple, list)):
            out = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    out.append(self._process_tensor(item))
                else:
                    out.append(item)
            self.records[key] = out
        elif isinstance(value, dict):
            out = {}
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    out[k] = self._process_tensor(v)
                else:
                    out[k] = v
            self.records[key] = out
        else:
            self.records[key] = value

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.records, path)

    def summary(self) -> Dict[str, Any]:
        s = {}
        for k, v in self.records.items():
            if isinstance(v, torch.Tensor):
                s[k] = {
                    "shape": tuple(v.shape),
                    "dtype": str(v.dtype),
                    "mean": float(v.float().mean()),
                    "std": float(v.float().std()),
                }
            elif isinstance(v, list):
                s[k] = {"type": "list", "len": len(v)}
            elif isinstance(v, dict):
                inner = {}
                for kk, vv in v.items():
                    if isinstance(vv, torch.Tensor):
                        inner[kk] = {"shape": tuple(vv.shape), "dtype": str(vv.dtype)}
                    else:
                        inner[kk] = str(type(vv))
                s[k] = inner
            else:
                s[k] = str(type(v))
        return s


def find_evoformer_blocks(model):
    candidates = [
        "evoformer.blocks",
        "evoformer.trunk.blocks",
        "model.evoformer.blocks",
    ]
    for name in candidates:
        obj = model
        ok = True
        for part in name.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj

    for name, module in model.named_modules():
        if name.endswith("evoformer.blocks"):
            return module

    raise RuntimeError("Could not find Evoformer blocks in model")


def _selected(layer_idx: int, layers: Optional[List[int]]) -> bool:
    return layers is None or layer_idx in layers


def attach_evoformer_block_output_hooks(model, recorder: EvoformerRecorder):
    handles = []
    blocks = find_evoformer_blocks(model)

    for layer_idx, block in enumerate(blocks):
        if not _selected(layer_idx, recorder.config.layers):
            continue

        def block_hook(module, inputs, output, layer_idx=layer_idx):
            if not recorder.enabled:
                return

            if isinstance(output, (tuple, list)):
                if recorder.config.capture_msa and len(output) > 0 and isinstance(output[0], torch.Tensor):
                    recorder.record(f"layer_{layer_idx:02d}.msa", output[0])
                if recorder.config.capture_pair and len(output) > 1 and isinstance(output[1], torch.Tensor):
                    recorder.record(f"layer_{layer_idx:02d}.pair", output[1])
            elif isinstance(output, dict):
                if recorder.config.capture_msa:
                    for k, v in output.items():
                        if "msa" in str(k).lower() and isinstance(v, torch.Tensor):
                            recorder.record(f"layer_{layer_idx:02d}.msa", v)
                            break
                if recorder.config.capture_pair:
                    for k, v in output.items():
                        if "pair" in str(k).lower() and isinstance(v, torch.Tensor):
                            recorder.record(f"layer_{layer_idx:02d}.pair", v)
                            break
            elif isinstance(output, torch.Tensor):
                recorder.record(f"layer_{layer_idx:02d}.output", output)

        handles.append(block.register_forward_hook(block_hook))

    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def attach_and_run_on_batch(model, batch: Dict[str, Any], layers=None, device: str = "cpu", out_path: str | Path | None = None):
    model = model.to(device)
    cfg = InstrumentationConfig(layers=layers, capture_msa=True, capture_pair=True)
    recorder = EvoformerRecorder(cfg)
    handles = attach_evoformer_block_output_hooks(model, recorder)

    with torch.no_grad():
        _ = model(batch)

    remove_hooks(handles)

    if out_path is not None:
        recorder.save(out_path)

    return recorder


def make_dummy_batch(n_res=64, n_seq=16, c_msa=49, device="cpu"):
    return {
        "target_feat": torch.randn(1, n_res, 22, device=device),
        "residue_index": torch.arange(n_res, device=device).unsqueeze(0),
        "msa_feat": torch.randn(1, n_seq, n_res, c_msa, device=device),
        "seq_mask": torch.ones(1, n_res, device=device),
        "msa_mask": torch.ones(1, n_seq, n_res, device=device),
        "aatype": torch.zeros(1, n_res, dtype=torch.long, device=device),
    }
