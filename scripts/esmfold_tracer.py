"""
ESMFold Tracer Prototype (VizFold Foundation)

- Loads ESMFold (fair-esm)
- Runs inference on a sample sequence
- Captures layer-wise activations using forward hooks
- Saves predicted structure (PDB) to outputs/

This is a first step toward integrating ESMFold traces into VizFold's
archive + visualization pipeline (Issue #43).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import esm


TensorOrTuple = Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]


@dataclass
class TraceSummary:
    num_layers_captured: int
    first_shape: Optional[Tuple[int, ...]]
    last_shape: Optional[Tuple[int, ...]]
    pdb_path: str


class ESMFoldTracer:
    def __init__(self, device: Optional[str] = None):
        print("Loading ESMFold model...")

        self.model = esm.pretrained.esmfold_v1()
        self.model.eval()

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.device = device
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")

        self.activations: List[torch.Tensor] = []
        self._hooks = []

    def _activation_hook(self, module, inputs, output: TensorOrTuple):
        # Some modules return tuples; capture the first tensor-like output
        if isinstance(output, (tuple, list)) and len(output) > 0:
            output = output[0]

        if torch.is_tensor(output):
            self.activations.append(output.detach().to("cpu"))

    def register_hooks(self):
        # Hook each transformer layer in the ESM backbone
        for layer in self.model.esm.layers:
            h = layer.register_forward_hook(self._activation_hook)
            self._hooks.append(h)

    def clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def run(self, sequence: str, out_pdb_path: str = "outputs/esmfold_traced.pdb") -> TraceSummary:
        self.activations.clear()
        self.register_hooks()

        os.makedirs(os.path.dirname(out_pdb_path), exist_ok=True)

        with torch.no_grad():
            pdb_str = self.model.infer_pdb(sequence)

        with open(out_pdb_path, "w") as f:
            f.write(pdb_str)

        self.clear_hooks()

        first_shape = tuple(self.activations[0].shape) if self.activations else None
        last_shape = tuple(self.activations[-1].shape) if self.activations else None

        return TraceSummary(
            num_layers_captured=len(self.activations),
            first_shape=first_shape,
            last_shape=last_shape,
            pdb_path=out_pdb_path,
        )


def main():
    # Small sequence to keep runtime reasonable
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQ"

    tracer = ESMFoldTracer()
    summary = tracer.run(sequence)

    print("Inference complete.")
    print("Saved PDB to:", summary.pdb_path)
    print("Captured activation tensors:", summary.num_layers_captured)
    print("First activation shape:", summary.first_shape)
    print("Last activation shape:", summary.last_shape)


if __name__ == "__main__":
    main()