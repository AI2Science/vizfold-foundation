"""
Backend interface for VizFold.

Implementations (OpenFold, ESMFold) provide model loading,
inference, and trace extraction so the visualization layer
can consume outputs uniformly.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# Trace config: what to extract and where to write
TraceConfig = Dict[str, Any]


class BackendBase(ABC):
    """Base interface for structure-prediction backends."""

    @abstractmethod
    def load_model(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Load the model. Returns the model object (backend-specific)."""
        pass

    @abstractmethod
    def run_inference(
        self,
        fasta_path: str,
        out_dir: str,
        trace_cfg: Optional[TraceConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run inference and optionally write traces.

        Returns a dict with at least: structure_path, meta (dict), optional trace paths.
        """
        pass

    @abstractmethod
    def supports_attention(self) -> bool:
        """Whether this backend can extract attention maps."""
        pass

    @abstractmethod
    def supports_activations(self) -> bool:
        """Whether this backend can extract layer activations (hidden states)."""
        pass
