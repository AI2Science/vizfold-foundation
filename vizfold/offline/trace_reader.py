from __future__ import annotations

from abc import ABC, abstractmethod

from .models import AttentionSlice, StructureData, TraceMetadata


class TraceReader(ABC):
    """
    Abstract reader interface for offline inference traces.

    Both legacy text-based readers and future archive readers should implement
    this exact API so downstream notebooks and Streamlit apps can remain stable.
    """

    @abstractmethod
    def metadata(self) -> TraceMetadata:
        raise NotImplementedError

    @abstractmethod
    def list_attention_types(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def list_layers(self, attention_type: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def list_heads(
        self,
        attention_type: str,
        layer: int,
        residue_idx: int | None = None,
    ) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def list_residue_indices(
        self,
        attention_type: str,
        layer: int,
    ) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def load_attention(
        self,
        attention_type: str,
        layer: int,
        head: int,
        residue_idx: int | None = None,
        top_k: int | None = None,
    ) -> AttentionSlice:
        raise NotImplementedError

    @abstractmethod
    def load_attention_heads(
        self,
        attention_type: str,
        layer: int,
        residue_idx: int | None = None,
        top_k: int | None = None,
    ) -> dict[int, AttentionSlice]:
        raise NotImplementedError

    @abstractmethod
    def load_structure(self) -> StructureData:
        raise NotImplementedError