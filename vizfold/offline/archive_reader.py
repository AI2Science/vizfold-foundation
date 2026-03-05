from __future__ import annotations

from pathlib import Path

from .models import AttentionSlice, StructureData, TraceMetadata
from .trace_reader import TraceReader


class ArchiveReader(TraceReader):
    """
    Placeholder reader for the future standardized archive from issue #39.

    The point of adding this now is to lock the interface that both the
    frontend and the archive writer will target.
    """

    def __init__(self, archive_root: str | Path) -> None:
        self.archive_root = Path(archive_root)

    def metadata(self) -> TraceMetadata:
        raise NotImplementedError(
            "ArchiveReader is waiting on the finalized archive schema from issue #39."
        )

    def list_attention_types(self) -> list[str]:
        raise NotImplementedError(
            "ArchiveReader is waiting on the finalized archive schema from issue #39."
        )

    def list_layers(self, attention_type: str) -> list[int]:
        raise NotImplementedError(
            "ArchiveReader is waiting on the finalized archive schema from issue #39."
        )

    def list_heads(
        self,
        attention_type: str,
        layer: int,
        residue_idx: int | None = None,
    ) -> list[int]:
        raise NotImplementedError(
            "ArchiveReader is waiting on the finalized archive schema from issue #39."
        )

    def list_residue_indices(
        self,
        attention_type: str,
        layer: int,
    ) -> list[int]:
        raise NotImplementedError(
            "ArchiveReader is waiting on the finalized archive schema from issue #39."
        )

    def load_attention(
        self,
        attention_type: str,
        layer: int,
        head: int,
        residue_idx: int | None = None,
        top_k: int | None = None,
    ) -> AttentionSlice:
        raise NotImplementedError(
            "ArchiveReader is waiting on the finalized archive schema from issue #39."
        )

    def load_attention_heads(
        self,
        attention_type: str,
        layer: int,
        residue_idx: int | None = None,
        top_k: int | None = None,
    ) -> dict[int, AttentionSlice]:
        raise NotImplementedError(
            "ArchiveReader is waiting on the finalized archive schema from issue #39."
        )

    def load_structure(self) -> StructureData:
        raise NotImplementedError(
            "ArchiveReader is waiting on the finalized archive schema from issue #39."
        )