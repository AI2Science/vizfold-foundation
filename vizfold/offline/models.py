from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AttentionConnection:
    """One residue-residue edge with a scalar attention weight."""

    src: int
    dst: int
    weight: float


@dataclass(frozen=True)
class AttentionSlice:
    """
    One logical attention slice.

    Examples:
    - MSA row attention at one layer/head
    - Triangle-start attention at one layer/residue/head
    """

    attention_type: str
    layer: int
    head: int
    residue_idx: int | None
    connections: list[AttentionConnection]

    def top_k(self, k: int) -> "AttentionSlice":
        if k < 0:
            raise ValueError("k must be >= 0")
        return replace(self, connections=self.connections[:k])

    def as_triplets(self) -> list[tuple[int, int, float]]:
        return [(c.src, c.dst, c.weight) for c in self.connections]


@dataclass(frozen=True)
class StructureData:
    """
    Minimal structure payload for offline visualization.
    """

    protein_id: str
    pdb_path: Path | None
    pdb_text: str | None
    sequence: str | None = None


@dataclass(frozen=True)
class TraceMetadata:
    """
    Metadata for one trace source.

    heads_by_type maps: attention_type -> layer -> list[head_idx]
    residue_indices_by_type maps: attention_type -> layer -> list[residue_idx]
    """

    protein_id: str
    source_root: Path
    fasta_path: Path | None = None
    pdb_path: Path | None = None
    sequence: str | None = None

    attention_types: list[str] = field(default_factory=list)
    layers_by_type: dict[str, list[int]] = field(default_factory=dict)
    heads_by_type: dict[str, dict[int, list[int]]] = field(default_factory=dict)
    residue_indices_by_type: dict[str, dict[int, list[int]]] = field(default_factory=dict)

    # New normalized archive fields
    schema_version: str | None = None
    archive_kind: str | None = None
    model_family: str | None = None
    model_version: str | None = None
    structure_available: bool = False
    capabilities: list[str] = field(default_factory=list)

    extras: dict[str, Any] = field(default_factory=dict)