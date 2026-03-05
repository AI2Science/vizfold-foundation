from __future__ import annotations

import re
from pathlib import Path

from .exceptions import TraceFormatError, TraceNotFoundError
from .models import AttentionConnection, AttentionSlice, StructureData, TraceMetadata
from .paths import parse_legacy_attention_filename, resolve_legacy_attention_path
from .trace_reader import TraceReader


class LegacyTxtReader(TraceReader):
    """
    Reader for VizFold's current legacy text-dump attention format.

    This wraps the existing file naming and parsing conventions behind a stable API
    so the frontend can stop depending on hardcoded filenames.
    """

    def __init__(
        self,
        attention_dir: str | Path,
        fasta_path: str | Path | None = None,
        pdb_path: str | Path | None = None,
        protein_id: str | None = None,
    ) -> None:
        self.attention_dir = Path(attention_dir)
        if not self.attention_dir.exists():
            raise TraceNotFoundError(f"attention_dir does not exist: {self.attention_dir}")

        self.fasta_path = Path(fasta_path) if fasta_path is not None else None
        self.pdb_path = Path(pdb_path) if pdb_path is not None else None
        self.protein_id = protein_id or self._infer_protein_id()

        self._metadata_cache: TraceMetadata | None = None

    def metadata(self) -> TraceMetadata:
        if self._metadata_cache is None:
            self._metadata_cache = self._build_metadata()
        return self._metadata_cache

    def list_attention_types(self) -> list[str]:
        return self.metadata().attention_types

    def list_layers(self, attention_type: str) -> list[int]:
        return self.metadata().layers_by_type.get(attention_type, [])

    def list_heads(
        self,
        attention_type: str,
        layer: int,
        residue_idx: int | None = None,
    ) -> list[int]:
        path = resolve_legacy_attention_path(
            self.attention_dir,
            attention_type=attention_type,
            layer=layer,
            residue_idx=residue_idx,
        )
        heads = self._parse_heads_file(path)
        return sorted(heads.keys())

    def list_residue_indices(
        self,
        attention_type: str,
        layer: int,
    ) -> list[int]:
        return self.metadata().residue_indices_by_type.get(attention_type, {}).get(layer, [])

    def load_attention(
        self,
        attention_type: str,
        layer: int,
        head: int,
        residue_idx: int | None = None,
        top_k: int | None = None,
    ) -> AttentionSlice:
        all_heads = self.load_attention_heads(
            attention_type=attention_type,
            layer=layer,
            residue_idx=residue_idx,
            top_k=top_k,
        )

        if head not in all_heads:
            available = sorted(all_heads.keys())
            raise TraceNotFoundError(
                f"Head {head} not found for attention_type={attention_type}, "
                f"layer={layer}, residue_idx={residue_idx}. Available heads: {available}"
            )

        return all_heads[head]

    def load_attention_heads(
        self,
        attention_type: str,
        layer: int,
        residue_idx: int | None = None,
        top_k: int | None = None,
    ) -> dict[int, AttentionSlice]:
        path = resolve_legacy_attention_path(
            self.attention_dir,
            attention_type=attention_type,
            layer=layer,
            residue_idx=residue_idx,
        )
        parsed = self._parse_heads_file(path)

        slices: dict[int, AttentionSlice] = {}
        for head_idx, connections in parsed.items():
            if top_k is not None:
                connections = connections[:top_k]

            slices[head_idx] = AttentionSlice(
                attention_type=attention_type,
                layer=layer,
                head=head_idx,
                residue_idx=residue_idx,
                connections=connections,
            )

        return slices

    def load_structure(self) -> StructureData:
        pdb_text = None
        if self.pdb_path is not None:
            if not self.pdb_path.exists():
                raise TraceNotFoundError(f"pdb_path does not exist: {self.pdb_path}")
            pdb_text = self.pdb_path.read_text(encoding="utf-8")

        sequence = self._read_fasta_sequence(self.fasta_path) if self.fasta_path else None

        return StructureData(
            protein_id=self.protein_id,
            pdb_path=self.pdb_path,
            pdb_text=pdb_text,
            sequence=sequence,
        )

    def _build_metadata(self) -> TraceMetadata:
        layers_by_type: dict[str, set[int]] = {}
        heads_by_type_sets: dict[str, dict[int, set[int]]] = {}
        residue_indices_by_type: dict[str, dict[int, set[int]]] = {}

        for path in self.attention_dir.iterdir():
            if not path.is_file():
                continue

            parsed = parse_legacy_attention_filename(path.name)
            if parsed is None:
                continue

            attention_type, layer, residue_idx = parsed

            layers_by_type.setdefault(attention_type, set()).add(layer)
            heads_by_type_sets.setdefault(attention_type, {}).setdefault(layer, set())

            file_heads = self._parse_heads_file(path).keys()
            heads_by_type_sets[attention_type][layer].update(file_heads)

            if residue_idx is not None:
                residue_indices_by_type.setdefault(attention_type, {}).setdefault(layer, set()).add(
                    residue_idx
                )

        attention_types = sorted(layers_by_type.keys())
        sequence = self._read_fasta_sequence(self.fasta_path) if self.fasta_path else None

        heads_by_type: dict[str, dict[int, list[int]]] = {
            attn_type: {
                layer: sorted(heads)
                for layer, heads in layer_map.items()
            }
            for attn_type, layer_map in heads_by_type_sets.items()
        }

        residue_indices_by_type_sorted: dict[str, dict[int, list[int]]] = {
            attn_type: {
                layer: sorted(residue_indices)
                for layer, residue_indices in layer_map.items()
            }
            for attn_type, layer_map in residue_indices_by_type.items()
        }

        return TraceMetadata(
            protein_id=self.protein_id,
            source_root=self.attention_dir,
            fasta_path=self.fasta_path,
            pdb_path=self.pdb_path,
            sequence=sequence,
            attention_types=attention_types,
            layers_by_type={
                attn_type: sorted(layers)
                for attn_type, layers in layers_by_type.items()
            },
            heads_by_type=heads_by_type,
            residue_indices_by_type=residue_indices_by_type_sorted,
            extras={
                "format": "legacy_txt",
            },
        )

    def _parse_heads_file(self, path: Path) -> dict[int, list[AttentionConnection]]:
        """
        Expected format:

        Layer 47, Head 0
        1 5 0.91
        2 9 0.88

        Layer 47, Head 1
        3 7 0.95
        """
        heads: dict[int, list[AttentionConnection]] = {}
        current_head: int | None = None

        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                if line.lower().startswith("layer"):
                    numbers = re.findall(r"-?\d+", line)
                    if not numbers:
                        raise TraceFormatError(f"Could not parse head header line: {line}")
                    current_head = int(numbers[-1])
                    heads[current_head] = []
                    continue

                if current_head is None:
                    raise TraceFormatError(
                        f"Found attention row before any head header in file: {path}"
                    )

                parts = line.split()
                if len(parts) != 3:
                    raise TraceFormatError(
                        f"Expected 3 columns in attention row, got {len(parts)}: {line}"
                    )

                src = int(float(parts[0]))
                dst = int(float(parts[1]))
                weight = float(parts[2])

                heads[current_head].append(
                    AttentionConnection(src=src, dst=dst, weight=weight)
                )

        for head_idx, conns in heads.items():
            conns.sort(key=lambda x: x.weight, reverse=True)

        return heads

    def _infer_protein_id(self) -> str:
        if self.pdb_path is not None:
            return self.pdb_path.stem
        if self.fasta_path is not None:
            return self.fasta_path.stem
        return self.attention_dir.name

    @staticmethod
    def _read_fasta_sequence(fasta_path: Path | None) -> str | None:
        if fasta_path is None:
            return None
        if not fasta_path.exists():
            raise TraceNotFoundError(f"FASTA path does not exist: {fasta_path}")

        lines = fasta_path.read_text(encoding="utf-8").splitlines()
        seq_lines = [line.strip() for line in lines if line and not line.startswith(">")]
        return "".join(seq_lines)