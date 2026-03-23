from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import AttentionSlice, StructureData, TraceMetadata
from .trace_reader import TraceReader

try:
    import zarr  # type: ignore
except Exception:
    zarr = None


class ArchiveReader(TraceReader):
    """
    Schema-aware scaffolding for the future standardized archive from issue #39.

    This reader intentionally does NOT hard-code the current issue-39 prototype
    layouts. Instead, it:
    - detects archive kind (currently only probes Zarr when available)
    - normalizes metadata into the TraceMetadata contract
    - exposes capability discovery now
    - defers actual tensor-path loading until the VizFold protein schema is final
    """

    def __init__(self, archive_root: str | Path) -> None:
        self.archive_root = Path(archive_root)
        self._zarr_root = None
        self._probe = self._probe_archive()

    def metadata(self) -> TraceMetadata:
        return TraceMetadata(
            protein_id=self._probe["protein_id"],
            source_root=self.archive_root,
            fasta_path=self._path_or_none(self._probe.get("fasta_path")),
            pdb_path=self._path_or_none(self._probe.get("pdb_path")),
            sequence=self._probe.get("sequence"),
            attention_types=self._probe["attention_types"],
            layers_by_type=self._probe["layers_by_type"],
            heads_by_type=self._probe["heads_by_type"],
            residue_indices_by_type=self._probe["residue_indices_by_type"],
            schema_version=self._probe.get("schema_version"),
            archive_kind=self._probe.get("archive_kind"),
            model_family=self._probe.get("model_family"),
            model_version=self._probe.get("model_version"),
            structure_available="structure" in self._probe["capabilities"],
            capabilities=self._probe["capabilities"],
            extras={
                "sequence_length": self._probe.get("sequence_length"),
                "raw_metadata": self._probe.get("raw_metadata", {}),
            },
        )

    def list_attention_types(self) -> list[str]:
        return list(self._probe["attention_types"])

    def list_layers(self, attention_type: str) -> list[int]:
        return list(self._probe["layers_by_type"].get(attention_type, []))

    def list_heads(
        self,
        attention_type: str,
        layer: int,
        residue_idx: int | None = None,
    ) -> list[int]:
        # Today we ignore residue_idx at the indexing layer unless the finalized
        # schema eventually needs a residue-specific head map.
        return list(self._probe["heads_by_type"].get(attention_type, {}).get(layer, []))

    def list_residue_indices(
        self,
        attention_type: str,
        layer: int,
    ) -> list[int]:
        return list(
            self._probe["residue_indices_by_type"].get(attention_type, {}).get(layer, [])
        )

    def load_attention(
        self,
        attention_type: str,
        layer: int,
        head: int,
        residue_idx: int | None = None,
        top_k: int | None = None,
    ) -> AttentionSlice:
        raise self._not_ready(
            "load_attention",
            attention_type=attention_type,
            layer=layer,
            head=head,
            residue_idx=residue_idx,
            top_k=top_k,
        )

    def load_attention_heads(
        self,
        attention_type: str,
        layer: int,
        residue_idx: int | None = None,
        top_k: int | None = None,
    ) -> dict[int, AttentionSlice]:
        raise self._not_ready(
            "load_attention_heads",
            attention_type=attention_type,
            layer=layer,
            residue_idx=residue_idx,
            top_k=top_k,
        )

    def load_structure(self) -> StructureData:
        raise self._not_ready("load_structure")

    def _probe_archive(self) -> dict[str, Any]:
        if not self.archive_root.exists():
            raise FileNotFoundError(f"Archive path does not exist: {self.archive_root}")

        probe: dict[str, Any] = {
            "protein_id": self.archive_root.stem,
            "schema_version": None,
            "archive_kind": None,
            "model_family": None,
            "model_version": None,
            "sequence": None,
            "sequence_length": None,
            "fasta_path": None,
            "pdb_path": None,
            "attention_types": [],
            "layers_by_type": {},
            "heads_by_type": {},
            "residue_indices_by_type": {},
            "capabilities": set(),
            "raw_metadata": {},
        }

        # Sidecar metadata.json support is a safe, schema-neutral bridge.
        metadata_path = self.archive_root / "metadata.json"
        if metadata_path.exists():
            payload = self._load_json(metadata_path)
            probe["raw_metadata"]["metadata.json"] = payload
            self._merge_probe(probe, self._normalize_metadata_payload(payload))

        # Probe Zarr archives when zarr is installed.
        if self.archive_root.suffix == ".zarr":
            probe["archive_kind"] = "zarr"
            probe["capabilities"].add("partial_loading")

            if zarr is not None:
                root = zarr.open(str(self.archive_root), mode="r")
                self._zarr_root = root

                attrs = dict(getattr(root, "attrs", {}))
                probe["raw_metadata"]["zarr_attrs"] = attrs
                self._merge_probe(probe, self._normalize_metadata_payload(attrs))

                self._infer_capabilities_from_zarr_root(probe, root)

        probe["attention_types"] = sorted(set(probe["attention_types"]))
        probe["layers_by_type"] = {
            attn_type: sorted(set(layers))
            for attn_type, layers in probe["layers_by_type"].items()
        }
        probe["heads_by_type"] = {
            attn_type: {
                int(layer): sorted(set(heads))
                for layer, heads in layer_map.items()
            }
            for attn_type, layer_map in probe["heads_by_type"].items()
        }
        probe["residue_indices_by_type"] = {
            attn_type: {
                int(layer): sorted(set(indices))
                for layer, indices in layer_map.items()
            }
            for attn_type, layer_map in probe["residue_indices_by_type"].items()
        }
        probe["capabilities"] = sorted(probe["capabilities"])

        return probe

    def _normalize_metadata_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "schema_version": self._first_non_empty(
                payload.get("schema_version"),
                payload.get("archive_version"),
                payload.get("spec_version"),
            ),
            "archive_kind": payload.get("archive_kind"),
            "model_family": self._first_non_empty(
                payload.get("model_family"),
                payload.get("model_name"),
            ),
            "model_version": payload.get("model_version"),
            "protein_id": self._first_non_empty(
                payload.get("protein_id"),
                payload.get("trace_id"),
                payload.get("sample_id"),
            ),
            "sequence": payload.get("sequence"),
            "sequence_length": payload.get("sequence_length"),
            "fasta_path": payload.get("fasta_path"),
            "pdb_path": payload.get("pdb_path"),
        }

        capabilities = payload.get("capabilities")
        if isinstance(capabilities, (list, tuple, set)):
            normalized["capabilities"] = {str(x) for x in capabilities}

        attention_index = payload.get("attention_index")
        if not isinstance(attention_index, dict):
            attention_index = payload

        normalized.update(self._normalize_attention_index(attention_index))
        return normalized

    def _normalize_attention_index(self, raw: dict[str, Any]) -> dict[str, Any]:
        attention_types = [str(x) for x in raw.get("attention_types", [])]

        layers_by_type = self._normalize_layers_by_type(raw.get("layers_by_type", {}))
        heads_by_type = self._normalize_nested_index(raw.get("heads_by_type", {}))
        residue_indices_by_type = self._normalize_nested_index(
            raw.get("residue_indices_by_type", {})
        )

        if not attention_types:
            attention_types = sorted(
                set(layers_by_type.keys())
                | set(heads_by_type.keys())
                | set(residue_indices_by_type.keys())
            )

        normalized: dict[str, Any] = {
            "attention_types": attention_types,
            "layers_by_type": layers_by_type,
            "heads_by_type": heads_by_type,
            "residue_indices_by_type": residue_indices_by_type,
        }

        if attention_types:
            normalized["capabilities"] = {"attention_index"}

        if any(residue_indices_by_type.values()):
            normalized.setdefault("capabilities", set()).add("residue_indexed_attention")

        return normalized

    @staticmethod
    def _normalize_layers_by_type(raw: Any) -> dict[str, list[int]]:
        if not isinstance(raw, dict):
            return {}

        out: dict[str, list[int]] = {}
        for attention_type, layers in raw.items():
            if not isinstance(layers, (list, tuple)):
                continue
            out[str(attention_type)] = [int(layer) for layer in layers]
        return out

    @staticmethod
    def _normalize_nested_index(raw: Any) -> dict[str, dict[int, list[int]]]:
        if not isinstance(raw, dict):
            return {}

        out: dict[str, dict[int, list[int]]] = {}
        for attention_type, layer_map in raw.items():
            if not isinstance(layer_map, dict):
                continue

            normalized_layer_map: dict[int, list[int]] = {}
            for layer, values in layer_map.items():
                if not isinstance(values, (list, tuple)):
                    continue
                normalized_layer_map[int(layer)] = [int(v) for v in values]

            out[str(attention_type)] = normalized_layer_map

        return out

    def _infer_capabilities_from_zarr_root(self, probe: dict[str, Any], root: Any) -> None:
        names = set()

        try:
            names.update(root.group_keys())
        except Exception:
            pass

        try:
            names.update(root.array_keys())
        except Exception:
            pass

        lowered = {name.lower() for name in names}

        if any(name in lowered for name in ("attention", "attn", "representations")):
            probe["capabilities"].add("attention")

        if any(name in lowered for name in ("structure", "structures", "pdb")):
            probe["capabilities"].add("structure")

        if any(name in lowered for name in ("metadata", "meta")):
            probe["capabilities"].add("metadata")

    @staticmethod
    def _merge_probe(base: dict[str, Any], new: dict[str, Any]) -> None:
        scalar_keys = (
            "protein_id",
            "schema_version",
            "archive_kind",
            "model_family",
            "model_version",
            "sequence",
            "sequence_length",
            "fasta_path",
            "pdb_path",
        )
        for key in scalar_keys:
            value = new.get(key)
            if value is not None:
                base[key] = value

        if new.get("attention_types"):
            base["attention_types"] = list(
                set(base["attention_types"]) | set(new["attention_types"])
            )

        for key in ("layers_by_type", "heads_by_type", "residue_indices_by_type"):
            incoming = new.get(key, {})
            if not incoming:
                continue

            if key == "layers_by_type":
                for attention_type, layers in incoming.items():
                    base[key].setdefault(attention_type, [])
                    base[key][attention_type].extend(layers)
            else:
                for attention_type, layer_map in incoming.items():
                    base[key].setdefault(attention_type, {})
                    for layer, values in layer_map.items():
                        base[key][attention_type].setdefault(int(layer), [])
                        base[key][attention_type][int(layer)].extend(values)

        for cap in new.get("capabilities", set()):
            base["capabilities"].add(cap)

    def _not_ready(self, method_name: str, **kwargs: Any) -> NotImplementedError:
        details = {
            "archive_kind": self._probe.get("archive_kind"),
            "schema_version": self._probe.get("schema_version"),
            "capabilities": self._probe.get("capabilities"),
        }
        if kwargs:
            details["request"] = kwargs

        return NotImplementedError(
            f"{method_name} is intentionally deferred until issue #39 finalizes "
            f"the VizFold protein archive schema. Probe summary: {details}"
        )

    @staticmethod
    def _first_non_empty(*values: Any) -> Any:
        for value in values:
            if value is not None and value != "":
                return value
        return None

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
        return data

    @staticmethod
    def _path_or_none(value: Any) -> Path | None:
        if value is None:
            return None
        return Path(value)