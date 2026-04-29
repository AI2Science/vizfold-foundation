from __future__ import annotations

import shutil
import tempfile
import zipfile

import re
from pathlib import Path
from typing import Any

import numpy as np

from .models import AttentionConnection, AttentionSlice, StructureData, TraceMetadata
from .trace_reader import TraceReader

try:
    import zarr  # type: ignore
except Exception:  # pragma: no cover
    zarr = None


_LAYER_RE = re.compile(r"(?:^|[/_\-])layer[_\-]?(\d+)$|^(\d+)$")


class ArchiveReader(TraceReader):
    """
    Working reader for standardized VizFold/OpenFold Zarr trace archives.

    Primary supported issue-39 layout:

        metadata/
        representations/single/layer_00
        representations/pair/layer_00
        attention/triangle_start/layer_00
        structure/atom_positions

    Also tolerates prototype layouts like:

        attention/layer_0
        attention                  # shape [layers, heads, N, N]
        inputs/sequence
        outputs/coordinates
        structure_pdb
    """

    def __init__(self, archive_root: str | Path) -> None:
        self.archive_root = Path(archive_root)
        self._store: Any | None = None
        self._extracted_archive_root: Path | None = None
        self._root = self._open_root(self.archive_root)
        self._metadata_cache: TraceMetadata | None = None

    def metadata(self) -> TraceMetadata:
        if self._metadata_cache is not None:
            return self._metadata_cache

        sequence = self.get_sequence()
        attention_types = self.list_attention_types()
        layers_by_type = {
            attn_type: self.list_layers(attn_type)
            for attn_type in attention_types
        }
        heads_by_type = {
            attn_type: {
                layer: self.list_heads(attn_type, layer)
                for layer in layers
            }
            for attn_type, layers in layers_by_type.items()
        }

        residue_indices_by_type: dict[str, dict[int, list[int]]] = {}
        for attn_type, layers in layers_by_type.items():
            for layer in layers:
                indices = self.list_residue_indices(attn_type, layer)
                if indices:
                    residue_indices_by_type.setdefault(attn_type, {})[layer] = indices

        has_structure = (
            self.get_pdb_string() is not None
            or self._find_first_array(
                "structure/atom_positions",
                "outputs/coordinates",
                "coordinates",
            )
            is not None
        )

        capabilities = ["metadata", "partial_loading"]
        if attention_types:
            capabilities.append("attention")
        if has_structure:
            capabilities.append("structure")

        self._metadata_cache = TraceMetadata(
            protein_id=str(self._root.attrs.get("protein_id", self.archive_root.stem)),
            source_root=self.archive_root,
            sequence=sequence,
            attention_types=attention_types,
            layers_by_type=layers_by_type,
            heads_by_type=heads_by_type,
            residue_indices_by_type=residue_indices_by_type,
            schema_version=self._read_text_metadata(
                "schema_version",
                "archive_version",
                "spec_version",
            ),
            archive_kind="zarr",
            model_family=self._read_text_metadata("model_family", "model_name"),
            model_version=self._read_text_metadata("model_version", "config_version"),
            structure_available=has_structure,
            capabilities=capabilities,
            extras={
                "num_residues": self._read_int_metadata("num_residues"),
                "num_recycles": self._read_int_metadata("num_recycles"),
                "arrays": self.list_all_arrays(),
            },
        )
        return self._metadata_cache

    def list_attention_types(self) -> list[str]:
        types: set[str] = set()

        if "attention" in self._root:
            node = self._root["attention"]

            if self._is_group(node):
                for key in self._keys(node):
                    child = node[key]
                    if self._is_group(child):
                        types.add(str(key))

                direct_layer_arrays = [
                    key
                    for key in self._keys(node)
                    if self._is_array(node[key])
                    and self._parse_layer_key(key) is not None
                ]
                if direct_layer_arrays:
                    types.add("triangle_start")

            elif self._is_array(node):
                types.add("attention")

        for name, shape in self.list_attention_arrays().items():
            if name != "attention" and not name.startswith("attention/"):
                types.add(name)

        return sorted(types)

    def list_layers(self, attention_type: str) -> list[int]:
        return sorted(self._discover_layer_numbers(attention_type))

    def list_heads(
        self,
        attention_type: str,
        layer: int,
        residue_idx: int | None = None,
    ) -> list[int]:
        arr = self._load_attention_tensor(attention_type, layer)
        return list(range(self._num_heads_from_shape(arr.shape)))

    def list_residue_indices(self, attention_type: str, layer: int) -> list[int]:
        arr = self._load_attention_tensor(attention_type, layer)
        if arr.ndim != 4:
            return []

        n = self._residue_count_from_attention_shape(arr.shape)
        return list(range(n)) if n is not None else []

    def load_attention(
        self,
        attention_type: str,
        layer: int,
        head: int,
        residue_idx: int | None = None,
        top_k: int | None = None,
    ) -> AttentionSlice:
        arr = self._load_attention_tensor(attention_type, layer)
        matrix = self._slice_head_matrix(arr, head=head, residue_idx=residue_idx)
        connections = self._matrix_to_connections(matrix, top_k=top_k)

        return AttentionSlice(
            attention_type=attention_type,
            layer=layer,
            head=head,
            residue_idx=residue_idx,
            connections=connections,
        )

    def load_attention_heads(
        self,
        attention_type: str,
        layer: int,
        residue_idx: int | None = None,
        top_k: int | None = None,
    ) -> dict[int, AttentionSlice]:
        return {
            head: self.load_attention(
                attention_type=attention_type,
                layer=layer,
                head=head,
                residue_idx=residue_idx,
                top_k=top_k,
            )
            for head in self.list_heads(attention_type, layer, residue_idx)
        }

    def load_structure(self) -> StructureData:
        pdb_text = self.get_pdb_string()
        sequence = self.get_sequence()

        if pdb_text is None:
            coords = self._find_first_array(
                "structure/atom_positions",
                "outputs/coordinates",
                "coordinates",
            )
            if coords is not None:
                pdb_text = self._coords_to_pdb(np.asarray(coords), sequence)

        protein_id = (
            self._metadata_cache.protein_id
            if self._metadata_cache is not None
            else str(self._root.attrs.get("protein_id", self.archive_root.stem))
        )

        return StructureData(
            protein_id=protein_id,
            pdb_path=None,
            pdb_text=pdb_text,
            sequence=sequence,
        )

    def load_single_representation(self, layer: int) -> np.ndarray:
        node = self._resolve_layered_array(
            layer,
            "representations/single",
            "single",
            "activations",
            dataset_names=("single_repr", "activation", "values"),
        )
        if node is None:
            raise KeyError(f"Single representation not found for layer {layer}")
        return self._to_numpy(node)

    def load_pair_representation(self, layer: int) -> np.ndarray:
        node = self._resolve_layered_array(
            layer,
            "representations/pair",
            "pair",
            "activations",
            dataset_names=("pair_repr", "values"),
        )
        if node is None:
            raise KeyError(f"Pair representation not found for layer {layer}")
        return self._to_numpy(node)

    def list_all_arrays(self) -> dict[str, tuple[int, ...]]:
        return {
            name: tuple(int(x) for x in array.shape)
            for name, array in self._walk_arrays(self._root)
        }

    def list_attention_arrays(self) -> dict[str, tuple[int, ...]]:
        out: dict[str, tuple[int, ...]] = {}
        for name, shape in self.list_all_arrays().items():
            lowered = name.lower()
            if self._is_attention_shape(shape) and (
                "att" in lowered or "triangle" in lowered or name == "attention"
            ):
                out[name] = shape
        return out

    def get_sequence(self) -> str | None:
        value = self._read_text_metadata("sequence")
        if value:
            return value

        for path in ("inputs/sequence", "sequence"):
            arr = self._get_array(path)
            if arr is not None:
                text = self._array_to_text(arr)
                if text:
                    return text
        return None

    def get_pdb_string(self) -> str | None:
        for path in (
            "structure_pdb",
            "structure/pdb",
            "structure/pdb_text",
            "outputs/structure_pdb",
        ):
            arr = self._get_array(path)
            if arr is not None:
                text = self._array_to_text(arr)
                if text:
                    return text
        return None

    def _open_root(self, path: Path) -> Any:
        if zarr is None:
            raise ImportError("ArchiveReader requires zarr. Install with `pip install zarr`.")

        if not path.exists():
            raise FileNotFoundError(f"Archive path does not exist: {path}")

        if path.suffix == ".zip":
            extracted_dir = Path(tempfile.mkdtemp(prefix="vizfold_zarr_"))

            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(extracted_dir)

            # Case 1: user zipped the contents of the .zarr folder
            if (extracted_dir / "zarr.json").exists() or (extracted_dir / ".zgroup").exists():
                self._extracted_archive_root = extracted_dir
                return zarr.open_group(str(extracted_dir), mode="r")

            # Case 2: user zipped the .zarr folder itself
            zarr_dirs = list(extracted_dir.glob("*.zarr"))
            if zarr_dirs:
                self._extracted_archive_root = zarr_dirs[0]
                return zarr.open_group(str(zarr_dirs[0]), mode="r")

            # Case 3: zip contains one top-level folder
            child_dirs = [p for p in extracted_dir.iterdir() if p.is_dir()]
            if len(child_dirs) == 1:
                self._extracted_archive_root = child_dirs[0]
                return zarr.open_group(str(child_dirs[0]), mode="r")

            raise ValueError(f"Could not find Zarr archive root inside zip file: {path}")

        return zarr.open_group(str(path), mode="r")

    def _discover_layer_numbers(self, attention_type: str) -> set[int]:
        layers: set[int] = set()

        group = self._attention_group_for_type(attention_type)
        if group is not None:
            for key in self._keys(group):
                child = group[key]
                layer = self._parse_layer_key(key)
                if layer is not None and self._is_array_or_group(child):
                    layers.add(layer)

        arr = self._attention_array_for_type(attention_type)
        if arr is not None and arr.ndim == 4 and self._looks_like_layered_attention(arr.shape):
            layers.update(range(int(arr.shape[0])))

        if layers:
            return layers

        for name in self.list_attention_arrays():
            layer = self._parse_layer_key(name.split("/")[-1])
            if layer is not None:
                layers.add(layer)

        return layers

    def _load_attention_tensor(self, attention_type: str, layer: int) -> np.ndarray:
        layered = self._attention_array_for_type(attention_type)
        if (
            layered is not None
            and layered.ndim == 4
            and self._looks_like_layered_attention(layered.shape)
        ):
            if layer >= layered.shape[0]:
                raise IndexError(f"Layer {layer} out of range for {attention_type}")
            return np.asarray(layered[layer])

        group = self._attention_group_for_type(attention_type)
        if group is not None:
            for layer_key in self._layer_key_candidates(layer):
                if layer_key not in group:
                    continue

                node = group[layer_key]
                if self._is_array(node):
                    return self._to_numpy(node)

                if self._is_group(node):
                    for dataset_name in ("attention", "values", "heads"):
                        if dataset_name in node and self._is_array(node[dataset_name]):
                            return np.asarray(node[dataset_name])

        for layer_key in self._layer_key_candidates(layer):
            arr = self._get_array(f"attention/{layer_key}")
            if arr is not None:
                return np.asarray(arr)

        arr = self._get_array(attention_type)
        if arr is not None:
            data = np.asarray(arr)
            if data.ndim == 4 and self._looks_like_layered_attention(data.shape):
                return data[layer]
            return data

        raise KeyError(f"Attention not found for type={attention_type}, layer={layer}")

    def _attention_group_for_type(self, attention_type: str) -> Any | None:
        candidates = []

        if attention_type != "attention":
            candidates.append(f"attention/{attention_type}")

        candidates.append("attention")

        for path in candidates:
            node = self._get_node(path)
            if node is not None and self._is_group(node):
                return node

        return None

    def _attention_array_for_type(self, attention_type: str) -> Any | None:
        candidates = []

        if attention_type != "attention":
            candidates.append(f"attention/{attention_type}")

        candidates.extend((attention_type, "attention"))

        for path in candidates:
            node = self._get_node(path)
            if node is not None and self._is_array(node):
                return node

        return None

    def _resolve_layered_array(
        self,
        layer: int,
        *groups: str,
        dataset_names: tuple[str, ...],
    ) -> Any | None:
        for group_path in groups:
            group = self._get_node(group_path)
            if group is None or not self._is_group(group):
                continue

            for layer_key in self._layer_key_candidates(layer):
                if layer_key not in group:
                    continue

                node = group[layer_key]

                if self._is_array(node):
                    return node

                if self._is_group(node):
                    for name in dataset_names:
                        if name in node and self._is_array(node[name]):
                            return node[name]

        return None

    @staticmethod
    def _slice_head_matrix(
        arr: np.ndarray,
        head: int,
        residue_idx: int | None,
    ) -> np.ndarray:
        if arr.ndim == 2:
            if head != 0:
                raise IndexError("2-D attention matrix only has head 0")
            return arr

        if arr.ndim == 3:
            axis = ArchiveReader._infer_head_axis(arr.shape)

            if axis == 0:
                return arr[head, :, :]

            if axis == 2:
                return arr[:, :, head]

            raise ValueError(f"Cannot infer head axis from attention shape {arr.shape}")

        if arr.ndim == 4:
            axis = ArchiveReader._infer_head_axis(arr.shape)

            if axis == 0:
                cube = arr[head]

            elif axis == 3:
                cube = arr[:, :, :, head]

            else:
                raise ValueError(f"Cannot infer head axis from attention shape {arr.shape}")

            if residue_idx is None:
                return np.asarray(cube).mean(axis=0)

            return cube[residue_idx, :, :]

        raise ValueError(f"Unsupported attention tensor shape {arr.shape}")

    @staticmethod
    def _matrix_to_connections(
        matrix: np.ndarray,
        top_k: int | None,
    ) -> list[AttentionConnection]:
        mat = np.asarray(matrix, dtype=float)

        if mat.ndim != 2:
            raise ValueError(f"Expected 2-D attention matrix, got {mat.shape}")

        flat = mat.reshape(-1)

        if flat.size == 0 or top_k == 0:
            return []

        k = flat.size if top_k is None else min(int(top_k), flat.size)

        if k < flat.size:
            idx = np.argpartition(-flat, k - 1)[:k]
            idx = idx[np.argsort(-flat[idx])]
        else:
            idx = np.argsort(-flat)

        n_cols = mat.shape[1]
        connections: list[AttentionConnection] = []

        for flat_idx in idx:
            weight = float(flat[flat_idx])

            if np.isnan(weight):
                continue

            src = int(flat_idx // n_cols)
            dst = int(flat_idx % n_cols)

            connections.append(
                AttentionConnection(
                    src=src,
                    dst=dst,
                    weight=weight,
                )
            )

        return connections

    @staticmethod
    def _infer_head_axis(shape: tuple[int, ...]) -> int:
        if len(shape) == 3:
            if shape[0] <= 64 and shape[1] == shape[2]:
                return 0
            if shape[2] <= 64 and shape[0] == shape[1]:
                return 2

        if len(shape) == 4:
            if shape[0] <= 64 and shape[1] == shape[2] == shape[3]:
                return 0
            if shape[3] <= 64 and shape[0] == shape[1] == shape[2]:
                return 3

        raise ValueError(f"Cannot infer head axis from shape {shape}")

    @staticmethod
    def _num_heads_from_shape(shape: tuple[int, ...]) -> int:
        if len(shape) == 2:
            return 1

        axis = ArchiveReader._infer_head_axis(shape)
        return int(shape[axis])

    @staticmethod
    def _residue_count_from_attention_shape(shape: tuple[int, ...]) -> int | None:
        if len(shape) == 2 and shape[0] == shape[1]:
            return int(shape[0])

        if len(shape) == 3:
            axis = ArchiveReader._infer_head_axis(shape)
            return int(shape[1] if axis == 0 else shape[0])

        if len(shape) == 4:
            axis = ArchiveReader._infer_head_axis(shape)
            return int(shape[1] if axis == 0 else shape[0])

        return None

    @staticmethod
    def _looks_like_layered_attention(shape: tuple[int, ...]) -> bool:
        return len(shape) == 4 and shape[1] <= 64 and shape[2] == shape[3]

    @staticmethod
    def _is_attention_shape(shape: tuple[int, ...]) -> bool:
        if len(shape) == 2:
            return shape[0] == shape[1]

        if len(shape) == 3:
            return (
                shape[0] <= 64
                and shape[1] == shape[2]
            ) or (
                shape[2] <= 64
                and shape[0] == shape[1]
            )

        if len(shape) == 4:
            return (
                shape[0] <= 64
                and shape[1] == shape[2] == shape[3]
            ) or (
                shape[1] <= 64
                and shape[2] == shape[3]
            ) or (
                shape[3] <= 64
                and shape[0] == shape[1] == shape[2]
            )

        return False

    def _read_text_metadata(self, *keys: str) -> str | None:
        for key in keys:
            if key in self._root.attrs and self._root.attrs[key] is not None:
                return str(self._root.attrs[key])

            arr = self._get_array(f"metadata/{key}")
            if arr is not None:
                text = self._array_to_text(arr)
                if text:
                    return text

            meta = self._get_node("metadata")
            if meta is not None and self._is_group(meta) and key in meta.attrs:
                return str(meta.attrs[key])

        return None

    def _read_int_metadata(self, key: str) -> int | None:
        text = self._read_text_metadata(key)

        if text is None:
            return None

        try:
            return int(float(text))
        except ValueError:
            return None

    def _find_first_array(self, *paths: str) -> np.ndarray | None:
        for path in paths:
            arr = self._get_array(path)
            if arr is not None:
                return self._to_numpy(arr)

        return None

    def _get_array(self, path: str) -> Any | None:
        node = self._get_node(path)

        if node is not None and self._is_array(node):
            return node

        return None

    def _get_node(self, path: str) -> Any | None:
        node = self._root

        if not path:
            return node

        for part in path.strip("/").split("/"):
            if not self._is_group(node) or part not in node:
                return None

            node = node[part]

        return node

    @staticmethod
    def _keys(group: Any) -> list[str]:
        return sorted(str(k) for k in group.keys())

    @staticmethod
    def _is_array(node: Any) -> bool:
        return hasattr(node, "shape") and hasattr(node, "dtype")

    @staticmethod
    def _is_group(node: Any) -> bool:
        return hasattr(node, "keys") and not ArchiveReader._is_array(node)

    @staticmethod
    def _is_array_or_group(node: Any) -> bool:
        return ArchiveReader._is_array(node) or ArchiveReader._is_group(node)

    def _walk_arrays(self, group: Any, prefix: str = ""):
        for key in self._keys(group):
            node = group[key]
            name = f"{prefix}/{key}" if prefix else key

            if self._is_array(node):
                yield name, node

            elif self._is_group(node):
                yield from self._walk_arrays(node, name)

    @staticmethod
    def _parse_layer_key(key: str) -> int | None:
        match = _LAYER_RE.search(key)

        if not match:
            return None

        value = match.group(1) or match.group(2)
        return int(value)

    @staticmethod
    def _layer_key_candidates(layer: int) -> tuple[str, ...]:
        return (
            f"layer_{layer:02d}",
            f"layer_{layer}",
            str(layer),
        )

    @staticmethod
    def _array_to_text(array: Any) -> str | None:
        value = ArchiveReader._to_numpy(array)

        if value.shape == ():
            item = value.item()

            if isinstance(item, bytes):
                return item.decode("utf-8")

            return str(item)

        if value.dtype.kind in {"U", "S", "O"}:
            parts = []

            for item in value.reshape(-1):
                if isinstance(item, bytes):
                    parts.append(item.decode("utf-8"))
                else:
                    parts.append(str(item))

            return "".join(parts)

        if value.dtype == np.uint8:
            try:
                return bytes(value.reshape(-1)).decode("utf-8")
            except UnicodeDecodeError:
                return None

        return None

    @staticmethod
    def _coords_to_pdb(coords: np.ndarray, sequence: str | None = None) -> str:
        arr = np.asarray(coords, dtype=float)

        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr[:, 1, :]

        elif (
            arr.ndim == 2
            and arr.shape[1] == 3
            and sequence
            and arr.shape[0] == len(sequence) * 37
        ):
            arr = arr.reshape(len(sequence), 37, 3)[:, 1, :]

        elif arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Cannot convert coordinates with shape {coords.shape} to PDB")

        lines = []

        for idx, (x, y, z) in enumerate(arr, start=1):
            res = sequence[idx - 1] if sequence and idx - 1 < len(sequence) else "X"
            lines.append(
                f"ATOM  {idx:5d}  CA  {res:>3s} A{idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )

        lines.append("END")
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _to_numpy(array: Any) -> np.ndarray:
        if hasattr(array, "shape") and hasattr(array, "__getitem__"):
            if tuple(array.shape) == ():
                return np.asarray(array[()])
            return np.asarray(array[:])

        return np.asarray(array)