from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import re
import shlex
from typing import Any, Dict, Iterable, List

import numpy as np

try:
    import zarr
except ImportError as exc:  # pragma: no cover - exercised in environments missing zarr
    raise ImportError(
        "zarr is required for standardizedarchive. Install with `pip install zarr`."
    ) from exc


@dataclass(frozen=True)
class OpenFoldRunResult:
    run_id: str
    score: float
    params: Dict[str, Any]
    output_dir: str
    command: str
    model_output_path: str
    changed_param: str | None = None
    from_value: Any | None = None
    to_value: Any | None = None
    score_delta: float | None = None
    step_index: int | None = None


def score_from_output_dict(output_dict: Dict[str, Any], score_key: str = "plddt") -> float:
    """Compute a scalar quality score from an OpenFold output dictionary."""
    if score_key not in output_dict:
        keys = ", ".join(sorted(output_dict.keys()))
        raise KeyError(f"Score key '{score_key}' not present in output dict. Keys: {keys}")

    values = np.asarray(output_dict[score_key], dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"Score key '{score_key}' contained no values")

    return float(values.mean())


def select_best_entries(entries: Iterable[OpenFoldRunResult], top_k: int = 1) -> List[OpenFoldRunResult]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    ranked = sorted(entries, key=lambda entry: entry.score, reverse=True)
    return ranked[:top_k]


class OpenFoldZarrArchive:
    """Persist best OpenFold sweep results in a Zarr hierarchy."""

    def __init__(self, archive_path: str | Path):
        self.archive_path = Path(archive_path)
        self.root = zarr.open_group(str(self.archive_path), mode="a")
        self.root.attrs.setdefault("archive_type", "openfold_parameter_sweep")
        self.root.attrs.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())

    def append_best_entries(self, entries: Iterable[OpenFoldRunResult]) -> None:
        best_group = self.root.require_group("best_entries")

        for entry in entries:
            run_group = best_group.require_group(entry.run_id)
            run_group.attrs["score"] = float(entry.score)
            run_group.attrs["params_json"] = json.dumps(entry.params, sort_keys=True)
            run_group.attrs["output_dir"] = entry.output_dir
            run_group.attrs["command"] = entry.command
            run_group.attrs["model_output_path"] = entry.model_output_path
            if entry.changed_param is not None:
                run_group.attrs["changed_param"] = entry.changed_param
            if entry.from_value is not None:
                run_group.attrs["from_value_json"] = json.dumps(entry.from_value)
            if entry.to_value is not None:
                run_group.attrs["to_value_json"] = json.dumps(entry.to_value)
            if entry.score_delta is not None:
                run_group.attrs["score_delta"] = float(entry.score_delta)
            if entry.step_index is not None:
                run_group.attrs["step_index"] = int(entry.step_index)
            run_group.attrs["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
            self._archive_run_artifacts(run_group, entry)

    def write_best_log(self, entries: Iterable[OpenFoldRunResult], log_path: str | Path) -> None:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as handle:
            for entry in entries:
                record = {
                    "run_id": entry.run_id,
                    "score": float(entry.score),
                    "params": entry.params,
                    "output_dir": entry.output_dir,
                    "command": entry.command,
                    "model_output_path": entry.model_output_path,
                }
                if entry.changed_param is not None:
                    record["changed_param"] = entry.changed_param
                if entry.from_value is not None:
                    record["from_value"] = entry.from_value
                if entry.to_value is not None:
                    record["to_value"] = entry.to_value
                if entry.score_delta is not None:
                    record["score_delta"] = float(entry.score_delta)
                if entry.step_index is not None:
                    record["step_index"] = int(entry.step_index)
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _sanitize_component(name: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
        safe = safe.strip("._")
        return safe or "item"

    @staticmethod
    def _extract_flag_value(command: str, flag: str) -> str | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None

        value: str | None = None
        for idx, token in enumerate(tokens):
            if token == flag and idx + 1 < len(tokens):
                value = tokens[idx + 1]
            elif token.startswith(f"{flag}="):
                value = token.split("=", 1)[1]

        return value

    @staticmethod
    def _as_numpy_array(value: Any) -> np.ndarray | None:
        if isinstance(value, np.ndarray):
            return value

        if isinstance(value, (int, float, bool, np.number)):
            return np.asarray(value)

        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            return np.asarray(value.detach().cpu().numpy())

        if isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
            except Exception:
                return None
            if arr.dtype == object:
                return None
            return arr

        return None

    def _write_array_dataset(self, parent_group: Any, name: str, array: np.ndarray) -> None:
        key = self._sanitize_component(name)
        if key in parent_group:
            del parent_group[key]
        parent_group.create_dataset(
            key,
            data=array,
            shape=array.shape,
            dtype=array.dtype,
            overwrite=True,
        )

    def _archive_output_dict_arrays(self, group: Any, obj: Any, depth: int = 0) -> None:
        if depth > 8:
            return

        if isinstance(obj, dict):
            for key, value in sorted(obj.items(), key=lambda item: str(item[0])):
                child_name = self._sanitize_component(str(key))
                child_group = group.require_group(child_name)
                self._archive_output_dict_arrays(child_group, value, depth + 1)
            return

        if isinstance(obj, (list, tuple)) and obj and any(isinstance(x, (dict, list, tuple)) for x in obj):
            for idx, item in enumerate(obj):
                child_group = group.require_group(f"idx_{idx}")
                self._archive_output_dict_arrays(child_group, item, depth + 1)
            return

        array = self._as_numpy_array(obj)
        if array is None or array.dtype == object:
            return

        self._write_array_dataset(group, "values", np.asarray(array))

    def _archive_file_bytes(self, files_group: Any, file_path: Path, relative_path: Path) -> None:
        # Preserve directory structure to avoid collisions for files sharing a basename.
        safe_parts = [self._sanitize_component(part) for part in relative_path.parts]
        file_group = files_group
        for part in safe_parts:
            file_group = file_group.require_group(part)
        if "bytes" in file_group:
            del file_group["bytes"]

        payload = np.frombuffer(file_path.read_bytes(), dtype=np.uint8)
        file_group.create_dataset(
            "bytes",
            data=payload,
            shape=payload.shape,
            dtype=payload.dtype,
            overwrite=True,
        )
        file_group.attrs["source_path"] = str(file_path)
        file_group.attrs["relative_path"] = str(relative_path)
        file_group.attrs["size_bytes"] = int(file_path.stat().st_size)

    def _archive_run_artifacts(self, run_group: Any, entry: OpenFoldRunResult) -> None:
        artifacts_group = run_group.require_group("artifacts")

        output_dict_path = Path(entry.model_output_path)
        if output_dict_path.exists():
            try:
                with output_dict_path.open("rb") as handle:
                    output_dict = pickle.load(handle)

                activations_group = artifacts_group.require_group("layer_wise_activations")
                self._archive_output_dict_arrays(activations_group, output_dict)
            except Exception as exc:
                artifacts_group.attrs["layer_wise_activations_error"] = str(exc)

        attention_dir = self._extract_flag_value(entry.command, "--attn_map_dir")
        attention_group = artifacts_group.require_group("attention_maps")
        if attention_dir:
            attn_path = Path(attention_dir)
            attention_group.attrs["attention_dir"] = str(attn_path)
            if attn_path.exists() and attn_path.is_dir():
                files_group = attention_group.require_group("files")
                for file_path in sorted(attn_path.rglob("*")):
                    if file_path.is_file():
                        self._archive_file_bytes(
                            files_group,
                            file_path,
                            file_path.relative_to(attn_path),
                        )

        structure_group = artifacts_group.require_group("structural_outputs")
        run_output_dir = Path(entry.output_dir)
        structure_group.attrs["run_output_dir"] = str(run_output_dir)
        if run_output_dir.exists() and run_output_dir.is_dir():
            files_group = structure_group.require_group("files")
            for file_path in sorted(run_output_dir.rglob("*")):
                if file_path.is_file() and file_path.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
                    self._archive_file_bytes(
                        files_group,
                        file_path,
                        file_path.relative_to(run_output_dir),
                    )

        metadata_group = artifacts_group.require_group("metadata")
        metadata_group.attrs["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_group.attrs["run_id"] = entry.run_id
        metadata_group.attrs["score"] = float(entry.score)
        metadata_group.attrs["params_json"] = json.dumps(entry.params, sort_keys=True)
        metadata_group.attrs["command"] = entry.command

        model_version = self._extract_flag_value(entry.command, "--config_preset")
        checkpoint_path = self._extract_flag_value(entry.command, "--openfold_checkpoint_path")
        if model_version is not None:
            metadata_group.attrs["model_version"] = model_version
        if checkpoint_path is not None:
            metadata_group.attrs["checkpoint_path"] = checkpoint_path
