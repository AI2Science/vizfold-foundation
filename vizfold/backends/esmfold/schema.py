"""
Trace schema and metadata helpers for ESMFold outputs.

Ensures VizFold-compatible archive format and publication-grade metadata.
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _git_head(repo_path: Optional[str] = None) -> Optional[str]:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        if repo_path:
            cmd = ["git", "-C", repo_path, "rev-parse", "HEAD"]
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return None


def _read_fasta_and_hash(path: str) -> Tuple[str, str]:
    with open(path) as f:
        raw = f.read()
    lines = [line.strip() for line in raw.splitlines() if line.strip() and not line.startswith(">")]
    seq = "".join(lines)
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return seq, h


def build_meta(
    *,
    backend: str = "esmfold",
    model_name: str,
    out_dir: str,
    fasta_path: Optional[str] = None,
    device: str,
    dtype: str,
    sequence_length: int,
    fasta_hash: str,
    layer_count: int,
    head_count: int,
    trace_mode: str,
    trace_formats: List[str],
    shapes_recorded: Dict[str, Any],
    seed: Optional[int] = None,
    deterministic: bool = False,
    save_fp16: bool = False,
    top_k: int = 50,
    repo_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build meta.json content for a VizFold-compatible run.

    shapes_recorded: e.g. {"attention": {"layer_000": [num_heads, N, N]}, "activations": {...}}
    """
    meta: Dict[str, Any] = {
        "backend": backend,
        "model_name": model_name,
        "date_time": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "dtype": dtype,
        "sequence_length": sequence_length,
        "input_fasta_hash": fasta_hash,
        "layer_count": layer_count,
        "head_count": head_count,
        "trace_mode": trace_mode,
        "trace_formats": trace_formats,
        "tensor_format": "fp16" if save_fp16 else "fp32",
        "top_k": top_k,
        "shapes_recorded": shapes_recorded,
    }
    if fasta_path:
        meta["input_fasta_path"] = fasta_path
    if seed is not None:
        meta["seed"] = seed
    if deterministic:
        meta["deterministic"] = True
    commit = _git_head(repo_path)
    if commit:
        meta["repo_commit"] = commit
    return meta


def write_meta(meta: Dict[str, Any], out_dir: str) -> str:
    path = os.path.join(out_dir, "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def write_trace_index(
    out_dir: str,
    attention: Dict[str, Any],
    activations: Dict[str, Any],
) -> str:
    """
    Write trace/index.json mapping layers/heads to file paths and shapes.
    """
    index = {
        "attention": attention,
        "activations": activations,
    }
    path = os.path.join(out_dir, "trace", "index.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)
    return path
