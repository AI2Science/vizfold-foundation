from __future__ import annotations

import re
from pathlib import Path

from .exceptions import TraceNotFoundError, UnsupportedAttentionTypeError

_MSA_ROW_RE = re.compile(r"^msa_row_attn_layer(?P<layer>\d+)\.txt$")
_TRIANGLE_START_RE = re.compile(
    r"^triangle_start_attn_layer(?P<layer>\d+)_residue_idx_(?P<residue>\d+)\.txt$"
)


def legacy_msa_row_path(attention_dir: str | Path, layer: int) -> Path:
    return Path(attention_dir) / f"msa_row_attn_layer{layer}.txt"


def legacy_triangle_start_path(
    attention_dir: str | Path,
    layer: int,
    residue_idx: int,
) -> Path:
    return Path(attention_dir) / f"triangle_start_attn_layer{layer}_residue_idx_{residue_idx}.txt"


def resolve_legacy_attention_path(
    attention_dir: str | Path,
    attention_type: str,
    layer: int,
    residue_idx: int | None = None,
) -> Path:
    if attention_type == "msa_row":
        path = legacy_msa_row_path(attention_dir, layer)
    elif attention_type == "triangle_start":
        if residue_idx is None:
            raise ValueError("residue_idx is required for triangle_start attention")
        path = legacy_triangle_start_path(attention_dir, layer, residue_idx)
    else:
        raise UnsupportedAttentionTypeError(f"Unsupported attention_type: {attention_type}")

    if not path.exists():
        raise TraceNotFoundError(f"Attention file not found: {path}")

    return path


def parse_legacy_attention_filename(filename: str) -> tuple[str, int, int | None] | None:
    """
    Returns:
        (attention_type, layer, residue_idx)

    Examples:
        msa_row_attn_layer47.txt -> ("msa_row", 47, None)
        triangle_start_attn_layer47_residue_idx_18.txt -> ("triangle_start", 47, 18)
    """
    msa_match = _MSA_ROW_RE.match(filename)
    if msa_match:
        return ("msa_row", int(msa_match.group("layer")), None)

    tri_match = _TRIANGLE_START_RE.match(filename)
    if tri_match:
        return (
            "triangle_start",
            int(tri_match.group("layer")),
            int(tri_match.group("residue")),
        )

    return None