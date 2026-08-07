"""
Demo for reading and incrementally updating a VizFold Zarr archive.

Run from repo root:
    python archive/demo.py

Run from the archive directory:
    python demo.py
"""

import sys
from pathlib import Path

import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archive import (  # noqa: E402
    load_attention_head,
    load_pair_representation,
    load_single_representation,
    store_attention,
    store_pair_representation,
    store_single_representation,
    validate_archive,
)


ARCHIVE_PATH = Path(__file__).resolve().parent / "test_1UBQ.zarr"
ARCHIVE = str(ARCHIVE_PATH)
ATTENTION_TYPE = "triangle_start"
SAMPLE_RESIDUE = 10
SAMPLE_TARGET = 20


def layer_names(group):
    return sorted(name for name in group.keys() if name.startswith("layer_"))


def latest_layer_index(group):
    layers = layer_names(group)
    if not layers:
        raise RuntimeError("No layer_XX datasets found in group")
    return int(layers[-1].removeprefix("layer_"))


def describe_array(label, array):
    array = np.asarray(array)
    print(
        f"  {label}: shape={array.shape}, dtype={array.dtype}, "
        f"min={float(array.min()):.6f}, max={float(array.max()):.6f}, "
        f"mean={float(array.mean()):.6f}, std={float(array.std()):.6f}"
    )


def format_vector(values, limit=5):
    values = np.asarray(values).reshape(-1)[:limit]
    return np.array2string(values, precision=4, separator=", ")


def print_archive_contents(root):
    print("\nArchive contents:")
    print(f"  representations/single: {layer_names(root['representations']['single'])}")
    print(f"  representations/pair: {layer_names(root['representations']['pair'])}")
    print(f"  attention/{ATTENTION_TYPE}: {layer_names(root['attention'][ATTENTION_TYPE])}")

    structure = root.get("structure")
    if structure is None:
        print("  structure: missing")
        return

    for name in ("atom_positions", "atom_mask", "ptm"):
        if name in structure:
            value = np.asarray(structure[name])
            preview = value.item() if value.size == 1 else f"shape={value.shape}, dtype={value.dtype}"
            print(f"  structure/{name}: {preview}")


def print_source_examples(single, pair, attention_head, root):
    residue = min(SAMPLE_RESIDUE, single.shape[0] - 1)
    target = min(SAMPLE_TARGET, single.shape[0] - 1)
    top_targets = np.argsort(attention_head[residue])[-5:][::-1]

    print("\nConcrete values from source layer:")
    print(f"  single[{residue}, :5] = {format_vector(single[residue, :5])}")
    print(f"  pair[{residue}, {target}, :5] = {format_vector(pair[residue, target, :5])}")
    print(
        f"  attention head 0 score residue {residue} -> {target}: "
        f"{float(attention_head[residue, target]):.6f}"
    )
    print(f"  top 5 attention targets for residue {residue}: {top_targets.tolist()}")

    structure = root.get("structure")
    if structure is not None and "atom_positions" in structure:
        atom_positions = np.asarray(structure["atom_positions"])
        residue_for_structure = min(residue, atom_positions.shape[0] - 1)
        if atom_positions.ndim == 3:
            ca_atom_index = 1 if atom_positions.shape[1] > 1 else 0
            coords = atom_positions[residue_for_structure, ca_atom_index]
            print(
                f"  CA coordinates for residue {residue_for_structure}: "
                f"{format_vector(coords, limit=3)}"
            )
        else:
            coords = atom_positions[residue_for_structure]
            print(f"  coordinates for residue {residue_for_structure}: {format_vector(coords, limit=3)}")


def print_delta_examples(source_single, source_pair, source_attention_head, new_single, new_pair, new_attention_head):
    residue = min(SAMPLE_RESIDUE, source_single.shape[0] - 1)
    target = min(SAMPLE_TARGET, source_single.shape[0] - 1)

    print("\nBefore/after proof for new layer:")
    print(
        f"  single[{residue}, 0]: "
        f"{float(source_single[residue, 0]):.6f} -> {float(new_single[residue, 0]):.6f}"
    )
    print(
        f"  pair[{residue}, {target}, 0]: "
        f"{float(source_pair[residue, target, 0]):.6f} -> {float(new_pair[residue, target, 0]):.6f}"
    )
    print(
        f"  attention[head=0, {residue}, {target}]: "
        f"{float(source_attention_head[residue, target]):.6f} -> "
        f"{float(new_attention_head[residue, target]):.6f}"
    )
    print(f"  single mean delta: {float((new_single - source_single).mean()):.6f}")
    print(f"  pair mean delta: {float((new_pair - source_pair).mean()):.6f}")
    print(f"  attention head 0 mean delta: {float((new_attention_head - source_attention_head).mean()):.6f}")


if not ARCHIVE_PATH.exists():
    raise FileNotFoundError(
        f"Demo archive not found: {ARCHIVE_PATH}\n"
        "Create it first with: python archive/test_archive.py"
    )

root = zarr.open(ARCHIVE_PATH, mode="r")
single_group = root["representations"]["single"]
pair_group = root["representations"]["pair"]
attention_group = root["attention"][ATTENTION_TYPE]

source_layer = latest_layer_index(single_group)
new_layer = source_layer + 1
source_name = f"layer_{source_layer:02d}"
new_name = f"layer_{new_layer:02d}"

print(f"Using archive: {ARCHIVE_PATH}")
print_archive_contents(root)

# Read existing data through the archive utilities.
single = load_single_representation(ARCHIVE, source_layer)
pair = load_pair_representation(ARCHIVE, source_layer)
attention = np.asarray(attention_group[source_name])
attention_head = load_attention_head(ARCHIVE, ATTENTION_TYPE, source_layer, 0)

print(f"\nSource {source_name} summaries:")
describe_array("single representation", single)
describe_array("pair representation", pair)
describe_array("attention layer", attention)
describe_array("attention head 0", attention_head)
print_source_examples(single, pair, attention_head, root)

# Add one incremental layer. The offset makes it easy to verify this is new data.
offset = np.float32(new_layer * 0.01)
print(f"\nAdding incremental archive paths for {new_name} with +{float(offset):.4f} offset:")
print(f"  representations/single/{new_name}")
print(f"  representations/pair/{new_name}")
print(f"  attention/{ATTENTION_TYPE}/{new_name}")
store_single_representation(ARCHIVE, new_layer, single + offset)
store_pair_representation(ARCHIVE, new_layer, pair + offset)
store_attention(ARCHIVE, ATTENTION_TYPE, new_layer, attention + offset)

# Re-open and read the new data back.
root = zarr.open(ARCHIVE_PATH, mode="r")
new_single = load_single_representation(ARCHIVE, new_layer)
new_pair = load_pair_representation(ARCHIVE, new_layer)
new_attention_head = load_attention_head(ARCHIVE, ATTENTION_TYPE, new_layer, 0)

print(f"\nRead back new {new_name} summaries:")
describe_array("single representation", new_single)
describe_array("pair representation", new_pair)
describe_array("attention head 0", new_attention_head)
print_delta_examples(single, pair, attention_head, new_single, new_pair, new_attention_head)
print_archive_contents(root)

report = validate_archive(ARCHIVE, strict=False)
print("\nValidation report:")
print(f"  valid: {report['valid']}")
print(f"  components_found: {report['components_found']}")
print(f"  warnings: {report['warnings']}")
print(f"  errors: {report['errors']}")
