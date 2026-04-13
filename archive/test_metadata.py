import numpy as np
import zarr
from vizfold_to_zarr import open_archive, store_metadata

archive_path = "run.vizfold.zarr"

# ── Step A: create the archive with the correct group structure ──────────────
root = open_archive(archive_path, overwrite=True)
print("Groups created:", sorted(root.group_keys()))

# ── Step B: write metadata (replace with real vizfold output later) ──────────
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
num_residues = len(sequence)
num_recycles = 3

store_metadata(
    archive_path=archive_path,
    model_version="openfold-v2.2.0",
    config_version="model_1_ptm",
    sequence=sequence,
    num_residues=num_residues,
    num_recycles=num_recycles,
    recycle_info=np.array([0.92, 0.95, 0.96], dtype=np.float32),
    residue_index=np.arange(num_residues, dtype=np.int64),
    representation_names=np.array(
        [f"evoformer_layer_{i:02d}" for i in range(48)]
    ),
)
print("Metadata written.")

# ── Step C: read back and verify ─────────────────────────────────────────────
root = zarr.open_group(archive_path, mode="r")
meta = root["metadata"]

print("\n--- metadata ---")
print("model_version    :", meta["model_version"][()])
print("config_version   :", meta["config_version"][()])
print("sequence         :", meta["sequence"][()])
print("num_residues     :", meta["num_residues"][()])
print("num_recycles     :", meta["num_recycles"][()])
print("recycle_info     :", meta["recycle_info"][:])
print("residue_index[:5]:", meta["residue_index"][:5])
print("repr_names[:3]   :", meta["representation_names"][:3])

print("\n--- top-level groups ---")
for name in sorted(root.group_keys()):
    print(" ", name)

print("\n--- representations sub-groups ---")
for name in sorted(root["representations"].group_keys()):
    print(" ", name)

print("\n--- attention sub-groups ---")
for name in sorted(root["attention"].group_keys()):
    print(" ", name)