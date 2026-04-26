from __future__ import annotations

import numpy as np
import pytest

from vizfold.offline import ArchiveReader

zarr = pytest.importorskip("zarr")


def test_archive_reader_loads_issue39_style_zarr(tmp_path):
    archive_path = tmp_path / "toy.vizfold.zarr"

    root = zarr.open_group(str(archive_path), mode="w")

    metadata = root.require_group("metadata")
    metadata.attrs["model_version"] = "openfold-test"
    metadata.attrs["config_version"] = "model_1"
    metadata.attrs["sequence"] = "ACDE"
    metadata.attrs["num_residues"] = 4
    metadata.attrs["num_recycles"] = 1
    metadata.create_dataset(
        "residue_index",
        data=np.arange(4),
        shape=(4,),
    )

    attention = root.require_group("attention").require_group("triangle_start")

    # Issue-39 documented shape for triangle attention:
    # (num_residues, num_residues, num_heads)
    arr = np.zeros((4, 4, 2), dtype=np.float32)
    arr[0, 2, 1] = 0.90
    arr[3, 1, 1] = 0.40
    arr[1, 0, 0] = 0.75

    attention.create_dataset(
        "layer_00",
        data=arr,
        shape=arr.shape,
        chunks=(4, 4, 1),
    )

    reps = root.require_group("representations")
    single_arr = np.ones((4, 8), dtype=np.float32)
    reps.require_group("single").create_dataset(
        "layer_00",
        data=single_arr,
        shape=single_arr.shape,
    )
    pair_arr = np.ones((4, 4, 16), dtype=np.float32)
    reps.require_group("pair").create_dataset(
        "layer_00",
        data=pair_arr,
        shape=pair_arr.shape,
    )

    structure = root.require_group("structure")
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    structure.create_dataset(
        "atom_positions",
        data=coords,
        shape=coords.shape,
    )

    reader = ArchiveReader(archive_path)

    meta = reader.metadata()
    assert meta.archive_kind == "zarr"
    assert meta.model_version == "openfold-test"
    assert meta.sequence == "ACDE"
    assert meta.structure_available is True

    assert reader.list_attention_types() == ["triangle_start"]
    assert reader.list_layers("triangle_start") == [0]
    assert reader.list_heads("triangle_start", 0) == [0, 1]

    loaded = reader.load_attention(
        attention_type="triangle_start",
        layer=0,
        head=1,
        top_k=2,
    )

    assert loaded.attention_type == "triangle_start"
    assert loaded.layer == 0
    assert loaded.head == 1
    assert loaded.as_triplets()[0] == (0, 2, pytest.approx(0.90))
    assert loaded.as_triplets()[1] == (3, 1, pytest.approx(0.40))

    all_heads = reader.load_attention_heads("triangle_start", 0, top_k=1)
    assert sorted(all_heads.keys()) == [0, 1]

    single = reader.load_single_representation(0)
    pair = reader.load_pair_representation(0)

    assert single.shape == (4, 8)
    assert pair.shape == (4, 4, 16)

    structure_data = reader.load_structure()
    assert structure_data.sequence == "ACDE"
    assert structure_data.pdb_text is not None
    assert "ATOM" in structure_data.pdb_text

def test_archive_reader_missing_structure_does_not_crash(tmp_path):
    archive_path = tmp_path / "no_structure.zarr"
    root = zarr.open_group(str(archive_path), mode="w")

    metadata = root.require_group("metadata")
    metadata.attrs["sequence"] = "ACDE"
    metadata.attrs["model_version"] = "openfold-test"

    attention = root.require_group("attention").require_group("triangle_start")

    arr = np.ones((4, 4, 1), dtype=np.float32)
    attention.create_dataset(
        "layer_00",
        data=arr,
        shape=arr.shape,
        chunks=(4, 4, 1),
    )

    reader = ArchiveReader(archive_path)

    meta = reader.metadata()
    assert meta.structure_available is False
    assert "attention" in meta.capabilities
    assert "structure" not in meta.capabilities

    structure = reader.load_structure()
    assert structure.sequence == "ACDE"
    assert structure.pdb_text is None


def test_archive_reader_rejects_bad_head(tmp_path):
    archive_path = tmp_path / "bad_head.zarr"
    root = zarr.open_group(str(archive_path), mode="w")

    metadata = root.require_group("metadata")
    metadata.attrs["sequence"] = "ACDE"

    attention = root.require_group("attention").require_group("triangle_start")

    arr = np.ones((4, 4, 2), dtype=np.float32)
    attention.create_dataset(
        "layer_00",
        data=arr,
        shape=arr.shape,
        chunks=(4, 4, 1),
    )

    reader = ArchiveReader(archive_path)

    assert reader.list_heads("triangle_start", 0) == [0, 1]

    with pytest.raises(IndexError):
        reader.load_attention("triangle_start", layer=0, head=5)


def test_archive_reader_handles_residue_specific_4d_attention(tmp_path):
    archive_path = tmp_path / "residue_attention.zarr"
    root = zarr.open_group(str(archive_path), mode="w")

    metadata = root.require_group("metadata")
    metadata.attrs["sequence"] = "ACDE"

    attention = root.require_group("attention").require_group("triangle_start")

    # Shape: (heads, residue_index, src_residue, dst_residue)
    arr = np.zeros((2, 4, 4, 4), dtype=np.float32)
    arr[1, 2, 0, 3] = 0.95
    arr[1, 2, 1, 1] = 0.50
    arr[0, 0, 2, 2] = 0.80

    attention.create_dataset(
        "layer_00",
        data=arr,
        shape=arr.shape,
        chunks=(1, 1, 4, 4),
    )

    reader = ArchiveReader(archive_path)

    assert reader.list_attention_types() == ["triangle_start"]
    assert reader.list_layers("triangle_start") == [0]
    assert reader.list_heads("triangle_start", 0) == [0, 1]
    assert reader.list_residue_indices("triangle_start", 0) == [0, 1, 2, 3]

    loaded = reader.load_attention(
        attention_type="triangle_start",
        layer=0,
        head=1,
        residue_idx=2,
        top_k=2,
    )

    assert loaded.residue_idx == 2
    assert loaded.as_triplets()[0] == (0, 3, pytest.approx(0.95))
    assert loaded.as_triplets()[1] == (1, 1, pytest.approx(0.50))
    