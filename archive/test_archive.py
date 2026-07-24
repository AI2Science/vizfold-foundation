"""
Test script for VizFold Archive Utilities using real OpenFold output data.
"""
import sys
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

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
    store_structure_coordinates,
    tensor_to_numpy,
    validate_archive,
)

# Path to a test pickle file
PICKLE_FILE = "./1UBQ_result_model_1_ptm.pickle"
TEST_ARCHIVE = "test_1UBQ.zarr"


def get_pickle_path():
    return os.path.join(os.path.dirname(__file__), PICKLE_FILE)


def load_pickle_data(pkl_path):
    """Load data from OpenFold output pickle."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def data():
    """Load real OpenFold output data for pytest runs."""
    pkl_path = get_pickle_path()
    if not os.path.exists(pkl_path):
        pytest.skip(f"Pickle file not found: {pkl_path}")
    return load_pickle_data(pkl_path)


@pytest.fixture
def archive_path(tmp_path):
    """Provide an isolated archive path for each pytest test."""
    return str(tmp_path / TEST_ARCHIVE)


def populate_archive(data, archive_path):
    """Populate all components required by validate_archive()."""
    single = data["representations"]["single"]
    pair = data["representations"]["pair"]
    sm = data["structure_module"]
    num_residues = single.shape[0]
    fake_attention = np.random.rand(8, num_residues, num_residues).astype(np.float32)

    store_single_representation(archive_path, layer_index=0, single_array=single, overwrite=True)
    store_pair_representation(archive_path, layer_index=0, pair_array=pair, overwrite=True)
    store_structure_coordinates(
        archive_path,
        sm["final_atom_positions"],
        atom_mask=sm["final_atom_mask"],
        ptm=float(data["ptm"]),
        overwrite=True,
    )
    store_attention(
        archive_path,
        attention_type="triangle_start",
        layer_index=0,
        attention_array=fake_attention,
        overwrite=True,
    )


def test_store_and_load_single_representation(data, archive_path):
    """Test storing and loading single representation."""
    print("\n=== Testing single representation ===")
    
    single = data["representations"]["single"]
    print(f"Original shape: {single.shape}, dtype: {single.dtype}")
    
    # Store it (treating as layer 0)
    store_single_representation(archive_path, layer_index=0, single_array=single, overwrite=True)
    print("Stored to archive")
    
    # Load it back
    loaded = load_single_representation(archive_path, layer_index=0)
    print(f"Loaded shape: {loaded.shape}, dtype: {loaded.dtype}")
    
    # Verify
    if np.allclose(single, loaded, rtol=1e-3):
        print("✓ Single representation: PASS")
        return

    print("✗ Single representation: FAIL - data mismatch")
    assert False, "single representation data mismatch"


def test_store_and_load_pair_representation(data, archive_path):
    """Test storing and loading pair representation."""
    print("\n=== Testing pair representation ===")
    
    pair = data["representations"]["pair"]
    print(f"Original shape: {pair.shape}, dtype: {pair.dtype}")
    
    # Store it (treating as layer 0)
    store_pair_representation(archive_path, layer_index=0, pair_array=pair, overwrite=True)
    print("Stored to archive")
    
    # Load it back
    loaded = load_pair_representation(archive_path, layer_index=0)
    print(f"Loaded shape: {loaded.shape}, dtype: {loaded.dtype}")
    
    # Verify
    if np.allclose(pair, loaded, rtol=1e-3):
        print("✓ Pair representation: PASS")
        return

    print("✗ Pair representation: FAIL - data mismatch")
    assert False, "pair representation data mismatch"


def test_store_structure(data, archive_path):
    """Test storing structure data."""
    print("\n=== Testing structure storage ===")
    
    sm = data["structure_module"]
    atom_positions = sm["final_atom_positions"]
    atom_mask = sm["final_atom_mask"]
    ptm = float(data["ptm"])
    
    print(f"atom_positions shape: {atom_positions.shape}")
    print(f"atom_mask shape: {atom_mask.shape}")
    print(f"ptm: {ptm}")
    
    # Store structure
    store_structure_coordinates(archive_path, atom_positions, atom_mask=atom_mask, ptm=ptm, overwrite=True)
    print("Stored to archive")
    
    # Verify by opening archive
    import zarr
    root = zarr.open(archive_path, mode="r")
    
    if "structure" in root:
        struct = root["structure"]
        if "atom_positions" in struct:
            loaded_pos = np.asarray(struct["atom_positions"])
            print(f"Loaded atom_positions shape: {loaded_pos.shape}")
            if np.allclose(atom_positions, loaded_pos, rtol=1e-3):
                print("✓ Structure storage: PASS")
                return
    
    print("✗ Structure storage: FAIL")
    assert False, "structure atom_positions data mismatch or missing dataset"


def test_store_attention(data, archive_path):
    """Test storing attention data (synthetic since not in pickle)."""
    print("\n=== Testing attention storage ===")
    
    # Create synthetic attention data based on sequence length
    num_residues = data["representations"]["single"].shape[0]
    num_heads = 8
    
    # Shape: (num_heads, num_residues, num_residues)
    fake_attention = np.random.rand(num_heads, num_residues, num_residues).astype(np.float32)
    print(f"Synthetic attention shape: {fake_attention.shape}")
    
    # Store it
    store_attention(archive_path, attention_type="triangle_start", layer_index=0, 
                   attention_array=fake_attention, overwrite=True)
    print("Stored to archive")
    
    # Load single head
    loaded_head = load_attention_head(archive_path, attention_type="triangle_start", 
                                      layer_index=0, head_index=3)
    print(f"Loaded head shape: {loaded_head.shape}")
    
    # Verify
    if np.allclose(fake_attention[3], loaded_head, rtol=1e-5):
        print("✓ Attention storage: PASS")
        return

    print("✗ Attention storage: FAIL")
    assert False, "attention head data mismatch"


def test_validate_archive(data, archive_path):
    """Test archive validation."""
    print("\n=== Testing archive validation ===")
    populate_archive(data, archive_path)
    
    report = validate_archive(archive_path, strict=False)
    print(f"Valid: {report['valid']}")
    print(f"Components found: {report['components_found']}")
    
    if report["errors"]:
        print(f"Errors: {report['errors']}")
    if report["warnings"]:
        print(f"Warnings: {report['warnings']}")
    
    if report["valid"]:
        print("✓ Archive validation: PASS")
        return

    print("✗ Archive validation: FAIL")
    assert False, f"archive validation failed: {report['errors']}"


def run_script_test(func, *args):
    try:
        func(*args)
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        return False
    return True


def main():
    print("=" * 60)
    print("VizFold Archive Utilities - Test with Real Data")
    print("=" * 60)
    
    # Check pickle file exists
    pkl_path = get_pickle_path()
    if not os.path.exists(pkl_path):
        print(f"ERROR: Pickle file not found: {pkl_path}")
        return 1
    
    print(f"\nLoading data from: {pkl_path}")
    data = load_pickle_data(pkl_path)
    
    archive_path = os.path.join(os.path.dirname(__file__), TEST_ARCHIVE)
    print(f"Test archive: {archive_path}")
    
    # Clean up old test archive
    if os.path.exists(archive_path):
        shutil.rmtree(archive_path)
    
    # Run tests
    results = []
    results.append(run_script_test(test_store_and_load_single_representation, data, archive_path))
    results.append(run_script_test(test_store_and_load_pair_representation, data, archive_path))
    results.append(run_script_test(test_store_structure, data, archive_path))
    results.append(run_script_test(test_store_attention, data, archive_path))
    results.append(run_script_test(test_validate_archive, data, archive_path))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
