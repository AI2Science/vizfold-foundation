import runpy
import tempfile
from pathlib import Path
import unittest

import numpy as np


OUTLINE_PATH = Path(__file__).resolve().parents[1] / "Archive Utils" / "outline"


class TestArchiveUtilsIncremental(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = runpy.run_path(str(OUTLINE_PATH))

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.archive_path = str(Path(self.tmpdir.name) / "trace.zarr")

    def test_load_metadata_roundtrip(self):
        store_metadata = self.api["store_metadata"]
        load_metadata = self.api["load_metadata"]

        store_metadata(
            self.archive_path,
            model_version="model-v1",
            config_version="config-v1",
            sequence="ACDE",
            num_residues=4,
            num_recycles=2,
            recycle_info=np.array([0.1, 0.2]),
            residue_index=np.array([5, 6, 7, 8]),
            representation_names=np.array(["single", "pair"]),
        )

        metadata = load_metadata(self.archive_path)
        self.assertEqual(metadata["model_version"], "model-v1")
        self.assertEqual(metadata["config_version"], "config-v1")
        self.assertEqual(metadata["sequence"], "ACDE")
        self.assertEqual(metadata["num_residues"], 4)
        self.assertEqual(metadata["num_recycles"], 2)
        np.testing.assert_array_equal(metadata["recycle_info"], np.array([0.1, 0.2]))
        np.testing.assert_array_equal(metadata["residue_index"], np.array([5, 6, 7, 8]))
        np.testing.assert_array_equal(metadata["representation_names"], np.array(["single", "pair"]))

    def test_load_single_representation_roundtrip(self):
        store_single_representation = self.api["store_single_representation"]
        load_single_representation = self.api["load_single_representation"]

        single = np.arange(12, dtype=np.float32).reshape(3, 4)
        store_single_representation(self.archive_path, 2, single)

        loaded = load_single_representation(self.archive_path, 2)
        np.testing.assert_array_equal(loaded, single)

    def test_load_pair_representation_roundtrip(self):
        store_pair_representation = self.api["store_pair_representation"]
        load_pair_representation = self.api["load_pair_representation"]

        pair = np.arange(48, dtype=np.float32).reshape(4, 4, 3)
        store_pair_representation(self.archive_path, 1, pair)

        loaded = load_pair_representation(self.archive_path, 1)
        np.testing.assert_array_equal(loaded, pair)

    def test_orchestrator_end_to_end(self):
        orchestrator_cls = self.api["ArchiveOrchestrator"]
        load_metadata = self.api["load_metadata"]
        load_single_representation = self.api["load_single_representation"]
        load_pair_representation = self.api["load_pair_representation"]

        orchestrator = orchestrator_cls(self.archive_path)

        orchestrator.add_metadata(
            model_version="model-v2",
            config_version="config-v2",
            sequence="WXYZ",
            num_residues=4,
            num_recycles=1,
            representation_names=np.array(["single", "pair"]),
        )
        orchestrator.add_single_layer(0, np.ones((4, 3), dtype=np.float32))
        orchestrator.add_pair_layer(0, np.ones((4, 4, 2), dtype=np.float32))
        orchestrator.add_attention(
            "triangle_start",
            0,
            np.ones((2, 4, 4), dtype=np.float32),
        )
        orchestrator.add_structure(
            np.arange(12, dtype=np.float32).reshape(4, 3),
            atom_mask=np.array([1, 1, 1, 1], dtype=np.float32),
            ptm=0.91,
        )
        orchestrator.validate(
            validator=lambda path, *args, **kwargs: {
                "valid": True,
                "path": path,
                "warnings": [],
                "errors": [],
            }
        )

        summary = orchestrator.summary()
        self.assertEqual(summary["archive_path"], self.archive_path)
        self.assertEqual(summary["events"][0]["target"], "metadata")
        self.assertEqual(summary["events"][1]["target"], "representations/single/layer_00")
        self.assertEqual(summary["events"][2]["target"], "representations/pair/layer_00")
        self.assertEqual(summary["events"][3]["target"], "attention/triangle_start/layer_00")
        self.assertEqual(summary["events"][4]["target"], "structure")
        self.assertEqual(summary["events"][5]["action"], "validate")
        self.assertTrue(summary["events"][5]["result"]["valid"])

        metadata = load_metadata(self.archive_path)
        np.testing.assert_array_equal(metadata["representation_names"], np.array(["single", "pair"]))
        np.testing.assert_array_equal(
            load_single_representation(self.archive_path, 0),
            np.ones((4, 3), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            load_pair_representation(self.archive_path, 0),
            np.ones((4, 4, 2), dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
