"""End-to-end correctness for the EvoformerRunArtifact -> Figure bridge.

These tests build a synthetic on-disk artifact (one attention text file in
the format Pranav's hooks emit, plus a reps.pt) and verify that the four
``*_from_artifact`` helpers in ``viz/integrations.py``:

  * pull the right tensor out of ``artifact.get_attention_matrix`` /
    ``artifact.reps``,
  * route it through Priyavi's ``prepare_*`` (when applicable), and
  * paint exactly that data onto the resulting Figure.

Run from the repo root with:

    python -m unittest tests.test_viz_artifact_bridge -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

from openfold.utils.evoformer_run_artifact import EvoformerRunArtifact  # noqa: E402
from representation_tensor_utils import (  # noqa: E402
    prepare_heatmap_data,
    prepare_lineplot_data,
)
from viz.integrations import (  # noqa: E402
    attention_heatmap_from_artifact,
    figure_from_artifact,
    representation_heatmap_from_artifact,
    representation_line_from_artifact,
    representation_tensor_from_artifact,
)


def _heatmap_data(fig: Figure) -> np.ndarray:
    images = fig.axes[0].get_images()
    assert len(images) == 1, f"expected 1 image, got {len(images)}"
    arr = images[0].get_array()
    return arr.filled() if hasattr(arr, "filled") else np.asarray(arr)


def _line_y(fig: Figure, idx: int = 0) -> np.ndarray:
    return fig.axes[0].get_lines()[idx].get_ydata()


def _write_attention_file(path: Path, layer: int, n_heads: int, n_res: int, rng) -> None:
    """Write a top-K attention text file in Pranav's format."""
    lines = []
    K = min(8, n_res)
    for h in range(n_heads):
        lines.append(f"Layer {layer}, Head {h}")
        for i in range(n_res):
            cols = rng.choice(n_res, size=K, replace=False)
            scores = rng.random(K).astype(np.float32)
            for j, s in zip(cols, scores):
                lines.append(f"{i} {int(j)} {float(s):.6f}")
    path.write_text("\n".join(lines) + "\n")


@unittest.skipUnless(_HAS_TORCH, "torch is required for the artifact bridge tests")
class TestArtifactBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        root = Path(cls.tmpdir.name)
        attn_dir = root / "attention"
        attn_dir.mkdir()

        rng = np.random.default_rng(7)
        cls.N_RES = 16
        cls.N_HEADS = 3
        cls.LAYER = 0
        cls.C_Z = 5
        cls.C_M = 4
        cls.N_SEQ = 6

        _write_attention_file(
            attn_dir / f"msa_row_attn_layer{cls.LAYER}.txt",
            layer=cls.LAYER,
            n_heads=cls.N_HEADS,
            n_res=cls.N_RES,
            rng=rng,
        )

        cls.pair_tensor = rng.standard_normal(
            (cls.N_RES, cls.N_RES, cls.C_Z)
        ).astype(np.float32)
        cls.msa_tensor = rng.standard_normal(
            (cls.N_SEQ, cls.N_RES, cls.C_M)
        ).astype(np.float32)
        reps = {
            f"layer_{cls.LAYER:02d}.pair": torch.from_numpy(cls.pair_tensor),
            f"layer_{cls.LAYER:02d}.msa": torch.from_numpy(cls.msa_tensor),
            "pair": torch.from_numpy(cls.pair_tensor),
            "msa": torch.from_numpy(cls.msa_tensor),
        }
        cls.reps_path = root / "reps.pt"
        torch.save(reps, cls.reps_path)

        cls.artifact = EvoformerRunArtifact(
            run_dir=root,
            attention_dir=attn_dir,
            reps_path=cls.reps_path,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def tearDown(self):
        plt.close("all")

    def test_attention_heatmap_single_head_matches_artifact(self):
        for h in range(self.N_HEADS):
            with self.subTest(head=h):
                expected = self.artifact.get_attention_matrix(
                    "msa_row_attn", layer=self.LAYER, head=h
                )
                fig = attention_heatmap_from_artifact(
                    self.artifact, "msa_row_attn", layer=self.LAYER, head=h
                )
                np.testing.assert_allclose(_heatmap_data(fig), expected)

    def test_attention_heatmap_mean_over_heads(self):
        expected = self.artifact.get_attention_matrix(
            "msa_row_attn", layer=self.LAYER, mean_across_heads=True
        )
        fig = attention_heatmap_from_artifact(
            self.artifact, "msa_row_attn", layer=self.LAYER, mean_across_heads=True
        )
        np.testing.assert_allclose(_heatmap_data(fig), expected)

    def test_representation_tensor_layer_keyed_and_fallback(self):
        layer_keyed = representation_tensor_from_artifact(
            self.artifact, "pair", layer=self.LAYER
        )
        np.testing.assert_allclose(layer_keyed, self.pair_tensor.astype(np.float64))

        no_layer = representation_tensor_from_artifact(self.artifact, "pair")
        np.testing.assert_allclose(no_layer, self.pair_tensor.astype(np.float64))

    def test_representation_tensor_missing_key_raises(self):
        with self.assertRaises(KeyError):
            representation_tensor_from_artifact(
                self.artifact, "single", layer=self.LAYER
            )

    def test_pair_heatmap_from_artifact_matches_prepare(self):
        for ch in (0, 2, self.C_Z - 1):
            with self.subTest(ch=ch):
                expected = prepare_heatmap_data(
                    self.pair_tensor.astype(np.float64),
                    "pair",
                    channel=ch,
                    normalize="minmax",
                )
                fig = representation_heatmap_from_artifact(
                    self.artifact,
                    "pair",
                    layer=self.LAYER,
                    channel=ch,
                    normalize="minmax",
                )
                np.testing.assert_allclose(_heatmap_data(fig), expected)

    def test_pair_heatmap_aggregate_from_artifact(self):
        expected = prepare_heatmap_data(
            self.pair_tensor.astype(np.float64),
            "pair",
            aggregate="mean",
            normalize="zscore",
        )
        fig = representation_heatmap_from_artifact(
            self.artifact,
            "pair",
            layer=self.LAYER,
            aggregate="mean",
            normalize="zscore",
        )
        np.testing.assert_allclose(_heatmap_data(fig), expected)

    def test_msa_line_from_artifact_matches_prepare(self):
        for ch in (0, 1, self.C_M - 1):
            with self.subTest(ch=ch):
                _, expected = prepare_lineplot_data(
                    self.msa_tensor.astype(np.float64), "msa", channel=ch
                )
                fig = representation_line_from_artifact(
                    self.artifact, "msa", channel=ch, layer=self.LAYER
                )
                np.testing.assert_allclose(_line_y(fig), expected)

    def test_figure_from_artifact_dispatch_attention(self):
        fig = figure_from_artifact(
            self.artifact,
            attn_kind="msa_row_attn",
            layer=self.LAYER,
            mean_across_heads=True,
        )
        expected = self.artifact.get_attention_matrix(
            "msa_row_attn", layer=self.LAYER, mean_across_heads=True
        )
        np.testing.assert_allclose(_heatmap_data(fig), expected)

    def test_figure_from_artifact_dispatch_rep_heatmap(self):
        fig = figure_from_artifact(
            self.artifact, rep_kind="pair", layer=self.LAYER, channel=0
        )
        expected = prepare_heatmap_data(
            self.pair_tensor.astype(np.float64),
            "pair",
            channel=0,
            normalize="minmax",
        )
        np.testing.assert_allclose(_heatmap_data(fig), expected)

    def test_figure_from_artifact_dispatch_rep_line(self):
        fig = figure_from_artifact(
            self.artifact,
            rep_kind="msa",
            layer=self.LAYER,
            channel=1,
            plot="line",
        )
        _, expected = prepare_lineplot_data(
            self.msa_tensor.astype(np.float64), "msa", channel=1
        )
        np.testing.assert_allclose(_line_y(fig), expected)

    def test_figure_from_artifact_requires_exactly_one_kind(self):
        with self.assertRaises(ValueError):
            figure_from_artifact(self.artifact)
        with self.assertRaises(ValueError):
            figure_from_artifact(
                self.artifact, attn_kind="msa_row_attn", rep_kind="pair", layer=0
            )

    def test_figure_from_artifact_attention_requires_layer(self):
        with self.assertRaises(ValueError):
            figure_from_artifact(self.artifact, attn_kind="msa_row_attn")

    def test_figure_from_artifact_line_requires_channel(self):
        with self.assertRaises(ValueError):
            figure_from_artifact(
                self.artifact, rep_kind="single", plot="line", layer=self.LAYER
            )


if __name__ == "__main__":
    unittest.main()
