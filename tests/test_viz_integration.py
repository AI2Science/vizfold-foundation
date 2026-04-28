"""End-to-end correctness for viz/integrations.py.

The bridge functions (``heatmap_from_representation`` etc.) compose
``representation_tensor_utils.prepare_*`` with the residue-indexed plot
functions in ``viz``. These tests assert that, for every supported
combination of ``kind`` / ``channel`` / ``aggregate`` / ``normalize``, the
data the bridge actually paints onto the Figure is bit-identical to the
NumPy array Priyavi's processing layer returns.

Run from the repo root with:

    python -m unittest tests.test_viz_integration -v
"""

from __future__ import annotations

import os
import sys
import unittest

import matplotlib
matplotlib.use("Agg")  # no display in CI / smoke runs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from representation_tensor_utils import (
    prepare_heatmap_data,
    prepare_lineplot_data,
)
from viz.integrations import (
    heatmap_from_representation,
    line_from_representation,
    lines_from_representation,
    pair_channel_grid,
)


def _heatmap_data(fig: Figure) -> np.ndarray:
    """Pull the AxesImage data back out of the Figure (first axis)."""
    images = fig.axes[0].get_images()
    assert len(images) == 1, f"expected 1 image on axis, got {len(images)}"
    arr = images[0].get_array()
    return arr.filled() if hasattr(arr, "filled") else np.asarray(arr)


def _line_y(fig: Figure, idx: int = 0) -> np.ndarray:
    return fig.axes[0].get_lines()[idx].get_ydata()


class TestVizIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.N_RES = 24
        cls.C_Z = 5
        cls.N_SEQ = 7
        cls.C_M = 5
        cls.C_S = 6
        cls.z = rng.standard_normal((1, 1, cls.N_RES, cls.N_RES, cls.C_Z)).astype(
            np.float64
        )
        cls.m = rng.standard_normal((cls.N_SEQ, cls.N_RES, cls.C_M)).astype(np.float64)
        cls.s = rng.standard_normal((cls.N_RES, cls.C_S)).astype(np.float64)

    def tearDown(self):
        plt.close("all")

    def test_pair_heatmap_channels_all_norms(self):
        for norm in ("minmax", "zscore", "none"):
            for ch in (0, 2, self.C_Z - 1):
                with self.subTest(norm=norm, ch=ch):
                    expected = prepare_heatmap_data(
                        self.z, "pair", channel=ch, normalize=norm
                    )
                    fig = heatmap_from_representation(
                        self.z, "pair", channel=ch, normalize=norm
                    )
                    np.testing.assert_allclose(
                        _heatmap_data(fig), expected, atol=0, equal_nan=True
                    )

    def test_pair_heatmap_aggregations(self):
        for agg in ("mean", "max", "l2"):
            with self.subTest(agg=agg):
                expected = prepare_heatmap_data(
                    self.z, "pair", aggregate=agg, normalize="minmax"
                )
                fig = heatmap_from_representation(
                    self.z, "pair", aggregate=agg, normalize="minmax"
                )
                np.testing.assert_allclose(
                    _heatmap_data(fig), expected, atol=0, equal_nan=True
                )

    def test_msa_heatmap_channels(self):
        for ch in (0, 1, self.C_M - 1):
            with self.subTest(ch=ch):
                expected = prepare_heatmap_data(self.m, "msa", channel=ch)
                fig = heatmap_from_representation(self.m, "msa", channel=ch)
                actual = _heatmap_data(fig)
                self.assertEqual(actual.shape, (self.N_SEQ, self.N_RES))
                np.testing.assert_allclose(actual, expected)

    def test_single_heatmap(self):
        expected = prepare_heatmap_data(self.s, "single")
        fig = heatmap_from_representation(self.s, "single")
        actual = _heatmap_data(fig)
        self.assertEqual(actual.shape, (self.N_RES, self.C_S))
        np.testing.assert_allclose(actual, expected)

    def test_single_line_channels(self):
        for ch in (0, 1, self.C_S - 1):
            with self.subTest(ch=ch):
                _, expected = prepare_lineplot_data(self.s, "single", channel=ch)
                fig = line_from_representation(self.s, "single", channel=ch)
                np.testing.assert_allclose(_line_y(fig), expected)

    def test_pair_diagonal_line(self):
        for ch in (0, 2, self.C_Z - 1):
            with self.subTest(ch=ch):
                _, expected = prepare_lineplot_data(self.z, "pair", channel=ch)
                fig = line_from_representation(self.z, "pair", channel=ch)
                np.testing.assert_allclose(_line_y(fig), expected)

    def test_msa_depth_mean_line(self):
        for ch in (0, 1, self.C_M - 1):
            with self.subTest(ch=ch):
                _, expected = prepare_lineplot_data(self.m, "msa", channel=ch)
                fig = line_from_representation(self.m, "msa", channel=ch)
                np.testing.assert_allclose(_line_y(fig), expected)

    def test_lines_overlay_per_channel(self):
        chs = [0, 2, 4]
        fig = lines_from_representation(self.s, "single", channels=chs)
        for i, c in enumerate(chs):
            _, expected = prepare_lineplot_data(self.s, "single", channel=c)
            np.testing.assert_allclose(_line_y(fig, idx=i), expected)

    def test_pair_channel_grid_panels(self):
        chs = [0, 1, 2, 3]
        fig = pair_channel_grid(self.z, channels=chs, ncols=2)
        panels = [ax.get_images() for ax in fig.axes if ax.get_images()]
        self.assertGreaterEqual(len(panels), len(chs))
        for i, c in enumerate(chs):
            expected = prepare_heatmap_data(
                self.z, "pair", channel=c, normalize="minmax"
            )
            np.testing.assert_allclose(np.asarray(panels[i][0].get_array()), expected)

    def test_pair_non_square_rejected(self):
        with self.assertRaises(ValueError):
            heatmap_from_representation(np.zeros((5, 6, 3)), "pair")

    def test_channel_out_of_bounds_rejected(self):
        with self.assertRaises(IndexError):
            heatmap_from_representation(self.z, "pair", channel=999)

    def test_channel_and_aggregate_conflict_rejected(self):
        with self.assertRaises(ValueError):
            heatmap_from_representation(
                self.z, "pair", channel=0, aggregate="mean"
            )

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            heatmap_from_representation(self.z, "bogus")  # type: ignore[arg-type]

    def test_minmax_invariants(self):
        out = prepare_heatmap_data(self.z, "pair", channel=0, normalize="minmax")
        self.assertGreaterEqual(out.min(), -1e-9)
        self.assertLessEqual(out.max(), 1.0 + 1e-9)

    def test_zscore_invariants(self):
        out = prepare_heatmap_data(self.z, "pair", channel=0, normalize="zscore")
        self.assertAlmostEqual(float(np.mean(out)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
