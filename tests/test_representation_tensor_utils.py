import os
import sys
import unittest

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from representation_tensor_utils import (
    aggregate_channels,
    convert_to_numpy,
    normalize_array,
    prepare_heatmap_data,
    prepare_lineplot_data,
    select_channel,
    summarize_tensor,
    validate_representation,
)


def _pair_dummy(n=8, c=4):
    rng = np.random.default_rng(0)
    return rng.standard_normal((1, 1, n, n, c))


def _msa_dummy(s=5, n=8, c=4):
    rng = np.random.default_rng(1)
    return rng.standard_normal((s, n, c))


def _single_dummy(n=8, c=4):
    rng = np.random.default_rng(2)
    return rng.standard_normal((n, c))


class TestRepresentationTensorUtils(unittest.TestCase):
    def test_validate_pair(self):
        z = _pair_dummy()
        info = validate_representation(z, "pair", n_res=8)
        self.assertEqual(info["n_res"], 8)
        self.assertEqual(info["n_channels"], 4)

    def test_validate_pair_not_square(self):
        z = np.zeros((3, 4, 2))
        with self.assertRaises(ValueError):
            validate_representation(z, "pair")

    def test_validate_single_two_dims(self):
        s = _single_dummy(n=12, c=6)
        info = validate_representation(s, "single", n_res=12)
        self.assertEqual(info["n_channels"], 6)

    def test_prepare_heatmap_pair_channel(self):
        z = _pair_dummy(n=6, c=3)
        h = prepare_heatmap_data(z, "pair", channel=1, normalize="minmax")
        self.assertEqual(h.shape, (6, 6))
        self.assertTrue(np.all(h >= -1e-9))
        self.assertTrue(np.all(h <= 1.0 + 1e-9))

    def test_prepare_heatmap_msa(self):
        m = _msa_dummy(s=4, n=5, c=3)
        h = prepare_heatmap_data(m, "msa", aggregate="mean", normalize="zscore")
        self.assertEqual(h.shape, (4, 5))

    def test_prepare_lineplot_single(self):
        s = _single_dummy(n=10, c=5)
        x, y = prepare_lineplot_data(s, "single", channel=2)
        self.assertEqual(len(x), 10)
        self.assertEqual(len(y), 10)

    def test_prepare_lineplot_pair_diagonal(self):
        z = _pair_dummy(n=7, c=3)
        x, y = prepare_lineplot_data(z, "pair", channel=0)
        self.assertEqual(len(x), 7)
        self.assertEqual(len(y), 7)

    def test_prepare_lineplot_msa_mean_depth(self):
        m = _msa_dummy(s=3, n=6, c=2)
        x, y = prepare_lineplot_data(m, "msa", channel=1)
        self.assertEqual(len(x), 6)
        self.assertEqual(len(y), 6)

    def test_aggregate_and_select(self):
        a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        self.assertEqual(select_channel(a, 2).shape, (2, 3))
        m = aggregate_channels(a, "mean")
        self.assertEqual(m.shape, (2, 3))

    def test_summarize_tensor(self):
        t = np.array([1.0, np.nan, 3.0])
        s = summarize_tensor(t)
        self.assertEqual(s["nan_count"], 1)
        self.assertEqual(s["shape"], (3,))

    def test_convert_to_numpy_identity(self):
        a = np.array([1, 2], dtype=np.float32)
        b = convert_to_numpy(a, dtype=np.float64)
        self.assertEqual(b.dtype, np.float64)

    def test_normalize_none_copies(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = normalize_array(a, "none")
        self.assertFalse(b is a)
        np.testing.assert_array_equal(b, a)


if __name__ == "__main__":
    unittest.main()
