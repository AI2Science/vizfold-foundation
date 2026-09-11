import os
import tempfile
import unittest

from visualize_attention_chord_diagrams import (
    aggregate_chord_edges,
    generate_chord_diagrams,
)


class TestVisualizeAttentionChordDiagrams(unittest.TestCase):
    def test_aggregate_chord_edges_uses_mean_by_default(self):
        heads = {
            0: [(0, 1, 0.2), (2, 3, 0.5)],
            1: [(0, 1, 0.6)],
        }

        self.assertEqual(
            aggregate_chord_edges(heads),
            [(2, 3, 0.5), (0, 1, 0.4)],
        )

    def test_generate_chord_diagrams_writes_expected_msa_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            attn_dir = os.path.join(tmpdir, "attention")
            out_dir = os.path.join(tmpdir, "chords")
            os.makedirs(attn_dir)
            with open(os.path.join(attn_dir, "msa_row_attn_layer47.txt"), "w") as f:
                f.write("layer 47 head 0\n")
                f.write("0 1 0.8\n")
                f.write("1 2 0.2\n")
                f.write("layer 47 head 1\n")
                f.write("2 3 0.7\n")
                f.write("3 4 0.1\n")

            paths = generate_chord_diagrams(
                attention_dir=attn_dir,
                residue_sequence="ABCDE",
                output_dir=out_dir,
                protein="TEST",
                attention_type="msa_row",
                top_k=1,
                layer_idx=47,
                save_individual=True,
                save_grid=True,
                save_aggregated=True,
            )

            self.assertEqual(len(paths), 4)
            for path in paths:
                self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
