import os
import tempfile
import unittest

from visualize_attention_data import (
    get_attention_file_path,
    load_attention_map,
    parse_fasta_sequence,
)


class TestVisualizeAttentionData(unittest.TestCase):
    def test_load_attention_map_groups_sorts_and_filters_heads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "msa_row_attn_layer47.txt")
            with open(path, "w") as f:
                f.write("layer 47 head 0\n")
                f.write("2 3 0.1\n")
                f.write("0 1 0.9\n")
                f.write("layer 47 head 1\n")
                f.write("4 5 0.4\n")
                f.write("6 7 0.2\n")

            heads = load_attention_map(path, top_k=1)

        self.assertEqual(heads, {0: [(0, 1, 0.9)], 1: [(4, 5, 0.4)]})

    def test_get_attention_file_path_uses_expected_names(self):
        self.assertEqual(
            get_attention_file_path("/tmp/attn", "msa_row", 47),
            "/tmp/attn/msa_row_attn_layer47.txt",
        )
        self.assertEqual(
            get_attention_file_path("/tmp/attn", "triangle_start", 47, residue_idx=18),
            "/tmp/attn/triangle_start_attn_layer47_residue_idx_18.txt",
        )

    def test_parse_fasta_sequence_joins_sequence_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "protein.fasta")
            with open(path, "w") as f:
                f.write(">protein\n")
                f.write("ACD\n")
                f.write("EFG\n")

            self.assertEqual(parse_fasta_sequence(path), "ACDEFG")


if __name__ == "__main__":
    unittest.main()
