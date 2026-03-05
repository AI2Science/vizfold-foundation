from pathlib import Path

from vizfold.offline import LegacyTxtReader


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_legacy_txt_reader_metadata_and_loading(tmp_path: Path) -> None:
    attention_dir = tmp_path / "attn"
    attention_dir.mkdir()

    fasta_path = tmp_path / "toy.fasta"
    pdb_path = tmp_path / "toy_unrelaxed.pdb"

    _write_text(
        fasta_path,
        """
        >toy
        ACDEFG
        """,
    )

    _write_text(
        pdb_path,
        """
        HEADER    TOY PDB
        ATOM      1  N   ALA A   1      11.104  13.207   9.947  1.00 50.00           N
        END
        """,
    )

    _write_text(
        attention_dir / "msa_row_attn_layer47.txt",
        """
        Layer 47, Head 0
        0 1 0.90
        1 3 0.70

        Layer 47, Head 2
        2 5 0.95
        0 4 0.60
        """,
    )

    _write_text(
        attention_dir / "triangle_start_attn_layer47_residue_idx_18.txt",
        """
        Layer 47, Head 0
        18 20 0.80
        18 21 0.50

        Layer 47, Head 1
        18 30 0.92
        """,
    )

    reader = LegacyTxtReader(
        attention_dir=attention_dir,
        fasta_path=fasta_path,
        pdb_path=pdb_path,
        protein_id="toy",
    )

    meta = reader.metadata()
    assert meta.protein_id == "toy"
    assert meta.sequence == "ACDEFG"
    assert set(meta.attention_types) == {"msa_row", "triangle_start"}
    assert meta.layers_by_type["msa_row"] == [47]
    assert meta.layers_by_type["triangle_start"] == [47]
    assert meta.residue_indices_by_type["triangle_start"][47] == [18]

    msa_heads = reader.list_heads("msa_row", 47)
    assert msa_heads == [0, 2]

    tri_heads = reader.list_heads("triangle_start", 47, residue_idx=18)
    assert tri_heads == [0, 1]

    msa_slice = reader.load_attention("msa_row", layer=47, head=2)
    assert msa_slice.as_triplets()[0] == (2, 5, 0.95)
    assert msa_slice.as_triplets()[1] == (0, 4, 0.60)

    tri_slice = reader.load_attention(
        "triangle_start",
        layer=47,
        head=0,
        residue_idx=18,
        top_k=1,
    )
    assert tri_slice.residue_idx == 18
    assert tri_slice.as_triplets() == [(18, 20, 0.80)]

    structure = reader.load_structure()
    assert structure.protein_id == "toy"
    assert structure.sequence == "ACDEFG"
    assert structure.pdb_text is not None
    assert "HEADER    TOY PDB" in structure.pdb_text