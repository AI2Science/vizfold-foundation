import pytest

from vizfold.offline import ArchiveReader


def test_archive_reader_constructor() -> None:
    reader = ArchiveReader("dummy_archive")
    assert str(reader.archive_root).endswith("dummy_archive")


@pytest.mark.parametrize(
    "method_name,args,kwargs",
    [
        ("metadata", tuple(), {}),
        ("list_attention_types", tuple(), {}),
        ("list_layers", ("msa_row",), {}),
        ("list_heads", ("msa_row", 47), {}),
        ("list_residue_indices", ("triangle_start", 47), {}),
        ("load_attention", ("msa_row", 47, 0), {}),
        ("load_attention_heads", ("msa_row", 47), {}),
        ("load_structure", tuple(), {}),
    ],
)
def test_archive_reader_methods_raise_not_implemented(method_name, args, kwargs) -> None:
    reader = ArchiveReader("dummy_archive")
    method = getattr(reader, method_name)

    with pytest.raises(NotImplementedError):
        method(*args, **kwargs)