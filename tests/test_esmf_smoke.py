"""
Smoke test for ESMFold backend: CLI and output layout.

Runs in CPU mode on a tiny sequence so CI stays fast.
Skips actual ESMFold inference if transformers is not installed.

Run with: pytest tests/test_esmf_smoke.py -v
Or without pytest: python tests/test_esmf_smoke.py
"""
import os
import subprocess
import sys
import tempfile
import torch
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pytest
except ImportError:
    pytest = None


def _has_esm() -> bool:
    try:
        from transformers import EsmForProteinFolding  # noqa: F401
        return True
    except ImportError:
        return False


def test_esmf_import():
    """Backend and runner can be imported when transformers is present."""
    if not _has_esm():
        return  # skip when no transformers
    from vizfold.backends.esmfold.inference import ESMFoldRunner
    from vizfold.backends.esmfold.schema import build_meta, write_meta
    assert ESMFoldRunner is not None
    meta = build_meta(
        backend="esmfold",
        model_name="facebook/esmfold_v1",
        out_dir="/tmp",
        fasta_path=None,
        device="cpu",
        dtype="float32",
        sequence_length=10,
        fasta_hash="abc",
        layer_count=1,
        head_count=4,
        trace_mode="none",
        trace_formats=[],
        shapes_recorded={},
    )
    assert meta["backend"] == "esmfold"
    assert "date_time" in meta


def test_cli_help():
    """CLI runs and shows help."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_pretrained_esmf.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--fasta" in result.stdout
    assert "--trace_mode" in result.stdout


def test_cli_missing_fasta():
    """CLI exits non-zero when FASTA is missing."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "run_pretrained_esmf.py"),
            "--fasta", "/nonexistent.fasta",
            "--out", "/tmp/out_esmf_smoke",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr or "Error" in result.stderr


def test_esmf_smoke_run_cpu(tmp_path=None):
    """
    Run ESMFold on a tiny sequence (CPU), check output layout.
    Kept minimal so it stays under a few minutes.
    """
    if not _has_esm():
        return  # skip when no transformers
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">tiny\nMKFLKFSLLTAVLLSVVFAFSSCGDDDD\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "run_pretrained_esmf.py"),
            "--fasta", str(fasta),
            "--out", str(out_dir),
            "--device", "cpu",
            "--trace_mode", "none",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)

    assert (out_dir / "meta.json").exists()
    assert (out_dir / "logs.txt").exists()
    structure_dir = out_dir / "structure"
    if structure_dir.exists():
        assert list(structure_dir.iterdir())


def test_esmf_trace_extraction(tmp_path=None):
    """
    Run ESMFold with attention+activations, verify trace tensors are
    written and have the expected shape.
    """
    if not _has_esm():
        return
    import torch

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">tiny\nMKFLKFSL\n")
    out_dir = tmp_path / "out_trace"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "run_pretrained_esmf.py"),
            "--fasta", str(fasta),
            "--out", str(out_dir),
            "--device", "cpu",
            "--trace_mode", "attention+activations",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)

    assert (out_dir / "meta.json").exists()
    assert (out_dir / "structure" / "predicted.pdb").exists()
    trace_dir = out_dir / "trace"
    assert trace_dir.exists()

    pt_files = list(trace_dir.rglob("*.pt"))
    assert len(pt_files) >= 36, f"Expected >=36 trace tensors, got {len(pt_files)}"

    attn_files = sorted((trace_dir / "attention").glob("*.pt"))
    assert attn_files, "No attention .pt files found"
    sample = torch.load(attn_files[0], map_location="cpu", weights_only=True)
    assert sample.dim() == 4, f"Expected 4D attention tensor [B,H,N,N], got {sample.shape}"
    assert sample.shape[0] == 1, f"Expected batch dim == 1, got {sample.shape[0]}"
    assert sample.shape[2] == sample.shape[3], f"Attention map not square: {sample.shape}"


# Pytest decorators when available
if pytest is not None:
    test_esmf_import = pytest.mark.skipif(not _has_esm(), reason="transformers not installed")(test_esmf_import)
    test_esmf_smoke_run_cpu = pytest.mark.skipif(not _has_esm(), reason="transformers not installed")(test_esmf_smoke_run_cpu)
    test_esmf_trace_extraction = pytest.mark.skipif(not _has_esm(), reason="transformers not installed")(test_esmf_trace_extraction)


if __name__ == "__main__":
    # Run without pytest
    failed = []
    # CLI help
    print("test_cli_help ...", end=" ")
    try:
        test_cli_help()
        print("ok")
    except AssertionError as e:
        print("FAIL", e)
        failed.append("test_cli_help")
    # CLI missing fasta
    print("test_cli_missing_fasta ...", end=" ")
    try:
        test_cli_missing_fasta()
        print("ok")
    except AssertionError as e:
        print("FAIL", e)
        failed.append("test_cli_missing_fasta")
    # Schema/build_meta (no ESM needed)
    print("test_schema_build_meta ...", end=" ")
    try:
        from vizfold.backends.esmfold.schema import build_meta
        meta = build_meta(
            backend="esmfold", model_name="x", out_dir="/tmp", fasta_path=None,
            device="cpu", dtype="float32", sequence_length=10, fasta_hash="h",
            layer_count=1, head_count=4, trace_mode="none", trace_formats=[],
            shapes_recorded={},
        )
        assert meta["backend"] == "esmfold" and "date_time" in meta
        print("ok")
    except Exception as e:
        print("FAIL", e)
        failed.append("test_schema_build_meta")
    # ESMFold import (when transformers present)
    print("test_esmf_import ...", end=" ")
    try:
        test_esmf_import()
        print("ok (or skipped)")
    except Exception as e:
        print("FAIL", e)
        failed.append("test_esmf_import")
    # Full smoke run (when transformers present)
    print("test_esmf_smoke_run_cpu ...", end=" ")
    try:
        test_esmf_smoke_run_cpu()
        print("ok (or skipped)")
    except Exception as e:
        print("FAIL", e)
        failed.append("test_esmf_smoke_run_cpu")
    # Trace extraction
    print("test_esmf_trace_extraction ...", end=" ")
    try:
        test_esmf_trace_extraction()
        print("ok (or skipped)")
    except Exception as e:
        print("FAIL", e)
        failed.append("test_esmf_trace_extraction")
    if failed:
        print("Failed:", failed)
        sys.exit(1)
    print("All checks passed.")


def test_esmfold_backend_smoke(tmp_path):
    output_dir = tmp_path / "test_trace_ci"

    cmd = [
        sys.executable,
        "run_pretrained_esmf.py",
        "--fasta",
        "examples/monomer/fasta_dir_6KWC/6KWC.fasta",
        "--out",
        str(output_dir),
        "--trace_mode",
        "attention+activations",
        "--device",
        "cpu",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    # expected top-level outputs
    assert os.path.exists(output_dir / "meta.json")
    assert os.path.exists(output_dir / "structure" / "predicted.pdb")
    assert os.path.exists(output_dir / "trace")

    # collect all .pt trace files
    trace_files = []
    for root, _, files in os.walk(output_dir / "trace"):
        trace_files += [
            os.path.join(root, f) for f in files if f.endswith(".pt")
        ]

    assert len(trace_files) >= 72

    # validate attention tensor shape specifically
    attention_dir = output_dir / "trace" / "attention"
    assert attention_dir.exists()

    attention_files = [
        attention_dir / f
        for f in os.listdir(attention_dir)
        if f.endswith(".pt")
    ]

    assert len(attention_files) == 36

    sample_attention = torch.load(attention_files[0], map_location="cpu")
    assert len(sample_attention.shape) == 4
    assert sample_attention.shape[0] == 1
    assert sample_attention.shape[2] == sample_attention.shape[3]

    # validate activation tensors exist
    activation_dir = output_dir / "trace" / "activations"
    assert activation_dir.exists()

    activation_files = [
        activation_dir / f
        for f in os.listdir(activation_dir)
        if f.endswith(".pt")
    ]

    assert len(activation_files) == 36

    sample_activation = torch.load(activation_files[0], map_location="cpu")
    assert len(sample_activation.shape) == 3
    assert sample_activation.shape[0] == 1

    # validate Evoformer trunk intermediates
    trunk_dir = output_dir / "trace" / "trunk"
    assert trunk_dir.exists()

    trunk_files = [
        trunk_dir / f
        for f in os.listdir(trunk_dir)
        if f.endswith(".pt")
    ]

    assert len(trunk_files) >= 96

    # if final or pair outputs exist, validate s_z style shape [N, N, D]
    trunk_sz_candidates = [
        f for f in trunk_files
        if "s_z" in f.name or "pair" in f.name
    ]
    if trunk_sz_candidates:
        sample_sz = torch.load(trunk_sz_candidates[0], map_location="cpu")
        assert len(sample_sz.shape) == 3
        assert sample_sz.shape[0] == sample_sz.shape[1]

    # validate attention text export
    attention_txt_dir = output_dir / "attention_files"
    assert attention_txt_dir.exists()

    attention_txt_files = [
        attention_txt_dir / f
        for f in os.listdir(attention_txt_dir)
        if f.endswith(".txt")
    ]

    assert len(attention_txt_files) == 36

    # optional validation for recycling outputs if present
    recycle_ss = [
        f for f in activation_files
        if "recycle_" in f.name and "_s_s" in f.name
    ]
    recycle_sz = [
        f for f in activation_files
        if "recycle_" in f.name and "_s_z" in f.name
    ]

    if recycle_ss:
        sample_recycle_ss = torch.load(recycle_ss[0], map_location="cpu")
        assert len(sample_recycle_ss.shape) == 3

    if recycle_sz:
        sample_recycle_sz = torch.load(recycle_sz[0], map_location="cpu")
        assert len(sample_recycle_sz.shape) == 3
        assert sample_recycle_sz.shape[0] == sample_recycle_sz.shape[1]

    # optional validation for structure-module / IPA outputs if present
    ipa_candidates = []
    for root, _, files in os.walk(output_dir / "trace"):
        ipa_candidates += [
            os.path.join(root, f)
            for f in files
            if f.endswith(".pt") and "ipa" in f.lower()
        ]

    if ipa_candidates:
        sample_ipa = torch.load(ipa_candidates[0], map_location="cpu")
        assert len(sample_ipa.shape) >= 3


def test_esmfold_full_validation(tmp_path):
    output_dir = tmp_path / "test_trace_ci"

    cmd = [
        sys.executable,
        "run_pretrained_esmf.py",
        "--fasta",
        "examples/monomer/fasta_dir_6KWC/6KWC.fasta",
        "--out",
        str(output_dir),
        "--trace_mode",
        "attention+activations",
        "--device",
        "cpu",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    # expected outputs
    assert os.path.exists(f"{output_dir}/meta.json")
    assert os.path.exists(f"{output_dir}/structure/predicted.pdb")
    assert os.path.exists(f"{output_dir}/trace")

    # collect all trace tensor files
    trace_files = []
    for root, _, files in os.walk(f"{output_dir}/trace"):
        trace_files += [
            os.path.join(root, f) for f in files if f.endswith(".pt")
        ]

    assert len(trace_files) >= 36

    # validate attention tensor shape specifically
    attention_dir = os.path.join(output_dir, "trace", "attention")
    attention_files = [
        os.path.join(attention_dir, f)
        for f in os.listdir(attention_dir)
        if f.endswith(".pt")
    ]

    assert len(attention_files) >= 1

    sample_tensor = torch.load(attention_files[0], map_location="cpu")
    assert len(sample_tensor.shape) == 4
    assert sample_tensor.shape[0] == 1
    assert sample_tensor.shape[2] == sample_tensor.shape[3]