# CPU-only regression for Boltz trace strict validation (Issue #42).
# No OpenFold/torch imports; safe on GitHub Actions without GPU deps.
#
# Licensed under the Apache License, Version 2.0.

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "scripts" / "boltz" / "fixtures"
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "boltz" / "validate_boltz_traces.py"


def _populate_minimal_strict_run_dir(run_dir: Path) -> None:
    """Layout matches a successful boltz trace run (Issue #42 / validate --strict)."""
    run_dir = run_dir.resolve()
    attn = run_dir / "attn_txt"
    pred = run_dir / "pred"
    act = run_dir / "act_npz"
    arc = run_dir / "arc_png"
    attn.mkdir(parents=True)
    pred.mkdir(parents=True)
    act.mkdir(parents=True)
    arc.mkdir(parents=True)

    shutil.copy(
        FIXTURES / "reference_msa_row_attn_layer0.txt",
        attn / "msa_row_attn_layer0.txt",
    )
    shutil.copy(
        FIXTURES / "reference_triangle_start_attn_layer0_residue_idx_18.txt",
        attn / "triangle_start_attn_layer0_residue_idx_18.txt",
    )
    shutil.copy(
        FIXTURES / "reference_triangle_end_attn_layer0_residue_idx_18.txt",
        attn / "triangle_end_attn_layer0_residue_idx_18.txt",
    )

    (pred / "structure_placeholder.txt").write_text("fixture\n", encoding="utf-8")

    status = {
        "msa": {"available": False, "source": None, "files_written": 0},
        "pairformer_boltz": {"available": True, "source": "fixture", "files_written": 1},
        "sm_boltz": {"available": False, "source": None, "files_written": 0},
    }
    (attn / "component_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )

    rd_s = str(run_dir)
    manifest = {
        "run_dir": rd_s,
        "timestamp": "fixture",
        "repo": {"path": str(REPO_ROOT), "git_sha": "test"},
        "inputs": {"yaml": "fixture.yaml", "fasta": "fixture.fasta"},
        "outputs": {
            "pred": str(pred),
            "attn_txt": str(attn),
            "act_npz": str(act),
            "arc_png": str(arc),
        },
        "trace": {"head": "all", "topk": "50", "residues": "18", "layers": "0"},
        "boltz": {"no_kernels": True, "seed": 0, "cache": "fixture"},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _run_validate(run_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--run_dir",
            str(run_dir.resolve()),
            "--layers",
            "0",
            "--residues",
            "18",
            "--strict",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


class TestBoltzTraceStrictValidate(unittest.TestCase):
    """Known-example layout: fixtures + manifest paths aligned with --run_dir."""

    def test_strict_passes_on_minimal_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "out_run"
            _populate_minimal_strict_run_dir(run_dir)
            cp = _run_validate(run_dir)
            self.assertEqual(
                cp.returncode,
                0,
                msg=f"stdout:\n{cp.stdout}\nstderr:\n{cp.stderr}",
            )
            self.assertIn("paths match --run_dir", cp.stdout)

    def test_strict_fails_when_manifest_run_dir_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "out_run"
            _populate_minimal_strict_run_dir(run_dir)
            man_path = run_dir / "manifest.json"
            man = json.loads(man_path.read_text(encoding="utf-8"))
            man["run_dir"] = "/this/is/not/the/run/directory"
            man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")

            cp = _run_validate(run_dir)
            self.assertNotEqual(cp.returncode, 0)
            self.assertIn("manifest run_dir", cp.stdout + cp.stderr)


if __name__ == "__main__":
    unittest.main()
