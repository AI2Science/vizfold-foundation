# Issue #42 — Reproducibility and demonstration evidence

Boltz tracing integration: commands to reproduce results, expected signals, and screenshots for review.

Technical background (inputs, layout, proxy attention, `expand_proxy_heads`): [`docs/source/Boltz_Inference.md`](../../docs/source/Boltz_Inference.md).

---

## Environment

- **Repository root:** all commands assume `cd` to the top of this clone (directory that contains `.git`).
- **Branch:** `issue42-boltz-tracing` (or the branch that contains this submission).
- **CPU-only checks:** Python 3; no GPU or Boltz runtime required.
- **GPU job:** Conda environment on cluster scratch with Boltz installed (see Quickstart §1 in `Boltz_Inference.md`). Before `sbatch`, set `BOLTZ_ENV` to that prefix (absolute path to the env directory that contains `bin/boltz`).
- **Slurm:** `scripts/boltz/run_boltz_trace.sbatch` writes logs under `outputs/boltz_runs/slurm-<JOBID>.out` and `.err`. Adjust `#SBATCH` partition, account, or GPU resources if your site differs from the defaults in that file.

---

## Verification checklist

| Step | Command | Expected |
|------|---------|----------|
| Trace format fixtures (CPU) | `python3 scripts/boltz/check_trace_format_fixtures.py` | `[OK] all 3 reference trace fixture(s) valid` |
| Strict layout + manifest paths (CPU) | `python3 -m unittest tests.test_boltz_trace_validate -v` | `Ran 2 tests ... OK` |
| Full inference (GPU cluster) | After `export BOLTZ_ENV=...`, submit the job and inspect logs (see below) | Slurm state **`COMPLETED`**; stdout includes **`[PASS] validate_boltz_traces.py`**; optional **`[INFO] expand_proxy_heads`** when proxy heads are expanded |
| Re-check a finished run | `python3 scripts/boltz/validate_boltz_traces.py --run_dir "$OUT_RUN" --layers 0 --residues 18 --strict` | **`[OK] manifest.json ... paths match --run_dir`**; no **`[FAIL]`** |

### Submit the GPU job and capture `JOBID`

```bash
export BOLTZ_ENV=/path/to/conda/envs/your_boltz_env
JOBID=$(sbatch --export=ALL scripts/boltz/run_boltz_trace.sbatch | awk '{print $4}')
echo "JOBID=$JOBID"
```

### Job status and logs

```bash
sacct -j "$JOBID" --format=JobID,State,ExitCode,Elapsed -n
tail -n 80 outputs/boltz_runs/slurm-${JOBID}.out
tail -n 120 outputs/boltz_runs/slurm-${JOBID}.err
```

### Resolve `OUT_RUN` and re-validate

```bash
OUT_RUN=$(grep -m1 '^\[INFO\] OUT_RUN=' outputs/boltz_runs/slurm-${JOBID}.out | sed 's/.*OUT_RUN=//')
echo "OUT_RUN=$OUT_RUN"
[ -n "$OUT_RUN" ] || { echo "[ERROR] OUT_RUN is empty." >&2; exit 2; }

python3 scripts/boltz/validate_boltz_traces.py \
  --run_dir "$OUT_RUN" --layers 0 --residues 18 --strict
```

---

## Demonstration evidence (screenshots)

Files live in [`screenshots/`](screenshots/). Captions match the filenames.

### 01 — Trace format fixtures (CPU)

![Trace format fixtures](<screenshots/01_Trace format fixtures (CPU).png>)

### 02 — Strict layout + manifest test (CPU)

![Strict layout unittest](<screenshots/02_Strict layout + manifest test (CPU).png>)

### 03 — Flake8

![Flake8](<screenshots/03_Flake8.png>)

### 04 — Conda environment used for Slurm

![Conda env for Slurm](<screenshots/04_Conda env used for Slurm.png>)

### 05 — GPU job and Slurm log

![GPU job and Slurm log](<screenshots/05_GPU Job and SLURM log.png>)

### 06 — Top-level run outputs

![Top-level outputs](<screenshots/06_Top-level Outputs.png>)

### 07 — `component_status.json` (proxy / pairformer)

![component_status.json](<screenshots/07_component_status.json (proxy : pairformer).png>)

### 08 — Sample trace text (format)

![Sample trace text](<screenshots/08_Sample trace text (format).png>)

### 09 — Strict re-validation

![Strict re-validation](<screenshots/09_Strict re-validation.png>)

### 10 — Sample arc diagram (`msa_row` head 0)

![Sample arc PNG](<screenshots/10_msa_row_head_0_layer_0_BOLTZ_arc.png>)
