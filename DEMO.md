# Folding a protein with the vizfold CLI

This is the full OpenFold lifecycle through the `vizfold` CLI on a cluster: seed the executor,
queue a sequence, fold it on a GPU, register the outputs, and view them. The commands below are
from a clean run on **NCSA Delta** (A100); the outputs shown are that run's actual results. Every
`vizfold` call is the installed binary on your `PATH` — nothing is run from a source checkout.

## Prerequisites

The `vizfold` binary and an installed OpenFold backend. If you have neither, do the two-command
bootstrap from the [README](README.md#install):

```bash
curl -sL https://raw.githubusercontent.com/AI2Science/vizfold-foundation/main/install.sh | bash
vizfold install openfold
```

On Delta the install is dispatched by SLURM `ClusterName`, so it needs no site arguments and takes
~8 min. When it finishes, `vizfold status` shows OpenFold `installed` and a populated config:

```text
$ vizfold status
VizFold status

Config: /u/yjayawardana/.config/vizfold/vizfold.json
  ESMFOLD_ENV_PREFIX =
  OPENFOLD_ACCOUNT = bbol-delta-cpu
  OPENFOLD_AF2_ROOT = /work/hdd/data/alphafold2/database
  OPENFOLD_DATA_DIR = /work/nvme/bbol/yjayawardana/vizfold/data
  OPENFOLD_DRIVER_CUDA = 12.2
  OPENFOLD_ENV_PREFIX = /work/nvme/bbol/yjayawardana/vizfold/envs/vizfold-openfold
  OPENFOLD_EXAMPLE = 6KWC_1
  OPENFOLD_FOLD_ARGS =
  OPENFOLD_GPU_ACCOUNT = bbol-delta-gpu
  OPENFOLD_GPU_GRES = gpu:1
  OPENFOLD_GPU_PARTITION = gpuA100x4-interactive
  OPENFOLD_GPU_RESOURCES = --cpus-per-task=8 --mem=32G
  OPENFOLD_GPU_TIME = 01:00:00
  OPENFOLD_HOME = /u/yjayawardana/vizfold-src
  OPENFOLD_MAX_CUDA = 12.8
  OPENFOLD_PARTITION = cpu
  OPENFOLD_PREFIX = /work/nvme/bbol/yjayawardana/vizfold
  OPENFOLD_SITE = delta
  VIZFOLD_DB = /work/nvme/bbol/yjayawardana/vizfold/vizfold.db
  VIZFOLD_ENV_BASE = /work/nvme/bbol/yjayawardana/vizfold/envs
  database = /work/nvme/bbol/yjayawardana/vizfold/vizfold.db (present)

Backends:
BACKEND   STATUS         ENV PREFIX
--------  -------------  --------------------------------------------------------
openfold  installed      /work/nvme/bbol/yjayawardana/vizfold/envs/vizfold-openfold
esmfold   not installed  /work/nvme/bbol/yjayawardana/vizfold/envs/vizfold-esmfold
```

That key set is fixed: every cluster and either backend writes the same names, and a name this
install did not settle (`ESMFOLD_ENV_PREFIX`, `OPENFOLD_FOLD_ARGS` above) is written empty, which
every reader treats as unset. The values are resolved live during install — on Delta the prefix and
data directory land under your `/work/nvme` allocation. Everything after this point reads them from
`vizfold.json`.

## 1. Seed the executor records

Once per install, create the catalog the runs reference — the OpenFold backend, the local
execution target, and the invocation profile that ties them together. It is existence-guarded, so
re-running is harmless.

```bash
vizfold seed
```

```text
Seeded default executor records.
```

You can inspect what it created:

```bash
vizfold list models
vizfold list targets
vizfold list profiles
```

## 2. Queue a run

Queue the bundled example, 6KWC (a ~190-residue monomer). On a cluster install the input paths all
default off the config and the checkout examples, so the only flags you need are the sequence and
its id — plus `--demo-attn` to also dump attention maps:

```bash
vizfold queue-run openfold \
  --input-id 6KWC_1 \
  --input-sequence GSTIQPGTGYNNGYFYSYWNDGHGGVTYTNGPGGQFSVNWSNSGEFVGGKGWQPGTKNKVINFSGSYNPNGNSYLSVYGWSRNPLIEYYIVENFGTYNPSTGATKLGEVTSDGSVYDIYRTQRVNQPSIIGTATFYQYWSVRRNHRSSGSVNTANHFNAWAQQGLTLGTMDYQIVAVQGYFSSGSASITVS \
  --demo-attn
```

```text
Queued OpenFold run 1
status: submitted
input_id: 6KWC_1

Next:
  vizfold execute-run 1
```

What the omitted flags default to (all overridable — see `vizfold queue-run openfold --help`):

| Flag | Default on a cluster install |
| --- | --- |
| `--fasta-dir` | `$OPENFOLD_HOME/examples/monomer/fasta_dir_6KWC` |
| `--alignment-dir` | `$OPENFOLD_HOME/examples/monomer/alignments` |
| `--data-dir` | `$OPENFOLD_DATA_DIR` (the staged AlphaFold2 databases) |
| `--model-device` | `cuda:0` — a GPU partition is configured, so the fold will `srun` onto a GPU node |
| `--use-precomputed-alignments` | `true` — reuse `alignment-dir/6KWC_1`, skipping the MSA search |

The FASTA file in `fasta-dir` must have a header matching `--input-id` (here `>6KWC_1`);
`--input-sequence` is that sequence, recorded with the run. With precomputed alignments enabled the
directory `alignment-dir/<input-id>` (here `examples/monomer/alignments/6KWC_1`) must exist. The
queue step canonicalizes and stores absolute paths in the run record, so all inputs must be present
when you queue.

## 3. Execute the run

```bash
vizfold execute-run 1
```

`execute-run` prints each preflight check — the `gpu` one warns because the login node has none —
then, because a GPU partition is configured but no allocation is held, wraps the OpenFold command
in the srun `vizfold.json` describes:
`srun -A bbol-delta-gpu -p gpuA100x4-interactive --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 01:00:00`,
streams its logs, and reports the final status. On Delta the fold itself took ~78 s on one A100
(a queue wait shows first as `srun: job N queued and waiting for resources`).

```text
Executing run 1

Preflight: passed
[warning] gpu: no GPU visible; the run will fall back to CPU
[passed] program configured: program 'python3' is configured
[passed] script argument configured: script argument 'scripts/openfold/run_pretrained_openfold.py' follows -u
[passed] working directory: '/u/yjayawardana/vizfold-src' exists
[passed] script file: '/u/yjayawardana/vizfold-src/scripts/openfold/run_pretrained_openfold.py' exists
[passed] input_id: run input_id '6KWC_1' is configured
[passed] fasta_dir: '/u/yjayawardana/vizfold-src/examples/monomer/fasta_dir_6KWC' contains one FASTA file with tag '6KWC_1' matching run input_id
...

Final status: completed

Run 1 completed in 78s. View it with: vizfold serve
```

Verify the structure directly — a relaxed 6KWC prediction is 2839 atoms, and `--demo-attn` wrote
one text trace per layer/head:

```bash
$ grep -c '^ATOM' /work/nvme/bbol/yjayawardana/vizfold/runs/1/predictions/6KWC_1_model_1_ptm_relaxed.pdb
2839
$ ls /work/nvme/bbol/yjayawardana/vizfold/runs/1/attention | wc -l
96
```

Outputs land in the run workspace `$OPENFOLD_PREFIX/runs/<run-id>`: `predictions/` (relaxed and
unrelaxed PDBs, `timings.json`) and, with `--demo-attn`, `attention/`.

## 4. Register artifacts

`execute-run` already did this — a completed run whose outputs were never registered is invisible
to the workbench, so registration rides along with execution. The command remains for re-running it
against a run whose outputs appeared later; it is idempotent and reports what is already present.

```bash
vizfold register-artifacts 1
```

```text
Registered artifacts for run 1

Output workspace:
  /work/nvme/bbol/yjayawardana/vizfold/runs/1

Artifacts:
  [registered] run_output_directory -> /work/nvme/bbol/yjayawardana/vizfold/runs/1
  [registered] attention_output_directory -> /work/nvme/bbol/yjayawardana/vizfold/runs/1/attention
```

Registration never blocks a partial run — if a run failed it warns and registers only the
directories that actually exist.

## 5. Inspect the run

```bash
vizfold show run 1
```

```text
Run 1
status: completed
input_id: 6KWC_1
model_backend_id: 1
execution_target_id: 1
invocation_profile_id: 1
submitted_at: 2026-07-24T16:22:35+00:00
started_at: 2026-07-24T16:41:14+00:00
completed_at: 2026-07-24T16:42:32+00:00
artifacts:
ID  TYPE ID  FORMAT     STORAGE URI
--  -------  ---------  -----------------------------------------------------
1   12       directory  /work/nvme/bbol/yjayawardana/vizfold/runs/1
2   13       directory  /work/nvme/bbol/yjayawardana/vizfold/runs/1/attention
```

`vizfold list runs` (optionally `--status completed`) lists all runs.

## 6. View it in the dashboard

```bash
vizfold serve
```

This stages and starts the Next.js workbench (installing its dependencies on first run) at
`http://localhost:3000`, linking `$OPENFOLD_PREFIX/runs` under the app's `public/` so the browser
can load each run's outputs. The dashboard renders the predicted structure in an interactive 3D
viewer and shows the attention maps. From a laptop, forward the port over SSH:

```bash
ssh -L 3000:localhost:3000 <you>@delta.ncsa.illinois.edu
```

## A quicker smoke test

To confirm the install without the executor at all, `vizfold install openfold` prints a one-liner
that folds the bundled example straight through `fold.sh` and counts the atoms:

```bash
srun -A bbol-delta-gpu -p gpuA100x4-interactive --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 01:00:00 \
  env OPENFOLD_PREFIX=/work/nvme/bbol/yjayawardana/vizfold \
  /u/yjayawardana/vizfold-src/scripts/openfold/fold.sh 6KWC_1
grep -c '^ATOM' /work/nvme/bbol/yjayawardana/vizfold/outputs/6KWC_1/predictions/6KWC_1_model_1_ptm_relaxed.pdb
```

The account, partition, and resources here are Delta's; `vizfold install openfold` prints the
command already filled in for whatever cluster it ran on — and prints it without the `srun` when
you install from inside a GPU allocation, where the fold runs as-is.

A few thousand atoms means it worked. The executor lifecycle above is the fuller path — it
persists the run, its provenance, and its artifacts, and feeds the dashboard.

## Common failure modes

### `run vizfold install openfold first`

The config isn't initialized — the install didn't finish, or on a shared filesystem the freshly
written `vizfold.json` hasn't propagated to the login node yet. Check with `vizfold status`; if the
install did complete, wait a few seconds and retry.

### FASTA / input-id mismatch

The executor checks that the FASTA header-derived tag matches the run `input_id`. With
`--input-id 6KWC_1`, the header in the FASTA must resolve to `6KWC_1`.

### Missing precomputed alignment

With `--use-precomputed-alignments=true` (the default), the directory `alignment-dir/<input-id>`
must exist — for the example, `examples/monomer/alignments/6KWC_1`. Pass
`--use-precomputed-alignments=false` to run the full MSA search instead (much slower; needs the
full databases).

### `srun: Requested time limit is invalid` / `Invalid account or account/partition combination`

The GPU partition and time cap come from the site profile
(`backends/openfold/install/sites/<ClusterName>.json`); the account is not in that file — it is
worked out during the install (on Delta, `slurm::nvme_alloc` names it from the allocation plus a
`-delta-gpu` suffix). All three are written to `vizfold.json`. If you override any
`OPENFOLD_GPU_*` value, keep it within the partition's limits (on Delta,
`gpuA100x4-interactive` caps at `01:00:00`).
