# Folding a protein with the vizfold CLI

The full OpenFold lifecycle through the `vizfold` CLI on a cluster: record a sequence, fold it on a
GPU, register the outputs, and view them. The transcripts below are from a clean run on **NCSA
Delta** (A100), so the paths, accounts and partitions in them are Delta's — `vizfold status` prints
yours. Every `vizfold` call is the installed binary on your `PATH`, not a source checkout.

## Prerequisites

The `vizfold` binary, the checkout it installs from, and an installed OpenFold backend — the
bootstrap from the [README](README.md#install):

```bash
curl -fsSL https://raw.githubusercontent.com/AI2Science/vizfold-foundation/main/install.sh | bash
vizfold install repo
vizfold install openfold
```

On Delta the install is dispatched by SLURM `ClusterName`, so it needs no site arguments and takes
~8 min. Afterwards `vizfold status` shows every part healthy and a populated config:

```text
$ vizfold status
VizFold status

COMPONENT  STATUS  DETAIL
---------  ------  ------
cli        ok      0.9.0 (latest)
repo       ok      /u/yjayawardana/vizfold-repo at v0.9.0
config     ok      19 keys
openfold   ok      /work/nvme/bbol/yjayawardana/vizfold/envs/vizfold-openfold
esmfold    absent  not installed (/work/nvme/bbol/yjayawardana/vizfold/envs/vizfold-esmfold)
scheduler  ok      cpu, gpuA100x4-interactive, bbol-delta-cpu, bbol-delta-gpu

Everything checks out.

Config: /u/yjayawardana/.config/vizfold/vizfold.json
  ...
  OPENFOLD_DATA_DIR = /work/nvme/bbol/yjayawardana/vizfold/openfold/data
  OPENFOLD_GPU_ACCOUNT = bbol-delta-gpu
  OPENFOLD_GPU_PARTITION = gpuA100x4-interactive
  OPENFOLD_GPU_TIME = 01:00:00
  OPENFOLD_HOME = /u/yjayawardana/vizfold-repo
  OPENFOLD_PREFIX = /work/nvme/bbol/yjayawardana/vizfold
  ...
  database = /work/nvme/bbol/yjayawardana/vizfold/vizfold.db (present)
```

The key set is fixed at 19 keys; a value this install did not settle is written empty and reads as
unset. Anything unhealthy is listed under `Problems:` with the command that fixes it.

## The short version

```bash
vizfold list proteins
vizfold run 6KWC_1
```

`vizfold run` takes bundled example ids, paths to FASTAs, directories of FASTAs — several at once,
folded in one execution with the model loaded once — or a queued run's id. Given anything but an
id it records the run itself. That is the line `vizfold install openfold` prints on success. The
rest of this page walks through each stage separately.

## 1. Record a run

Record the bundled example, 6KWC (a 191-residue monomer), without folding it yet. On a cluster
install every input path defaults off the config and the checkout examples, so the target is the
only argument you need:

```bash
vizfold run 6KWC_1 --no-exec
```

Attention maps are dumped by default; pass `--attn=false` to skip them.

```text
Queued OpenFold run 1 (6KWC_1, 191 residues)
```

What the omitted flags default to (all overridable — see `vizfold run --help`):

| Flag | Default on a cluster install |
| --- | --- |
| `<TARGET>` | the bundled `$OPENFOLD_HOME/examples/monomer/fasta_dir_6KWC/6KWC.fasta` |
| `--alignment-dir` | `$OPENFOLD_HOME/examples/monomer/alignments` |
| `--data-dir` | `$OPENFOLD_DATA_DIR` (the staged AlphaFold2 databases) |
| `--model-device` | `cuda:0` — a GPU partition is configured, so the fold will `srun` onto a GPU node |
| `--use-precomputed-alignments` | `true` — every target is a bundled example, so `alignment-dir/6KWC_1` is reused and the MSA search skipped |

The sequence is always read from the FASTA; `--input-id` only names the run, and preflight rejects
it when it does not match the FASTA's header tags. Recording canonicalizes and stores absolute
paths, so every input must exist at that point.

Several targets record as one run whose `input_id` is their tags joined with `+`:

```bash
vizfold run 1UBQ_1 6KWC_1 --no-exec
```

```text
Queued OpenFold run 2 (1UBQ_1+6KWC_1, 267 residues)
```

Recording also seeds the catalog — the OpenFold backend, the local execution target, and the
invocation profile tying them together. Inspect what it created with `vizfold list models`,
`vizfold list targets`, and `vizfold list profiles`.

## 2. Execute the run

```bash
vizfold run 1
```

A GPU partition is configured and no allocation is held, so the OpenFold command is wrapped in the
srun `vizfold.json` describes:

```bash
srun -A bbol-delta-gpu -p gpuA100x4-interactive --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 01:00:00
```

Its output streams as it comes (a queue wait shows first as `srun: job N queued and waiting for
resources`); the preflight report and the final status print after it finishes. The `gpu` check
warns because the login node has none.

```text
Executing run 1
... OpenFold's own output streams here ...

Preflight: passed
[warning] gpu: no GPU visible; the run will fall back to CPU
[passed] program configured: program 'python3' is configured
[passed] script argument configured: script argument 'scripts/openfold/run_pretrained_openfold.py' follows -u
[passed] working directory: '/u/yjayawardana/vizfold-repo' exists
[passed] script file: '/u/yjayawardana/vizfold-repo/scripts/openfold/run_pretrained_openfold.py' exists
[passed] input_id: run input_id '6KWC_1' is configured
[passed] fasta_dir: '/u/yjayawardana/vizfold-repo/examples/monomer/fasta_dir_6KWC/6KWC.fasta' holds 1 FASTA file(s), tagged '6KWC_1' as run input_id says
...

Command exit_code: 0

Final status: completed

Run 1 completed in 78s. View it with: vizfold serve
```

Verify the structure directly — a relaxed 6KWC prediction is 2839 atoms, and `--attn` wrote one
text trace per layer/head:

```bash
$ grep -c '^ATOM' /work/nvme/bbol/yjayawardana/vizfold/runs/1/predictions/6KWC_1_model_1_ptm_relaxed.pdb
2839
$ ls /work/nvme/bbol/yjayawardana/vizfold/runs/1/attention/6KWC_1 | wc -l
96
```

Outputs land in the run workspace `$OPENFOLD_PREFIX/runs/<run-id>`: `predictions/` (relaxed and
unrelaxed PDBs, `timings.json`) and `attention/<tag>/`. Every output is keyed by FASTA tag, so a
batch of targets shares the one workspace without colliding. An OpenFold run always creates
`attention/`; `--attn=false` only leaves it empty.

## 3. Register artifacts

`vizfold run` already did this — a completed run whose outputs were never registered is invisible to
the workbench, so registration rides along with execution. The command remains for a run whose
outputs appeared later:

```bash
vizfold register-artifacts 1
```

It is idempotent, and it never blocks a partial run: if a run failed it warns and registers only the
directories that actually exist.

## 4. Inspect the run

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

## 5. View it in the dashboard

```bash
vizfold serve            # every installed backend; name them to serve a subset
```

This starts the Bun workbench — `vizfold install repo` staged it and installed its dependencies — at
`http://localhost:3000`, serving each run's outputs out of `$OPENFOLD_PREFIX/runs` itself. The
dashboard renders the predicted structure in an interactive 3D viewer, draws arc diagrams in the
browser from the run's own attention dump, and lists the activations it stored. It folds with, and
lists runs from, the backends it serves —
`vizfold serve openfold` on a host with both installed hides the ESMFold runs. From a laptop,
forward the port over SSH:

```bash
ssh -L 3000:localhost:3000 <you>@delta.ncsa.illinois.edu
```

## Driving the model directly

Everything above goes through the executor, which fills the model's arguments in from the config and
records the run. To use the backend on its own instead, run its CLI inside its environment —
`vizfold install openfold` prints this line:

```bash
micromamba run -p /work/nvme/bbol/yjayawardana/vizfold/envs/vizfold-openfold openfold --help
```

`run -p` applies the environment's `activate.d` hook — `CUTLASS_PATH`, `OPENFOLD_DATA_DIR`, the
library paths, and the NVRTC `LD_PRELOAD` where the install pinned one — which is exactly how
`vizfold run` invokes it. You then supply every path yourself. ESMFold is the same command against
its own environment, and neither form needs the `vizfold` binary on your `PATH`.

## Common failure modes

### `run vizfold install openfold first`

The config isn't initialized — the install didn't finish, or on a shared filesystem the freshly
written `vizfold.json` hasn't propagated to the login node yet. Check with `vizfold status`; if the
install did complete, wait a few seconds and retry.

### FASTA / input-id mismatch

The executor checks that the FASTA header-derived tags are exactly the `+`-joined tags in the run
`input_id`. With `--input-id 6KWC_1`, the header in the FASTA must resolve to `6KWC_1`. A
multi-record FASTA is refused here rather than silently skipped by OpenFold's monomer mode.

### Missing precomputed alignment

With `--use-precomputed-alignments=true`, `alignment-dir/<tag>` must exist for every tag in the
batch — for the example, `examples/monomer/alignments/6KWC_1`. It defaults on only when every
target is a bundled example, so a FASTA of your own runs the full search unless you say
otherwise. Pass `--use-precomputed-alignments=false` to run the full MSA search instead (much slower; needs the full
databases).

### `srun: Requested time limit is invalid` / `Invalid account or account/partition combination`

The GPU partition and time cap come from the site profile (`sites/<ClusterName>.json`); the account
is worked out during the install instead — on Delta, `slurm::nvme_alloc` names it from the allocation
plus a `-delta-gpu` suffix, and on Delta-AI a `-dtai-gh` one. All three are written to
`vizfold.json`. If you override any `OPENFOLD_GPU_*` value, keep it within that partition's limits;
Delta's `gpuA100x4-interactive` caps at `01:00:00`.
