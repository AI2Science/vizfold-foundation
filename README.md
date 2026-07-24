# Vizfold Foundations

Vizfold is a platform for running protein-structure models and inspecting what they compute:

1. Model inference & feature extraction: Run protein structure prediction models and extract intermediate activations (hidden representations) and attention maps from any chosen layer.
2. Visualization & analysis: Explore, visualize, and analyze the extracted activations and attention maps.

The `vizfold` CLI is the platform; a model backend plugs in underneath it. Install one with
`vizfold install <backend>` — **OpenFold** (the full cluster install: micromamba env, CUDA
extension build, AlphaFold2 databases) or **ESMFold** (a lightweight venv with PyTorch +
Transformers, weights pulled from HuggingFace at run time). `vizfold status` shows the resolved
config and which backends are installed. The same `install/` scripts are built to host others
(openfold3, boltz) as they land.

---

Link to Openfold implimentation - [README_vizfold_openfold.md](https://github.com/vizfold/vizfold-foundation/blob/main/README_vizfold_openfold.md)

---

## Install

Two steps on a cluster. First bootstrap the `vizfold` CLI — one command, needs nothing from you:

```bash
curl -sL https://raw.githubusercontent.com/AI2Science/vizfold-foundation/main/install.sh | bash
```

That downloads the prebuilt `vizfold` binary for your architecture from the latest GitHub
release and installs it to `~/.local/bin` (set `VIZFOLD_VERSION=vX.Y.Z` to pin a release). Then
install a backend — OpenFold below, or the lighter `vizfold install esmfold` (see
[docs/esmfold.md](docs/esmfold.md)):

```bash
vizfold install openfold
```

`vizfold install <backend>` clones the matching checkout to `$HOME/vizfold-src` on first run (the binary
ships only itself; the `install/` scripts and dashboard come from there), works out where it is
running, picks the site, submits the OpenFold install to the scheduler, and prints the exact
command to fold a test sequence. Cold: ~8 min on NCSA Delta, ~25 min where the AlphaFold
databases have to be downloaded.

`vizfold install openfold` holds your terminal and streams every step of the install as it happens. On a
cluster it runs as a blocking `srun` job, so a queue wait shows as
`srun: job N queued and waiting for resources`. Use `tmux` or `screen` for long installs — if the
connection drops, re-run `vizfold install openfold` and it continues from the last completed step.

To keep a log, wrap the whole command rather than piping it:

```bash
script -q -e -c 'vizfold install openfold' install.log
```

Do not pipe to `tee` — that replaces the terminal with a pipe, which suppresses download progress
meters and makes the output arrive in delayed bursts.

### Uninstall

```bash
vizfold uninstall
```

Lists everything the install generated — the conda environment (and any ESMFold venv) and the
rest of the install prefix, the package caches beside it, the symlinks and build droppings it left in the checkout,
the run database, the checkout it cloned into `$HOME/vizfold-src`, and
`~/.config/vizfold/vizfold.json` — then removes it once you confirm (`--yes` skips the prompt).
Fold outputs under the prefix, a checkout you pointed it at yourself, and the `vizfold` binary
are left alone; drop the binary with `rm ~/.local/bin/vizfold`.

### Supported clusters

Dispatch is on the SLURM `ClusterName`, so on these machines `vizfold install openfold` needs no
site arguments. Accounts and the install prefix are worked out live (your project space, the
accounts you can charge); the values below are what a fresh install settles on.

| `ClusterName` (cluster) | Verified | Arch | AF2 databases | Build → fold partition (GPU) | Install prefix |
| --- | --- | --- | --- | --- | --- |
| `delta` (NCSA Delta) | ✅ install + fold | x86-64 | mirror¹ | `cpu` → `gpuA100x4-interactive` (A100) | `/work/nvme/<alloc>/<user>/vizfold` |
| `delta-gh` (NCSA Delta-AI) | ✅ install + fold³ | aarch64 (GH200) | mirror¹ | `ghx4` → `ghx4-interactive` (GH200) | `/work/nvme/<alloc>/<user>/vizfold-gh`² |
| `nexus-dev` (Nexus) | ◐ install⁵ | x86-64 | downloaded | `gpu` → `gpu` (A100 10 GB vGPU)⁴ | `/projects/<user>/vizfold` |
| `anvil` (Purdue Anvil) | ◐ install⁵ | x86-64 | downloaded | `shared` → `gpu` (A100) | `$PROJECT/<user>/vizfold` |
| `bridges2` (PSC Bridges-2) | ◐ install⁵ | x86-64 | mirror¹ | `RM-shared` → `GPU-shared` (V100-32) | `/ocean/projects/<acct>/<user>/vizfold` |
| `expanse` (SDSC Expanse) | ⚙️ profile | x86-64 | downloaded | `shared` → `gpu-shared` (V100) | `/expanse/lustre/projects/<acct>/<user>/vizfold` |
| `ice-slurm` (GT PACE ICE) | ⚙️ profile | x86-64 | mirror¹ | `ice-cpu` → `ice-gpu` (A100) | `<scratch>/vizfold` (`/storage/ice1/…`) |
| `phoenix-slurm` (GT PACE Phoenix) | ⚙️ profile | x86-64 | mirror¹ | `cpu-small` → `gpu-a100` (A100) | `<scratch>/vizfold` (`/storage/scratch1/…`) |

Legend — ✅ install + fold verified end-to-end from `vizfold install` (fold → 2839-atom relaxed
structure); ◐ install run on the cluster with its site-specific fixes, final fold not re-confirmed
in this pass⁵; ⚙️ site profile written and its paths probed live, full install not yet run.

1. AF2 mirrors: Delta & Delta-AI (shared `/work/hdd`) `/work/hdd/data/alphafold2/database`,
   Phoenix `/storage/coda1/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data`, ICE
   `/storage/ice1/shared/d-pace_community/…`, Bridges-2 `/ocean/datasets/community/alphafold/v2.3.2`.
   Each mirror lays out `uniclust30` differently (real single- or double-nested set, or none), so
   the install stages it into a canonical dir — real set if present, else aliased from uniref30.
   Where there is no mirror the install downloads the ~4 GB parameters + the example's templates.
2. Delta and Delta-AI share `/work/nvme`, so the aarch64 site uses an `-gh` suffix — otherwise the
   two architectures' environments would clobber each other.
3. The aarch64 conda OpenMM ships no CUDA platform, so relaxation falls back to CPU (~15 s for the
   example) and yields the same structure as the x86 CUDA path.
4. Nexus's 535 driver is older than the env's NVRTC, so the install pins a matching NVRTC via
   `LD_PRELOAD`; the 10 GB vGPU gets the smaller `1UBQ_1` example. CUDA is capped at 12.8 on every
   x86 site and 12.9 on aarch64 (the 13.x build won't compile OpenFold's extension).
5. `◐` detail — nexus: cold-start install completed this session, NVRTC-pinned relaxation confirmed
   in earlier runs; anvil: install reached the dataset stage (fixed a conda-libcurl mmCIF bug),
   A100 fold queue-bound; bridges2: build + fold ran through to relaxation (memory / gcc / CUDA-arch
   / NVRTC fixes applied).

### Settings

Three layers, highest first. Each only fills what the one above left unset, so you override
exactly what you care about and nothing else:

| | | |
| --- | --- | --- |
| 1 | inline environment | `OPENFOLD_PREFIX=/scratch/me/vizfold vizfold install openfold` |
| 2 | `~/.config/vizfold/vizfold.json` | written by the install; edit to make a choice stick |
| 3 | `install/sites/<site>.json` | the site's defaults, in the repo — edit to change them for everyone |

A `<site>.json` carries every variable and templates paths off `$VAR` references, resolved
recursively (`$VAR` against the environment first, then other keys in the same file). The site's
`<site>.sh` discovers only the one login-specific atom the templates need — the allocation, the
SLURM account, or `OPENFOLD_BASE` (the install directory). `install/sites/delta.json`:

```json
{
  "OPENFOLD_ACCOUNT": "$ALLOC-delta-cpu",
  "OPENFOLD_AF2_ROOT": "/work/hdd/data/alphafold2/database",
  "OPENFOLD_BASE": "/work/nvme/$ALLOC/$USER",
  "OPENFOLD_EXAMPLE": "6KWC_1",
  "OPENFOLD_GPU_ACCOUNT": "$ALLOC-delta-gpu",
  "OPENFOLD_GPU_PARTITION": "gpuA100x4-interactive",
  "OPENFOLD_GPU_RESOURCES": "--cpus-per-task=8 --mem=32G",
  "OPENFOLD_MAX_CUDA": "12.8",
  "OPENFOLD_PARTITION": "cpu",
  "OPENFOLD_PREFIX": "$OPENFOLD_BASE/vizfold"
}
```

Here `delta.sh` discovers just `$ALLOC` (the `/work/nvme` allocation, via `sacctmgr`); the
account, base, and prefix all template off it.

`install/sites/nexus-dev.json` — no database mirror, so `OPENFOLD_AF2_ROOT` is absent and the
install fetches the parameters itself. Its GPU is a 10 GB vGPU, hence the smaller example and
memory:

```json
{
  "OPENFOLD_EXAMPLE": "1UBQ_1",
  "OPENFOLD_GPU_PARTITION": "gpu",
  "OPENFOLD_GPU_RESOURCES": "--cpus-per-task=8 --mem=24G",
  "OPENFOLD_MAX_CUDA": "12.8",
  "OPENFOLD_PARTITION": "gpu"
}
```

To override for one run, put the variable inline — it wins over both files:

```bash
OPENFOLD_EXAMPLE=1UBQ_1 OPENFOLD_PARTITION=cpuA100x4 vizfold install openfold
```

Only the login-specific atom is discovered at run time (the allocation, account, or install
base); the templates in the `.json` derive the rest. Every value it settles on — fully
expanded — is written to `~/.config/vizfold/vizfold.json`, so other tools can read where things
ended up instead of guessing.

### Adding a cluster

Two files in `install/sites/`, named after the cluster's SLURM `ClusterName`: `<name>.sh` — a
single `slurm::discover` that exports the one login-specific atom — and `<name>.json`, which
declares everything else and templates paths/accounts off that atom (and `$USER`). `vizfold
init` (via `install/init.sh`) dispatches on `ClusterName`, so nothing else needs to change.

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
See the [LICENSE](./LICENSE) file for details.

---
