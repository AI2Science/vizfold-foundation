# Vizfold Foundations

Vizfold is a platform for running protein-structure models and inspecting what they compute:

1. Model inference & feature extraction: Run protein structure prediction models and extract intermediate activations (hidden representations) and attention maps from any chosen layer.
2. Visualization & analysis: Explore, visualize, and analyze the extracted activations and attention maps.

The `vizfold` CLI is the platform; a model backend plugs in underneath it. **OpenFold** is
today's default (and only) backend, installed by `vizfold init`; the same `install/` scripts
are built to host others (openfold3, boltz, esmfold) as they land.

---

Link to Openfold implimentation - [README_vizfold_openfold.md](https://github.com/vizfold/vizfold-foundation/blob/main/README_vizfold_openfold.md)

---

## Install

Two steps on a cluster. First bootstrap the `vizfold` CLI — one command, needs nothing from you:

```bash
curl -sL https://raw.githubusercontent.com/AI2Science/vizfold-foundation/main/install.sh | bash
```

That clones a checkout, builds the `vizfold` binary, and installs it to `~/.local/bin`. Then
install the OpenFold backend:

```bash
vizfold init
```

`vizfold init` works out where it is running, picks the site, submits the OpenFold install to
the scheduler, and prints the exact command to fold a test sequence. Cold: ~8 min on NCSA
Delta, ~25 min where the AlphaFold databases have to be downloaded.

### Supported clusters

Dispatch is on the SLURM `ClusterName`, so on these machines `vizfold init` needs no
arguments. Accounts and the install prefix are worked out live (your project space, the
accounts you can charge); the values below are what a fresh install settles on.

| `ClusterName` (cluster) | Verified | Arch | AF2 databases | Build → fold partition (GPU) | Install prefix |
| --- | --- | --- | --- | --- | --- |
| `delta` (NCSA Delta) | ✅ install + fold | x86-64 | mirror¹ | `cpu` → `gpuA100x4-interactive` (A100) | `/work/nvme/<alloc>/<user>/openfold` |
| `delta-gh` (NCSA Delta-AI) | ✅ install + fold³ | aarch64 (GH200) | downloaded | `ghx4` → `ghx4-interactive` (GH200) | `/work/nvme/<alloc>/<user>/openfold-gh`² |
| `nexus-dev` (Nexus) | ◐ install⁵ | x86-64 | downloaded | `gpu` → `gpu` (A100 10 GB vGPU)⁴ | `/projects/<user>/openfold` |
| `anvil` (Purdue Anvil) | ◐ install⁵ | x86-64 | downloaded | `shared` → `gpu` (A100) | `$PROJECT/<user>/openfold` |
| `bridges2` (PSC Bridges-2) | ◐ install⁵ | x86-64 | mirror¹ | `RM-shared` → `GPU-shared` (V100-32) | `/ocean/projects/<acct>/<user>/openfold` |
| `expanse` (SDSC Expanse) | ⚙️ profile | x86-64 | downloaded | `shared` → `gpu-shared` (V100) | `/expanse/lustre/projects/<acct>/<user>/openfold` |
| `ice-slurm` (GT PACE ICE) | ⚙️ profile | x86-64 | mirror¹ | `ice-cpu` → `ice-gpu` (A100) | `~/scratch` real root (`/storage/ice1/…`) |
| `phoenix-slurm` (GT PACE Phoenix) | ⚙️ profile | x86-64 | downloaded | `cpu-small` → `gpu-a100` (A100) | `~/scratch` real root (`/storage/scratch1/…`) |

Legend — ✅ install + fold verified end-to-end from `vizfold init` (fold → 2839-atom relaxed
structure); ◐ install run on the cluster with its site-specific fixes, final fold not re-confirmed
in this pass⁵; ⚙️ site profile written and its paths probed live, full install not yet run.

1. AF2 mirrors: Delta `/sw/external/alphafold2/data_hyun_official`, Bridges-2
   `/ocean/datasets/community/alphafold/v2.3.2`, ICE `/storage/ice1/shared/d-pace_community/…`.
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
| 1 | inline environment | `OPENFOLD_PREFIX=/scratch/me/openfold vizfold init` |
| 2 | `~/.config/vizfold/vizfold.json` | written by the install; edit to make a choice stick |
| 3 | `install/sites/<site>.json` | the site's defaults, in the repo — edit to change them for everyone |

`install/sites/delta.json`:

```json
{
  "OPENFOLD_AF2_ROOT": "/sw/external/alphafold2/data_hyun_official",
  "OPENFOLD_EXAMPLE": "6KWC_1",
  "OPENFOLD_GPU_PARTITION": "gpuA100x4-interactive",
  "OPENFOLD_GPU_RESOURCES": "--cpus-per-task=8 --mem=32G",
  "OPENFOLD_MAX_CUDA": "12.8",
  "OPENFOLD_PARTITION": "cpu"
}
```

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
OPENFOLD_EXAMPLE=1UBQ_1 OPENFOLD_PARTITION=cpuA100x4 vizfold init
```

Paths and accounts are worked out per site and are not in these files: the install prefix
comes from your project space, and the SLURM accounts from the ones you can actually charge.
Every value it settles on is written to `~/.config/vizfold/vizfold.json`, so other tools can
read where things ended up instead of guessing.

### Adding a cluster

Two files in `install/sites/`, named after the cluster's SLURM `ClusterName`: `<name>.sh` for
what has to be computed (prefix, accounts, launcher) and `<name>.json` for what is just a
value. `vizfold init` (via `install/init.sh`) dispatches on `ClusterName`, so nothing else
needs to change.

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
See the [LICENSE](./LICENSE) file for details.

---
