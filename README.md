# Vizfold Foundations

This repository has two main components:

1. Model inference & feature extraction: Run protein structure prediction models and extract intermediate activations (hidden representations) and attention maps from any chosen layer.
2. Visualization & analysis: Explore, visualize, and analyze the extracted activations and attention maps.

---

Link to Openfold implimentation - [README_vizfold_openfold.md](https://github.com/vizfold/vizfold-foundation/blob/main/README_vizfold_openfold.md)

---

## Install

On a cluster, one command. It works out where it is running and needs nothing from you:

```bash
curl -sL https://raw.githubusercontent.com/AI2Science/vizfold-foundation/main/install.sh | bash
```

It clones a checkout, picks the site, submits itself to the scheduler, and prints the exact
command to fold a test sequence. Cold: ~8 min on NCSA Delta, ~25 min where the AlphaFold
databases have to be downloaded.

### Supported clusters

Dispatch is on the SLURM `ClusterName`, so on these machines the one command above needs no
arguments. Accounts and the install prefix are worked out live (your project space, the
accounts you can charge); the values below are what a fresh install settles on.

| `ClusterName` (cluster) | Arch | Install prefix | AF2 databases | Build → fold partition (GPU) | Verified |
| --- | --- | --- | --- | --- | --- |
| `delta` (NCSA Delta) | x86-64 | `/work/nvme/<alloc>/<user>/openfold` | mirror¹ | `cpu` → `gpuA100x4-interactive` (A100) | ✅ install + fold |
| `delta-gh` (NCSA Delta-AI) | aarch64 (GH200) | `/work/nvme/<alloc>/<user>/openfold-gh`² | downloaded | `ghx4` → `ghx4-interactive` (GH200) | ✅ install + fold³ |
| `nexus-dev` (Nexus) | x86-64 | `/projects/<user>/openfold` | downloaded | `gpu` → `gpu` (A100 10 GB vGPU)⁴ | ◐ install |
| `anvil` (Purdue Anvil) | x86-64 | `$PROJECT/<user>/openfold` | downloaded | `shared` → `gpu` (A100) | ⚙️ profile |
| `bridges2` (PSC Bridges-2) | x86-64 | `/ocean/projects/<acct>/<user>/openfold` | mirror¹ | `RM-shared` → `GPU-shared` (V100-32) | ⚙️ profile |
| `expanse` (SDSC Expanse) | x86-64 | `/expanse/lustre/projects/<acct>/<user>/openfold` | downloaded | `shared` → `gpu-shared` (V100) | ⚙️ profile |
| `ice-slurm` (GT PACE ICE) | x86-64 | `~/scratch` real root (`/storage/ice1/…`) | mirror¹ | `ice-cpu` → `ice-gpu` (A100) | ⚙️ profile |
| `phoenix-slurm` (GT PACE Phoenix) | x86-64 | `~/scratch` real root (`/storage/scratch1/…`) | downloaded | `cpu-small` → `gpu-a100` (A100) | ⚙️ profile |

Legend — ✅ install + fold verified end-to-end from the one command (fold `6KWC_1` → 2839-atom
relaxed structure); ◐ install verified; ⚙️ site profile written, full run still pending.

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

### Settings

Three layers, highest first. Each only fills what the one above left unset, so you override
exactly what you care about and nothing else:

| | | |
| --- | --- | --- |
| 1 | inline environment | `OPENFOLD_PREFIX=/scratch/me/openfold ... \| bash` |
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
OPENFOLD_EXAMPLE=1UBQ_1 OPENFOLD_PARTITION=cpuA100x4 \
  curl -sL https://raw.githubusercontent.com/AI2Science/vizfold-foundation/main/install.sh | bash
```

Paths and accounts are worked out per site and are not in these files: the install prefix
comes from your project space, and the SLURM accounts from the ones you can actually charge.
Every value it settles on is written to `~/.config/vizfold/vizfold.json`, so other tools can
read where things ended up instead of guessing.

### Adding a cluster

Two files in `install/sites/`, named after the cluster's SLURM `ClusterName`: `<name>.sh` for
what has to be computed (prefix, accounts, launcher) and `<name>.json` for what is just a
value. `install.sh` dispatches on `ClusterName`, so nothing else needs to change.

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
See the [LICENSE](./LICENSE) file for details.

---
