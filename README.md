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
