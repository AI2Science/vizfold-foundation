# OpenFold demo environment

The Rust executor OpenFold demo uses a local OpenFold-compatible Python environment.

The included `environment.yml` is based on the OpenFold environment definition and is intended as a starting point for reproducing the local demo setup. It is not yet a fully managed VizFold installer.

## 1. Create the environment

From the repository root:

```bash
micromamba env create -f science-gateway/openfold-demo/environment.openfold.yml
micromamba activate openfold-env
````

or with conda:

```bash
conda env create -f science-gateway/openfold-demo/environment.openfold.yml
conda activate openfold-env
```


## 2. Install/build OpenFold

Follow the [OpenFold installation](https://openfold.readthedocs.io/en/latest/Installation.html) steps after creating the environment. The VizFold executor demo assumes that OpenFold can be run from the repository checkout and that its Python dependencies and alignment binaries are available in the active environment.

## 3. Smoke checks

After activating the environment, verify:

```bash
python3 -c "import torch; import attn_core_inplace_cuda; print(torch.__version__); print(torch.cuda.is_available())"
which jackhmmer
which hhblits
which hhsearch
which kalign || which kalign2
```

For the GPU demo, `torch.cuda.is_available()` should print `True`.

## 4. Prepare demo data

The demo expects the OpenFold data directory to be available outside the repo. If using the provided `vizfold_data` zip, extract it one level above the repository root:

```text
<workspace>/
  vizfold-foundation/
  vizfold_data/
```

For example:

```bash
cd <workspace>
unzip vizfold_data.zip
```

## 5. Run the OpenFold workflow example

From the executor directory:

```bash
cd science-gateway/apps/executor
```

Set the required environment variables:

```bash
export VIZFOLD_OPENFOLD_DATA_DIR="$(realpath ../../../../vizfold_data)"
export VIZFOLD_OPENFOLD_MODEL_DEVICE="cuda:0"
```

Then run:

```bash
cargo run --example run_openfold_workflow
```

The example uses repo-aware defaults for the OpenFold script, FASTA input, precomputed alignments, output directory, and attention output directory.

## 6. Optional environment variable overrides

Most users only need `VIZFOLD_OPENFOLD_DATA_DIR`.

Useful overrides:

```bash
export VIZFOLD_OPENFOLD_DEMO_RUN_ID="openfold-demo-run"
export VIZFOLD_OPENFOLD_INPUT_ID="6KWC_1"
export VIZFOLD_OPENFOLD_FASTA_DIR="/path/to/fasta_dir"
export VIZFOLD_OPENFOLD_ALIGNMENT_DIR="/path/to/alignments"
export VIZFOLD_OPENFOLD_OUTPUT_DIR="/path/to/output"
export VIZFOLD_OPENFOLD_ATTN_MAP_DIR="/path/to/attention-output"
export VIZFOLD_OPENFOLD_RESIDUE_IDX="1"
export VIZFOLD_OPENFOLD_DEMO_ATTN="true"
export VIZFOLD_OPENFOLD_MODEL_DEVICE="cuda:0"
```

If overriding `VIZFOLD_OPENFOLD_INPUT_ID`, make sure the FASTA header and precomputed alignment directory match. For example, with:

```bash
export VIZFOLD_OPENFOLD_INPUT_ID="6KWC_1"
```

the FASTA header should resolve to `6KWC_1`, and precomputed alignments should exist at:

```text
<alignment_dir>/6KWC_1
```

## 7. Register known demo artifacts

After the workflow runs, register the known output directories as artifacts:

```bash
cargo run --example register_openfold_demo_artifacts
```

This example indexes known output locations into the executor database using `ARTIFACT` and `ARTIFACT_TYPE`.

It does not recursively scan all files, generate visualizations, serve files, or register reused precomputed alignments as produced artifacts.

## 8. Notes

For local execution, researchers still need a working OpenFold environment. This is a local-execution limitation, not a core executor limitation.

Future Docker, HPC, or Science Gateway execution targets should make this easier by providing or managing the model environment.