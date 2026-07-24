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

## 5. Useful queue-run flags

Every OpenFold-specific input (input ID, FASTA/alignment directories, model device, residue index, `--demo-attn`) is a `queue-run openfold` flag -- see DEMO.md for the full command. The one environment override most users need is the data directory, consulted when `--data-dir` is omitted:

```bash
export OPENFOLD_DATA_DIR="/path/to/vizfold_data"
```

If you pass `--input-id`, make sure the FASTA header and precomputed alignment directory match. For example, with `--input-id 6KWC_1`, the FASTA header should resolve to `6KWC_1`, and precomputed alignments should exist at:

```text
<alignment_dir>/6KWC_1
```

The output location is not a per-run flag: it is written to `ModelInvocationProfile.config_json.output_location` by `vizfold seed`. The run workspace is `<output_location>/<run.id>`; OpenFold receives it as `--output_dir`, and attention output is derived under `<output_location>/<run.id>/attention`.

## 6. Notes

For local execution, researchers still need a working OpenFold environment. This is a local-execution limitation, not a core executor limitation.

Future Docker, HPC, or Science Gateway execution targets should make this easier by providing or managing the model environment.
