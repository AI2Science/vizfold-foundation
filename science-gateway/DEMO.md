# Running the OpenFold executor demo

This demo runs the OpenFold workflow through the `vizfold` CLI. It uses repo-aware defaults for the OpenFold script, FASTA input, precomputed alignments, and output directory.

The only required external path is the OpenFold data directory.

## 1. Expected directory layout

Place the `vizfold_data` directory one level above the `vizfold-foundation` repository:

```text
<workspace>/
  vizfold-foundation/
  vizfold_data/
````

For example:

```text
/home/<user>/
  vizfold-foundation/
  vizfold_data/
```

If you received `vizfold_data.zip`, extract it next to the repository:

```bash
cd /home/<user>
unzip vizfold_data.zip
```

After extraction, you should have:

```text
/home/<user>/vizfold_data
/home/<user>/vizfold-foundation
```

## 2. Activate the Python/OpenFold environment

Run the demo from an environment where OpenFold dependencies are installed.

For example:

```bash
conda activate vizfold
```

or, if using micromamba:

```bash
micromamba activate vizfold
```

You can quickly verify that PyTorch and CUDA are visible:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected on a GPU machine:

```text
True
```

## 3. Useful environment variable overrides

Most users only need `VIZFOLD_OPENFOLD_DATA_DIR`.

Optional overrides:

```bash
export VIZFOLD_OPENFOLD_DATA_DIR="/path/to/vizfold_data"
export VIZFOLD_OPENFOLD_MODEL_DEVICE="cuda:0"
export VIZFOLD_OPENFOLD_INPUT_ID="1UBQ_1"
export VIZFOLD_OPENFOLD_FASTA_DIR="/path/to/fasta_dir"
export VIZFOLD_OPENFOLD_ALIGNMENT_DIR="/path/to/alignments"
export VIZFOLD_OPENFOLD_OUTPUT_LOCATION="/path/to/output-root"
export VIZFOLD_OPENFOLD_RESIDUE_IDX="1"
export VIZFOLD_OPENFOLD_DEMO_ATTN="true"
```

If you override `VIZFOLD_OPENFOLD_INPUT_ID`, make sure the FASTA header and precomputed alignment directory match. For example, with:

```bash
export VIZFOLD_OPENFOLD_INPUT_ID="1UBQ_1"
```

the FASTA header should resolve to `1UBQ_1`, and precomputed alignments should exist at:

```text
alignment_dir/1UBQ_1
```

`VIZFOLD_OPENFOLD_OUTPUT_LOCATION` is the base output location. The workflow resolves the run workspace as `<output_location>/<run.id>`, passes it to OpenFold as `--output_dir`, and derives attention output under `<output_location>/<run.id>/attention`.

## 4. CLI workflow

The `vizfold` CLI is the development path for testing the full persisted workflow: seeded model/target/profile records, queued runs, execution status, and registered output artifacts.

Run these commands from the executor crate:

```bash
cd vizfold-foundation/science-gateway/apps/executor
```

### Set up the local output workspace

The seeded `local-openfold` profile uses the repository root as its working directory and writes runs below:

```text
<repo-root>/science-gateway/openfold-demo-output/<run-id>
```

Create the output parent before executing a run:

```bash
mkdir -p ../../../science-gateway/openfold-demo-output
```

In PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path ../../../science-gateway/openfold-demo-output
```

### Seed and inspect the local records

Seed is safe to repeat. It creates (or refreshes) the development OpenFold backend, `local-openfold` target, and matching `local_subprocess` invocation profile.

```bash
cargo run --bin vizfold -- seed
cargo run --bin vizfold -- list models
cargo run --bin vizfold -- list targets
cargo run --bin vizfold -- list profiles
```

Verify that the output includes:

* model backend `openfold`;
* execution target `local-openfold` with type `local`;
* an invocation profile that references those two IDs and has invocation kind `local_subprocess`.

The CLI uses `DATABASE_URL` when present; otherwise it uses the executor's local SQLite default. Set `DATABASE_URL` if you want a separate development database.

### Queue an OpenFold run

Unlike shell-relative paths, relative local paths supplied to `queue-run` are resolved from the seeded profile's repository-root `working_dir`. The following paths are therefore relative to `vizfold-foundation`, even though the command is executed from the executor crate.

```bash
cargo run --bin vizfold -- queue-run openfold \
  --input-id 6KWC_1 \
  --input-sequence GSTIQPGTGYNNGYFYSYWNDGHGGVTYTNGPGGQFSVNWSNSGEFVGGKGWQPGTKNKVINFSGSYNPNGNSYLSVYGWSRNPLIEYYIVENFGTYNPSTGATKLGEVTSDGSVYDIYRTQRVNQPSIIGTATFYQYWSVRRNHRSSGSVNTANHFNAWAQQGLTLGTMDYQIVAVQGYFSSGSASITVS \
  --fasta-dir examples/monomer/fasta_dir_6KWC \
  --data-dir ../vizfold_data \
  --alignment-dir examples/monomer/alignments \
  --model-device cuda:0 \
  --residue-idx 1 \
  --demo-attn \
  --use-precomputed-alignments
```

`--data-dir ../vizfold_data` matches the directory layout in this guide. All local FASTA, data, and alignment paths must exist when queuing; the CLI stores their canonical absolute paths in the run record.

The command prints a run ID. Inspect it before execution:

```bash
cargo run --bin vizfold -- list runs
cargo run --bin vizfold -- show run <run-id>
```

Confirm its status is `submitted`, its input ID is correct, and its backend/target/profile IDs point to the seeded OpenFold records.

### Execute, register artifacts, and inspect the result

Activate the Python/OpenFold environment first, as described above, then execute the queued run:

```bash
cargo run --bin vizfold -- execute-run <run-id>
```

The command prints every preflight check and either launches the configured OpenFold command or reports why execution was skipped. On a successful run, the output workspace is `<repo-root>/science-gateway/openfold-demo-output/<run-id>`.

Register the known output directories after execution:

```bash
cargo run --bin vizfold -- register-artifacts <run-id>
cargo run --bin vizfold -- register-artifacts <run-id>
cargo run --bin vizfold -- show run <run-id>
```

The first registration records the workspace as `run_output_directory` and, if present, its `attention` child as `attention_output_directory`. The second call is idempotent and reports those artifacts as already present. `show run` lists the resulting artifact IDs, types, formats, and storage paths.

Artifact registration does not block failed or incomplete runs: it prints a warning because available output may be partial, then registers only directories that actually exist.

## 5. Common failure modes

### `ModuleNotFoundError: No module named 'torch'`

The executor successfully launched Python, but it used a Python environment without PyTorch/OpenFold dependencies.

Activate the correct environment first:

```bash
micromamba activate vizfold
# or
conda activate vizfold
```

Then verify:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Missing `data_dir`

Set:

```bash
export VIZFOLD_OPENFOLD_DATA_DIR="$(realpath ../../../../vizfold_data)"
```

from:

```text
vizfold-foundation/science-gateway/apps/executor
```

### FASTA/input ID mismatch

The executor validates that the FASTA header-derived OpenFold tag matches the run `input_id`.

For the default demo:

```text
input_id = 1UBQ_1
```

So the FASTA header must resolve to:

```text
1UBQ_1
```

### Missing precomputed alignment key

If precomputed alignments are enabled, the executor expects:

```text
alignment_dir/input_id
```

For the default demo:

```text
examples/monomer/alignments/1UBQ_1
```

## 6. Notes

This demo does not automatically register newly generated alignments as reusable artifacts.

Precomputed alignment reuse currently requires that the expected alignment directory already exists on disk. Future work should index generated alignments as artifacts so later runs can select and reuse them through the executor/workbench flow.
