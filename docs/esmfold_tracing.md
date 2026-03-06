# ESMFold Tracing (Prototype)

This repo includes a prototype script to run ESMFold inference and capture
layer-wise activations with forward hooks.

## Environment notes

ESMFold in `fair-esm` is most reliable on Python 3.10. (Python 3.11+ may
error due to dataclass default validation in older ESMFold code.)

## Setup

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install torch fair-esm numpy omegaconf ml-collections dm-tree biopython scipy einops tqdm modelcif

python -m pip install -U setuptools wheel
python -m pip install -e . --no-build-isolation