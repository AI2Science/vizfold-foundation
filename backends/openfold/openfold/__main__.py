"""The OpenFold backend's own entrypoint: `openfold ...` or `python -m openfold ...` inside its
environment, which is the model's CLI with nothing in front of it.

Through vizfold, `queue-run` and `execute-run` are the way in -- they fill these arguments from
`~/.config/vizfold/vizfold.json` and put the run in the database. This is for driving the model
directly. `--help` is the runner's own, so it names run_pretrained_openfold.py.
"""
import runpy
import sys
from pathlib import Path

# <repo>/backends/openfold/openfold/__main__.py. The package is always an editable install of the
# checkout -- its CUDA extension has to build against the environment's own torch -- so the runner
# next to it is the one this environment was built for.
RUNNER = Path(__file__).resolve().parents[3] / "scripts/openfold/run_pretrained_openfold.py"


def main() -> int:
    if not RUNNER.is_file():
        print(f"Error: no runner at {RUNNER}; run `vizfold install openfold`", file=sys.stderr)
        return 1
    runpy.run_path(str(RUNNER), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
