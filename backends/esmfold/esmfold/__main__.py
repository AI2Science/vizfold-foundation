"""`python -m esmfold ...`, the same entrypoint the `esmfold` command runs."""
import sys

from esmfold.cli import main

if __name__ == "__main__":
    sys.exit(main())
