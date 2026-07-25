#!/usr/bin/env python3
"""Run the ESMFold backend by path, for the executor and for anyone with the checkout in hand.

The entrypoint itself lives in the installed package (`esmfold.cli`, also on PATH inside the
environment as `vizfold-esmfold`) so that folding needs nothing outside it. Run this with the
install's own interpreter -- `$ESMFOLD_ENV_PREFIX/bin/python`, which `vizfold status` prints.
"""
import sys

from esmfold.cli import main

if __name__ == "__main__":
    sys.exit(main())
