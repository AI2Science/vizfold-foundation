#!/usr/bin/env python3
"""Run the ESMFold backend by path, under the install's own interpreter
(`$ESMFOLD_ENV_PREFIX/bin/python`, which `vizfold status` prints).
"""
import sys

from esmfold.cli import main

if __name__ == "__main__":
    sys.exit(main())
