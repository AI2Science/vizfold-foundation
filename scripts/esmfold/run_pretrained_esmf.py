#!/usr/bin/env python3
"""Run the ESMFold backend by path: `micromamba run -p $ESMFOLD_ENV_PREFIX python <this>`."""
import sys

from esmfold.cli import main

if __name__ == "__main__":
    sys.exit(main())
