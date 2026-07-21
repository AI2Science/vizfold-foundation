#!/bin/bash
# No batch scheduler. Reached from ../../install.sh, or run directly.
set -euo pipefail
REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
exec bash "$REPO/install/setup.sh"
