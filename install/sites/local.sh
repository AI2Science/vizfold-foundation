#!/bin/bash
# No batch scheduler. Reached from ../../install.sh, or run directly.
# setup.sh finds the checkout itself (via config.sh), so just hand off.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../setup.sh"
