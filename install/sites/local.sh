#!/bin/bash

# No batch scheduler: hand off to setup.sh, which finds the checkout itself.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../setup.sh"
