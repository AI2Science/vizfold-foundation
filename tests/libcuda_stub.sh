#!/bin/bash
# Where the CUDA driver stub goes, and where it must not. Run: bash tests/libcuda_stub.sh
#
# Triton links `-lcuda` when it JITs its cuda_utils shim. The driver installs only libcuda.so.1, so
# on a node without the .so dev symlink the link fails and every fold dies before it starts. The
# stub fixes it -- but only on LIBRARY_PATH. On LD_LIBRARY_PATH it would load ahead of the real
# driver and the GPU would silently disappear, which is the worse bug of the two.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); export REPO OPENFOLD_HOME=$REPO
SANDBOX=${TMPDIR:-/tmp}/vizfold-libcuda-$$
trap 'rm -rf "$SANDBOX"' EXIT
fail=0

VIZFOLD_CONFIG=/nonexistent/vizfold-test.json; export VIZFOLD_CONFIG
. "$REPO/backends/openfold/install/setup.sh"
set +e                                        # the installer's own set -e, not this harness's
log() { :; }
mamba::activate() { :; }                      # no micromamba in CI; CONDA_PREFIX is set per case

# $1 case name, $2 relative path to plant the stub at (empty: plant none)
plant() {
    CASE=$1
    rm -rf "$SANDBOX"; mkdir -p "$SANDBOX/lib"
    export CONDA_PREFIX=$SANDBOX ENV_DIR=$SANDBOX CUTLASS=$SANDBOX/cutlass DATA=$SANDBOX/data
    [ -n "$2" ] && { mkdir -p "$SANDBOX/$(dirname "$2")"; : > "$SANDBOX/$2"; }
    setup::activate
    RC=$SANDBOX/etc/conda/activate.d/openfold.sh
}

check() { if [ "$2" = "$3" ]; then echo "ok   $4"; else
    echo "FAIL [$CASE] $4"; echo "  want $3"; echo "  got  $2"; fail=1; fi; }

line() { grep -m1 "^export $1=" "$RC" | sed "s/^export $1=//"; }

plant symlinked lib/stubs/libcuda.so
check . "$(setup::libcuda_stubs)" "$SANDBOX/lib/stubs" "the stub beside lib/ is found"
case "$(line LIBRARY_PATH)" in *"$SANDBOX/lib/stubs"*) echo "ok   and lands on LIBRARY_PATH";;
    *) echo "FAIL [$CASE] the stub is not on LIBRARY_PATH"; fail=1;; esac

# The arch dir is not always x86_64: DeltaAI is aarch64, and only the glob finds it there.
plant arch-nested targets/sbsa-linux/lib/stubs/libcuda.so
check . "$(setup::libcuda_stubs)" "$SANDBOX/targets/sbsa-linux/lib/stubs" "a targets/<arch>/ stub is found too"

plant absent ""
check . "$(setup::libcuda_stubs)" "" "no stub, no path"
check . "$(line LIBRARY_PATH)" '$CONDA_PREFIX/lib:${LIBRARY_PATH:-}' "and LIBRARY_PATH keeps its shape"

# The one that matters: loading the stub would cost the GPU, quietly.
plant never-runtime lib/stubs/libcuda.so
case "$(line LD_LIBRARY_PATH)" in *stubs*)
      echo "FAIL [$CASE] the stub reached LD_LIBRARY_PATH; folds would lose the GPU"; fail=1;;
    *) echo "ok   and never reaches LD_LIBRARY_PATH";; esac

exit $fail
