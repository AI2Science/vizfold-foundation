#!/bin/bash
# Assertions for esmfold::present -- the gate that decides whether an env already here may stay, and
# so whether a re-install can repair a torch the driver cannot load. Run: bash tests/esmfold_env.sh
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); export REPO OPENFOLD_HOME=$REPO
cd "$REPO"
VIZFOLD_CONFIG=/nonexistent/vizfold-test.json; export VIZFOLD_CONFIG  # hermetic: no dev's real config
. "$REPO/backends/esmfold/install/install.sh"
set +e                                        # the installer's own set -e, not this harness's

SANDBOX=${TMPDIR:-/tmp}/vizfold-esmfold-env-$$
trap 'rm -rf "$SANDBOX"' EXIT
fail=0

# Stands in for the backend env's python: a torch whose CUDA build is $STUB_TORCH_CUDA, nothing else.
mkdir -p "$SANDBOX/bin"
cat > "$SANDBOX/bin/python" <<'STUB'
#!/bin/bash
exec python3 -c "
import sys, types
for name in ('transformers', 'esmfold'): sys.modules[name] = types.ModuleType(name)
torch = types.ModuleType('torch')
torch.version = types.SimpleNamespace(cuda='$STUB_TORCH_CUDA' or None)
sys.modules['torch'] = torch
exec(sys.argv[1])" "$2"
STUB
chmod +x "$SANDBOX/bin/python"
ENV=$SANDBOX

check() { # $1 driver, $2 the installed torch's CUDA, $3 want present, $4 name
    local got
    OPENFOLD_DRIVER_CUDA=$1 STUB_TORCH_CUDA=$2 esmfold::present && got=yes || got=no
    if [ "$got" = "$3" ]; then echo "ok   $4"; else
        echo "FAIL $4"; echo "  want present=$3"; echo "  got  present=$got"; fail=1
    fi
}

# The reported bug: Delta's 12.8 driver against the cu130 torch pip took off PyPI.
check 12.8 13.0 no  "a CUDA 13 torch under a 12.8 driver is rebuilt, not kept"
check 12.8 12.8 yes "a matching build is left alone"
check 12.8 12.9 yes "a newer minor is loadable, so no needless rebuild"
check 12.8 12.6 yes "an older build stays too"
check 13.0 13.0 yes "a 13.0 driver keeps its CUDA 13 torch"
check 12.8 ""   no  "a CPU-only torch under a real driver is rebuilt"
check ""   13.0 yes "an unknown driver gates nothing: any importable env counts as installed"
check ""   ""   yes "nor does it force CUDA where there is no driver at all"

specs() { # $1 driver, $2 want, $3 name
    local got
    got=$(OPENFOLD_DRIVER_CUDA=$1 esmfold::cuda_specs)
    if [ "$got" = "$2" ]; then echo "ok   $3"; else
        echo "FAIL $3"; echo "  want: $2"; echo "  got:  $got"; fail=1
    fi
}

specs 12.8 "pytorch=*=cuda* cuda-version<=12.8" "a driver asks for a GPU build capped at it"
specs 13.0 "pytorch=*=cuda* cuda-version<=13.0" "and follows the driver, whatever it reports"
specs ""   ""                                   "no driver asks for nothing: the solver picks CPU"

exit $fail
