#!/bin/bash
# Assertions for the torch build ESMFold installs: the wheel index a driver can load, and the
# already-installed build it refuses to keep. Run: bash tests/torch_index.sh
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); export REPO OPENFOLD_HOME=$REPO
cd "$REPO"
VIZFOLD_CONFIG=/nonexistent/vizfold-test.json; export VIZFOLD_CONFIG  # hermetic: no dev's real config
. "$REPO/backends/esmfold/install/install.sh"
set +e                                        # the installer's own set -e, not this harness's

WHL=https://download.pytorch.org/whl
SANDBOX=${TMPDIR:-/tmp}/vizfold-torch-index-$$
trap 'rm -rf "$SANDBOX"' EXIT

fail=0
check() {
    local want=$1 got=$2 name=$3
    if [ "$want" = "$got" ]; then
        echo "ok   $name"
    else
        echo "FAIL $name"; echo "  want: $want"; echo "  got:  $got"; fail=1
    fi
}

# Stubbed, so the pick is the machine's driver on no CI runner and a GPU workstation alike.
vizfold::driver_cuda() { echo "${STUB_DRIVER:-}"; }

pin() { # $1 driver, $2.. inline overrides -- prints the index chosen and the gate present() applies
    ( export STUB_DRIVER=$1; shift
      unset OPENFOLD_DRIVER_CUDA
      for kv in "$@"; do export "${kv?}"; done
      esmfold::torch_cuda >/dev/null
      echo "${ESMFOLD_PIP_INDEX_URL:-<none>} gate=$TORCH_CUDA" )
}

check "$WHL/cu128 gate=128" "$(pin 12.8)" "Delta's 12.8 driver takes the cu128 wheel, not PyPI's newest"
check "$WHL/cu130 gate=130" "$(pin 13.0)" "a 13.0 driver takes cu130"
check "$WHL/cu121 gate=121" "$(pin 12.2)" "a driver between published indexes rounds down"
check "<none> gate=0"       "$(pin '')"   "no driver pins nothing: a CPU install still resolves"
check "https://mirror/simple gate=128" "$(pin 12.8 ESMFOLD_PIP_INDEX_URL=https://mirror/simple)" \
    "an explicit index wins, and the driver still gates what may stay installed"
check "<none> gate=0" "$(pin 12.8 ESMFOLD_PIP_INDEX_URL=)" "an empty index opts out of the pin"

# present() against a stub env: only torch.version.cuda decides, so the fix reaches an install already here.
mkdir -p "$SANDBOX/bin"
cat > "$SANDBOX/bin/python" <<'STUB'
#!/bin/bash
exec python3 -c "
import sys, types
for name in ('transformers', 'esmfold'): sys.modules[name] = types.ModuleType(name)
torch = types.ModuleType('torch')
torch.version = types.SimpleNamespace(cuda='$STUB_TORCH_CUDA')
sys.modules['torch'] = torch
exec(sys.argv[1])" "$2"
STUB
chmod +x "$SANDBOX/bin/python"
ENV=$SANDBOX

TORCH_CUDA=128
STUB_TORCH_CUDA=13.0 esmfold::present && { echo "FAIL a cu130 torch under a 12.8 driver counts as installed"; fail=1; } ||
    echo "ok   a cu130 torch under a 12.8 driver is reinstalled, not kept"
STUB_TORCH_CUDA=12.8 esmfold::present && echo "ok   a matching build is left alone" ||
    { echo "FAIL a cu128 torch under a 12.8 driver was not accepted"; fail=1; }
STUB_TORCH_CUDA=12.6 esmfold::present && echo "ok   an older build is still loadable, so it stays" ||
    { echo "FAIL a cu126 torch under a 12.8 driver was needlessly rebuilt"; fail=1; }

TORCH_CUDA=0
STUB_TORCH_CUDA=13.0 esmfold::present && echo "ok   with no pin any importable build counts as installed" ||
    { echo "FAIL an unpinned install rebuilt a working env"; fail=1; }

exit $fail
