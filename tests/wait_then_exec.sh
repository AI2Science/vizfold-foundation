#!/bin/bash
# The gap between `install repo` writing the checkout and a compute node being able to see it.
# Run: bash tests/wait_then_exec.sh
#
# $HOME is NFS on these clusters. srun execve'ing a script the node's attribute cache does not hold
# yet fails with a bare "No such file or directory" -- naming a file that is plainly there on the
# login node, which is exactly the report this guards against.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); export REPO OPENFOLD_HOME=$REPO
export VIZFOLD_CONFIG=/nonexistent/vizfold-test.json
. "$REPO/lib/slurm.sh"
set +e
SANDBOX=${TMPDIR:-/tmp}/vizfold-wait-$$; mkdir -p "$SANDBOX"
trap 'rm -rf "$SANDBOX"' EXIT
fail=0

check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else
    echo "FAIL $1"; echo "  want [$3]"; echo "  got  [$2]"; fail=1; fi; }

script=$SANDBOX/setup.sh
printf '#!/bin/bash\necho ran\n' > "$script"; chmod +x "$script"
check "a script already visible runs at once" \
    "$(bash -c "$(slurm::wait_then_exec)" _ "$script")" "ran"

# The reported failure: the file lands a moment after the step starts.
late=$SANDBOX/late.sh
( sleep 3; printf '#!/bin/bash\necho ran late\n' > "$late" ) &
check "a script that appears late is waited for" \
    "$(bash -c "$(slurm::wait_then_exec)" _ "$late")" "ran late"
wait

# It must not wait forever on a file that never lands -- that would hang an install with no output.
# `seq` is shadowed so the bound is reached in two ticks instead of sixty; the loop is what is under
# test, not the wall time.
out=$(bash -c "seq() { command seq 2; }; $(slurm::wait_then_exec)" _ "$SANDBOX/never.sh" 2>&1)
check "a file that never appears gives up rather than hanging" "$?" "127"

exit $fail
