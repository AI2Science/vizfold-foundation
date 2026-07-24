#!/bin/bash
# Assertions for slurm::launch_args. Run: bash install/tests/launch_args.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$(cd .. && pwd); export REPO
VIZFOLD_CONFIG=/nonexistent/vizfold-test.json; export VIZFOLD_CONFIG  # hermetic: no dev's real config, no config: line
. ./slurm.sh

fail=0
check() {
    local want=$1 got=$2 name=$3
    if [ "$want" = "$got" ]; then
        echo "ok   $name"
    else
        echo "FAIL $name"; echo "  want: $want"; echo "  got:  $got"; fail=1
    fi
}

# Already inside an srun step: run in place, never nest srun.
got=$(SLURM_STEP_ID=0 SLURM_JOB_ID=1 slurm::launch_args acct part --pty | tr '\n' ' ')
check "bash " "$got" "step id means bash"

# Holding an allocation but on the submit host: a bare step is enough.
got=$(SLURM_JOB_ID=1 slurm::launch_args acct part --pty | tr '\n' ' ')
check "srun --ntasks=1 " "$got" "job id means plain srun"

# No allocation: full srun with resources.
base="srun -u %s--job-name=vizfold-install --account=acct --partition=part --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G --time=02:00:00 "

got=$( (unset SLURM_STEP_ID SLURM_JOB_ID; slurm::launch_args acct part --pty) | tr '\n' ' ')
want=$(printf "$base" "--pty ")
check "$want" "$got" "no allocation means full srun with pty"

# Not a terminal: identical but without --pty.
got=$( (unset SLURM_STEP_ID SLURM_JOB_ID; slurm::launch_args acct part "") | tr '\n' ' ')
want=$(printf "$base" "")
check "$want" "$got" "no tty means no pty"

# sbatch must be gone entirely.
grep -q sbatch ./slurm.sh && { echo "FAIL sbatch still referenced"; fail=1; } || echo "ok   no sbatch"

exit $fail
