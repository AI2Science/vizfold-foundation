#!/bin/bash
# Assertions for slurm::launch_args and slurm::cluster. Run: bash install/tests/launch_args.sh
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

# slurm::cluster: each source in turn, because no single one works everywhere. Delta's login nodes
# cannot reach slurmctld (scontrol exits 1, empty), DeltaAI has no readable slurm.conf.
conf=${TMPDIR:-/tmp}/vizfold-slurm-conf.$$
printf 'ControlMachine=x\nClusterName=Delta\nSelectType=cray\n' > "$conf"
trap 'rm -f "$conf"' EXIT
scontrol() { return 1; }                                  # controller unreachable, as on dt-login02

got=$(SLURM_CLUSTER_NAME=delta-gh SLURM_CONF=$conf slurm::cluster)
check "delta-gh" "$got" "a job's own cluster name wins"

got=$(SLURM_CONF=$conf slurm::cluster)
check "delta" "$got" "slurm.conf answers when the controller does not, lower-cased"

scontrol() { echo "ClusterName             = delta-gh"; }
got=$(SLURM_CONF=$conf slurm::cluster)
check "delta-gh" "$got" "the controller outranks slurm.conf"

scontrol() { return 1; }
got=$(SLURM_CONF=/nonexistent slurm::cluster)
check "" "$got" "no source means no site, so install.sh falls back to local"

# Every name slurm::cluster can resolve must have a site file, or detection finds nothing to load.
for c in delta delta-gh; do
    [ -f "sites/$c.sh" ] && echo "ok   sites/$c.sh exists" || { echo "FAIL no sites/$c.sh"; fail=1; }
done

# sbatch must be gone entirely.
grep -q sbatch ./slurm.sh && { echo "FAIL sbatch still referenced"; fail=1; } || echo "ok   no sbatch"

# A saved config must not pin the site. Sourcing the libs no longer loads it, so what an earlier
# install settled arrives under the site's own defaults instead of ahead of live detection --
# otherwise an OPENFOLD_SITE written on a login node where scontrol failed ("local") sticks forever.
# A fresh bash each time: config.sh guards against re-sourcing, so testing this in a subshell of a
# process that already sourced it would pass no matter what the file does.
saved=${TMPDIR:-/tmp}/vizfold-saved-config.$$
printf '{"OPENFOLD_SITE": "local", "OPENFOLD_PREFIX": "/invented/by/an/earlier/install"}\n' > "$saved"
trap 'rm -f "$conf" "$saved"' EXIT
got=$(env -u OPENFOLD_SITE -u OPENFOLD_PREFIX VIZFOLD_CONFIG=$saved \
    bash -c '. "$1"; echo "${OPENFOLD_SITE:-<unset>} ${OPENFOLD_PREFIX:-<unset>}"' bash "$REPO/lib/config.sh")
check "<unset> <unset>" "$got" "sourcing the libs does not pin the site or prefix from a saved config"

got=$(env -u OPENFOLD_SITE -u OPENFOLD_PREFIX VIZFOLD_CONFIG=$saved \
    bash -c '. "$1"; config::load 2>/dev/null; echo "${OPENFOLD_SITE:-<unset>} ${OPENFOLD_PREFIX:-<unset>}"' \
    bash "$REPO/lib/config.sh")
check "local /invented/by/an/earlier/install" "$got" "an explicit config::load still supplies it"

exit $fail
