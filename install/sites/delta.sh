#!/bin/bash
# NCSA Delta (ClusterName "delta"). Reached from ../../install.sh.
# Per-allocation project space and -delta-{cpu,gpu} accounts; AF2 mirror in <site>.json.
set -euo pipefail

REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
. "$REPO/install/hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# Project space is /work/nvme/<allocation>/<user>, which names the accounts too.
# Usable means both: a directory to install into and cpu+gpu accounts to charge --
# this cluster hands out space you cannot charge, and accounts with no space.
usable() {
    local dir alloc accounts
    accounts=$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | sort -u)
    for dir in /work/nvme/*/"$USER"; do
        [ -d "$dir" ] || continue
        alloc=$(basename "$(dirname "$dir")")
        grep -qx "$alloc-delta-cpu" <<<"$accounts" &&
            grep -qx "$alloc-delta-gpu" <<<"$accounts" && echo "$alloc"
    done
}
# Pick one and say which -- never a question -- preferring one that already holds
# an install, so re-running lands where the last one did.
allocation() {
    local u a
    u=$(usable); [ -n "$u" ] || return 1
    a=$(while read -r x; do
        [ -n "$x" ] && [ -d "/work/nvme/$x/$USER/openfold" ] && echo "$x"
    done <<<"$u" | head -1)
    echo "${a:-$(head -1 <<<"$u")}"
}

ALLOCATION=$(interactive::resolve OPENFOLD_ALLOCATION allocation "$(allocation || true)")
[ -n "$ALLOCATION" ] ||
    die "no usable allocation: need /work/nvme space and <alloc>-delta-{cpu,gpu} accounts"
export OPENFOLD_GPU_ACCOUNT=${OPENFOLD_GPU_ACCOUNT:-$ALLOCATION-delta-gpu}
hpc::submit "/work/nvme/$ALLOCATION/$USER/openfold" "$ALLOCATION-delta-cpu"
