#!/bin/bash

# NCSA Delta ("delta"). Per-allocation /work/nvme + -delta-{cpu,gpu} accounts; AF2 mirror in <site>.json.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# Usable = both a /work/nvme/<alloc>/<user> dir and chargeable cpu+gpu accounts (this cluster hands out one without the other).
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
# Pick one (never a prompt), preferring one that already holds an install so a re-run lands there.
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
