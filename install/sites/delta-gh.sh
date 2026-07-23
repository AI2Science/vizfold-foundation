#!/bin/bash
# NCSA Delta-AI (ClusterName "delta-gh"). Reached from ../../install.sh.
# Grace-Hopper (aarch64); setup.sh fetches an aarch64 micromamba. No CPU-only
# queue, so the build runs on a GH200 node. No mirror; params are downloaded.
set -euo pipefail

REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
. "$REPO/install/hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# Delta-AI shares Delta's /work/nvme; accounts are <allocation>-dtai-gh.
usable() {
    local dir alloc accounts
    accounts=$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | sort -u)
    for dir in /work/nvme/*/"$USER"; do
        [ -d "$dir" ] || continue
        alloc=$(basename "$(dirname "$dir")")
        grep -qx "$alloc-dtai-gh" <<<"$accounts" && echo "$alloc"
    done
}
allocation() {
    local u a
    u=$(usable); [ -n "$u" ] || return 1
    a=$(while read -r x; do
        [ -n "$x" ] && [ -d "/work/nvme/$x/$USER/openfold" ] && echo "$x"
    done <<<"$u" | head -1)
    echo "${a:-$(head -1 <<<"$u")}"
}

ALLOCATION=$(interactive::resolve OPENFOLD_ALLOCATION allocation "$(allocation || true)")
[ -n "$ALLOCATION" ] || die "no usable allocation: need /work/nvme space and <alloc>-dtai-gh account"
hpc::submit "/work/nvme/$ALLOCATION/$USER/openfold" "$ALLOCATION-dtai-gh"
