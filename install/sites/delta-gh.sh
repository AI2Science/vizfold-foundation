#!/bin/bash
# NCSA Delta-AI (ClusterName "delta-gh"). Reached from ../../install.sh.
# Grace-Hopper (aarch64); setup.sh fetches an aarch64 micromamba. No CPU-only
# queue, so the build runs on a GH200 node. No mirror; params are downloaded.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# environment.yml pins mkl (x86-only) and deepspeed=*=cuda* (no aarch64 CUDA build),
# so the env cannot solve on Grace-Hopper yet. Refuse here rather than burn GH200
# hours on a build that dies at the solver. Needs an aarch64 environment variant.
[ "$(uname -m)" = aarch64 ] && die "Delta-AI is aarch64: environment.yml (mkl, deepspeed=cuda) has no aarch64 solve. Needs an aarch64 env variant (nomkl/openblas + deepspeed via pip)."

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
