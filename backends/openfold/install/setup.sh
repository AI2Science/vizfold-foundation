#!/bin/bash

# Install OpenFold into a micromamba env on the node that runs this. Idempotent per step.
set -euo pipefail

# sbatch spools this, breaking BASH_SOURCE; OPENFOLD_HOME (site-exported) finds the libs under
# the backend subtree. The BASH_SOURCE fallback covers a direct local run (libs are siblings here).
LIB=${OPENFOLD_HOME:+$OPENFOLD_HOME/lib}
. "${LIB:-$(dirname "${BASH_SOURCE[0]}")/../../../lib}/config.sh"

log()    { echo "== $* (+$((SECONDS))s)"; }
have()   { test -e "$1" || compgen -G "${1}_*.ffindex" >/dev/null; }   # ffindex sets are prefixes
sealed() { [ -e "$sentinel/$1" ]; }
seal()   { mkdir -p "$sentinel"; touch "$sentinel/$1"; }
older()  { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$1" ] && [ "$1" != "$2" ]; }
# A step seals only after finishing, so an interrupted create/clone/download is redone next run, not skipped.
step()   { log "$1"; sealed "$1" && { echo "  cached"; return; }; "$2"; seal "$1"; }

setup::config() {
    PREFIX=${OPENFOLD_PREFIX:-$HOME/openfold}
    AF2=${OPENFOLD_AF2_ROOT:-}                       # set by a site with a database mirror
    ENV_NAME=${OPENFOLD_ENV_NAME:-openfold-env}
    # aarch64 (Grace-Hopper) needs its own env: py3.13, GH200-only sm_90, cuda<=12.9 -- the 13.x aarch64 pytorch build won't compile OpenFold's extension.
    case $(uname -m) in
        aarch64|arm64) ENV_YML=$OF/environment-aarch64.yml; ARCH_DEFAULT=9.0; MAX_CUDA=${OPENFOLD_MAX_CUDA:-12.9} ;;
        *)             ENV_YML=$OF/environment.yml; ARCH_DEFAULT="7.0;8.0;8.6;9.0"; MAX_CUDA=${OPENFOLD_MAX_CUDA:-12.8} ;;
    esac
    DATA=$PREFIX/data
    ENV_DIR=$PREFIX/mamba/envs/$ENV_NAME
    MM=$PREFIX/bin/micromamba
    CUTLASS=$PREFIX/cutlass
    UNICLUST=$DATA/uniclust30/uniclust30_2018_08
    STEREO=$OF/openfold/resources/stereo_chemical_props.txt
    sentinel=$PREFIX/.done

    export CONDA_PKGS_DIRS=${OPENFOLD_PKGS_DIR:-$PREFIX/../.openfold-pkgs}
    export MAMBA_ROOT_PREFIX=$PREFIX/mamba TMPDIR=$PREFIX/tmp
    export PIP_CACHE_DIR=$PREFIX/../.openfold-pip
    export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
    # Every GPU these sites schedule (7.0 V100 .. 9.0 H100); a missing arch = "no kernel image".
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$ARCH_DEFAULT}"
    # A CPU build node exposes no __cuda; without this, conda resolves pytorch-gpu to junk.
    export CONDA_OVERRIDE_CUDA=${OPENFOLD_MAX_CUDA:-12.8}

    MIRROR=$([ -n "$AF2" ] && [ -d "$AF2" ] && echo yes || echo no)
}

setup::preflight() {
    mkdir -p "$PREFIX/bin" "$TMPDIR" "$DATA" "$OF/openfold/resources"
    hostname
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo "no GPU on this node"
    echo "prefix=$PREFIX repo=$REPO env=$ENV_NAME max_cuda=$MAX_CUDA mirror=$MIRROR${AF2:+ ($AF2)}"
    test -f "$OF/setup.py" || die "$REPO is not an OpenFold checkout"
}

setup::micromamba() {
    local mm_arch; case $(uname -m) in aarch64|arm64) mm_arch=linux-aarch64 ;; *) mm_arch=linux-64 ;; esac
    curl -Ls "https://micro.mamba.pm/api/micromamba/$mm_arch/latest" | tar -xj -C "$PREFIX" bin/micromamba
}

# By path + --no-rc so a ~/.condarc envs_dirs/channels can't hijack a reproducible env.
setup::env() {
    rm -rf "$ENV_DIR"   # clear a partial env; create fails on a non-empty dir
    "$MM" create -y --no-rc -p "$ENV_DIR" -f "$ENV_YML" "cuda-version<=$MAX_CUDA"
}

setup::activate() {
    log activate
    mamba::activate "$MM" "$ENV_DIR"
    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
    cat > "$CONDA_PREFIX/etc/conda/activate.d/openfold.sh" <<ACTIVATE
export CUTLASS_PATH=$CUTLASS
export KMP_AFFINITY=none
export LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}
ACTIVATE
    . "$CONDA_PREFIX/etc/conda/activate.d/openfold.sh"
}

setup::cutlass() {
    rm -rf "$CUTLASS"
    git clone -q https://github.com/NVIDIA/cutlass --branch v3.6.0 --depth 1 "$CUTLASS"
}

# OpenMM JITs via NVRTC; older driver rejects newer PTX. Detect whether the env's NVRTC outruns the driver.
setup::nvrtc_detect() {
    log nvrtc
    DRIVER_CUDA=${OPENFOLD_DRIVER_CUDA:-$(python3 -c "
import ctypes
v = ctypes.c_int()
ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(v))
print(f'{v.value // 1000}.{v.value % 1000 // 10}')" 2>/dev/null)} || true
    # CPU build node can't probe the GPU driver; assume old (12.2 PTX loads on any >=12.2).
    : "${DRIVER_CUDA:=${OPENFOLD_FALLBACK_CUDA:-12.2}}"
    ENV_CUDA=$(ls "$CONDA_PREFIX"/lib/libnvrtc.so.*.*.* 2>/dev/null |
        sed 's/.*so\.//; s/\.[0-9]*$//' | head -1) || true
    export OPENFOLD_DRIVER_CUDA=$DRIVER_CUDA
    if [ -n "${DRIVER_CUDA:-}" ] && [ -n "${ENV_CUDA:-}" ] && older "$DRIVER_CUDA" "$ENV_CUDA"; then
        return 0
    fi
    echo "driver CUDA ${DRIVER_CUDA:-unknown}, NVRTC ${ENV_CUDA:-unknown}; no pin needed"
    return 1
}

# Pin NVRTC beside the env, LD_PRELOAD (beats its RPATH), so OpenMM emits PTX the driver accepts.
setup::nvrtc_create() {
    local nvrtc=$PREFIX/nvrtc-$DRIVER_CUDA
    rm -rf "$nvrtc"
    "$MM" create -y --no-rc -p "$nvrtc" -c conda-forge "cuda-nvrtc<=$DRIVER_CUDA"
}

# Append every run: setup::activate rewrites openfold.sh from scratch, so this must not sit behind the create's sentinel.
setup::nvrtc_preload() {
    local lib; lib=$(ls "$PREFIX/nvrtc-$DRIVER_CUDA"/lib/libnvrtc.so.* 2>/dev/null | sort -V | tail -1)
    test -n "$lib" || die "no libnvrtc in $PREFIX/nvrtc-$DRIVER_CUDA"
    echo "export LD_PRELOAD=$lib\${LD_PRELOAD:+:\$LD_PRELOAD}" \
        >> "$CONDA_PREFIX/etc/conda/activate.d/openfold.sh"
    . "$CONDA_PREFIX/etc/conda/activate.d/openfold.sh"
    echo "driver CUDA $DRIVER_CUDA is older than NVRTC $ENV_CUDA; preloading ${lib##*/}"
}

setup::openfold_present() { python3 -c 'import torch, openfold, attn_core_inplace_cuda' 2>/dev/null; }

# Curated env -i drops a site's leaked CC/CXX/CPATH (break nvcc); CC/CXX = this env's gcc 12, not RHEL8's gcc 8.
setup::build_openfold() {
    log openfold
    local cc cxx; cc=$(echo "$CONDA_PREFIX"/bin/*-conda-linux-gnu-gcc); cxx=$(echo "$CONDA_PREFIX"/bin/*-conda-linux-gnu-g++)
    env -i HOME="$HOME" PATH="$CONDA_PREFIX/bin:/usr/bin:/bin" \
        CC="$cc" CXX="$cxx" \
        CUDA_HOME="$CONDA_PREFIX" TMPDIR="$TMPDIR" MAX_JOBS="$MAX_JOBS" \
        TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
        pip install --no-build-isolation -e "$OF"
}

setup::link_mirror() {
    log datasets
    for d in "$AF2"/*; do
        [ "${d##*/}" = uniclust30 ] && continue
        ln -sfn "$d" "$DATA/"
    done
    ln -sfn "$AF2/params" "$OF/openfold/resources/params"
    # uniclust30_2018_08 goes in a writable canonical dir (not the read-only mirror symlink): the mirror's real set if present (single- or double-nested), else aliased from uniref30.
    mkdir -p "$UNICLUST"
    local src="" c
    for c in "$AF2/uniclust30/uniclust30_2018_08" "$AF2/uniclust30"; do
        compgen -G "$c/uniclust30_2018_08_*" >/dev/null 2>&1 && { src=$c; break; }
    done
    if [ -n "$src" ]; then
        ln -sfn "$src"/uniclust30_2018_08_* "$UNICLUST/"
    else
        for f in "$AF2"/uniref30/UniRef30_[0-9][0-9][0-9][0-9]_[0-9][0-9]*; do
            [ -e "$f" ] || continue
            ln -sfn "$f" "$UNICLUST/uniclust30_2018_08${f##*/UniRef30_[0-9][0-9][0-9][0-9]_[0-9][0-9]}"
        done
    fi
}

# No mirror: fetch params (4 GB, into the prefix) and the mmCIFs the examples cite.
setup::fetch_params() {
    rm -rf "$PREFIX/params"   # a half-extracted tar would pass a single-file check
    bash "$REPO/downloaders/openfold/download_alphafold_params.sh" "$PREFIX"
}

setup::fetch_templates() {
    log templates
    ln -sfn "$PREFIX/params" "$OF/openfold/resources/params"
    mkdir -p "$DATA/pdb_mmcif/mmcif_files"
    # env -u LD_LIBRARY_PATH: else system curl binds conda's feature-poor libcurl and fails. || true tolerates a 404; assert catches total failure.
    grep -ohE "^ *[0-9]+ [0-9A-Za-z]{4}_" "$REPO"/examples/monomer/alignments/*/*.hhr |
        awk '{ print tolower(substr($2, 1, 4)) }' | sort -u |
        xargs -P 8 -I{} sh -c \
            '[ -s "$1/{}.cif" ] || env -u LD_LIBRARY_PATH curl -Lsf -o "$1/{}.cif" https://files.rcsb.org/download/{}.cif' _ \
            "$DATA/pdb_mmcif/mmcif_files" || true
    local n; n=$(ls "$DATA/pdb_mmcif/mmcif_files" | wc -l)
    echo "fetched $n template mmCIFs"
    [ "$n" -gt 0 ] || die "no template mmCIFs fetched; check outbound https from the compute node"
}

setup::stereo() {
    [ -f "$STEREO" ] || { env -u LD_LIBRARY_PATH curl -Lsf -o "$STEREO.part" \
        https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt &&
        mv "$STEREO.part" "$STEREO"; }
    mkdir -p "$OF/tests/test_data/alphafold/common"
    ln -sfn "$STEREO" "$OF/tests/test_data/alphafold/common/stereo_chemical_props.txt"
}

setup::verify() {
    log verify
    python3 - <<'PY'
import importlib.util as util, os, torch, attn_core_inplace_cuda, openfold
from openfold.model.primitives import Linear
print("torch", torch.__version__, "cuda_devices", torch.cuda.device_count())
print("openfold", openfold.__file__)
assert util.find_spec("flash_attn"), "flash_attn is not importable"
assert os.path.isdir(os.environ.get("CUTLASS_PATH", "")), "CUTLASS_PATH is unset"
print("flash_attn ok, CUTLASS_PATH", os.environ["CUTLASS_PATH"])
PY
    local b p required=("$OF/openfold/resources/params/params_model_1_ptm.npz" "$STEREO" "$DATA/pdb_mmcif/mmcif_files")
    [ "$MIRROR" = yes ] && required+=(
        "$DATA/uniref90/uniref90.fasta"
        "$DATA/mgnify/mgy_clusters_2022_05.fa"
        "$DATA/pdb70/pdb70"
        "$UNICLUST/uniclust30_2018_08"
        "$DATA/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
    )
    for b in jackhmmer hhblits hhsearch; do command -v "$b" >/dev/null || die "missing binary: $b"; done
    for p in "${required[@]}"; do have "$p" || die "missing: $p"; done
}

setup::fold_vars() {
    # --mem: the per-CPU default OOM-kills relaxation. GRES needs a type on a mixed queue. Site-set.
    GPU_RES=${OPENFOLD_GPU_RESOURCES:---cpus-per-task=8 --mem=32G}
    GPU_GRES=${OPENFOLD_GPU_GRES:-gpu:1}
    LAUNCH="${OPENFOLD_GPU_PARTITION:+srun ${OPENFOLD_GPU_ACCOUNT:+-A $OPENFOLD_GPU_ACCOUNT }-p $OPENFOLD_GPU_PARTITION --gres=$GPU_GRES $GPU_RES -t 00:30:00 }"
    EXAMPLE=${OPENFOLD_EXAMPLE:-6KWC_1}
    FOLD_ARGS=${OPENFOLD_FOLD_ARGS:-}
    STRUCTURE=relaxed
    case $FOLD_ARGS in *skip_relaxation*) STRUCTURE=unrelaxed ;; esac
}

# Record resolved values, not the caller's -- a consumer shouldn't know our fallbacks.
setup::config_save() {
    log config
    export OPENFOLD_HOME=$REPO OPENFOLD_PREFIX=$PREFIX OPENFOLD_ENV_NAME=$ENV_NAME
    export OPENFOLD_ENV_PREFIX=$CONDA_PREFIX OPENFOLD_DATA_DIR=$DATA OPENFOLD_MAX_CUDA=$MAX_CUDA
    export OPENFOLD_GPU_RESOURCES=$GPU_RES OPENFOLD_EXAMPLE=$EXAMPLE OPENFOLD_GPU_GRES=$GPU_GRES
    export VIZFOLD_DB=$PREFIX/vizfold.db
    config::save OPENFOLD_HOME OPENFOLD_PREFIX OPENFOLD_ENV_NAME OPENFOLD_ENV_PREFIX \
        OPENFOLD_DATA_DIR OPENFOLD_SITE OPENFOLD_AF2_ROOT OPENFOLD_MAX_CUDA \
        OPENFOLD_DRIVER_CUDA OPENFOLD_GPU_ACCOUNT OPENFOLD_GPU_PARTITION \
        OPENFOLD_GPU_RESOURCES OPENFOLD_GPU_GRES OPENFOLD_GPU_TIME \
        OPENFOLD_EXAMPLE OPENFOLD_FOLD_ARGS VIZFOLD_DB
}

setup::ready() {
    cat <<EOF
== ready (+$((SECONDS))s)

Check it works -- fold the bundled example and count the atoms:

  ${LAUNCH}env OPENFOLD_PREFIX=$PREFIX $REPO/scripts/openfold/fold.sh $EXAMPLE${FOLD_ARGS:+ $FOLD_ARGS}
  grep -c '^ATOM' $PREFIX/outputs/$EXAMPLE/predictions/${EXAMPLE}_model_1_ptm_$STRUCTURE.pdb

A few thousand atoms means it worked. To use the environment directly:

  export MAMBA_ROOT_PREFIX=$PREFIX/mamba
  eval "\$($MM shell hook --shell bash)" && micromamba activate $ENV_DIR
  export OPENFOLD_HOME=$REPO OPENFOLD_DATA_DIR=$DATA
EOF
}

main() {
    setup::config
    setup::preflight
    step micromamba setup::micromamba
    step env        setup::env
    setup::activate
    step cutlass    setup::cutlass
    if setup::nvrtc_detect; then
        step "nvrtc-$OPENFOLD_DRIVER_CUDA" setup::nvrtc_create
        setup::nvrtc_preload                          # unconditional: openfold.sh is rewritten every run
    fi
    setup::openfold_present || setup::build_openfold
    if [ "$MIRROR" = yes ]; then
        setup::link_mirror
    else
        step params setup::fetch_params
        setup::fetch_templates
    fi
    setup::stereo
    setup::verify
    setup::fold_vars
    setup::config_save
    setup::ready
}

# A site's hooks are pure function defs; sourcing them here lets it override any setup:: step above.
[ -n "${OPENFOLD_SITE:-}" ] && [ -f "$OF/install/sites/$OPENFOLD_SITE.sh" ] && . "$OF/install/sites/$OPENFOLD_SITE.sh"
main "$@"
