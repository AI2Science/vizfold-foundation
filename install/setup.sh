#!/bin/bash
# Install OpenFold into a micromamba environment, on the node that runs this.
# Reached from a site script; ../install.sh picks one. Idempotent per step.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"   # sets REPO, die
PREFIX=${OPENFOLD_PREFIX:-$HOME/openfold}
AF2=${OPENFOLD_AF2_ROOT:-}          # a site with a database mirror names it
ENV_NAME=${OPENFOLD_ENV_NAME:-openfold-env}
MAX_CUDA=${OPENFOLD_MAX_CUDA:-12.8}   # a runtime above the driver breaks Amber relaxation

DATA=$PREFIX/data
ENV_DIR=$PREFIX/mamba/envs/$ENV_NAME
MM=$PREFIX/bin/micromamba
CUTLASS=$PREFIX/cutlass
UNICLUST=$DATA/uniclust30/uniclust30_2018_08
STEREO=$REPO/openfold/resources/stereo_chemical_props.txt

export CONDA_PKGS_DIRS=${OPENFOLD_PKGS_DIR:-$PREFIX/../.openfold-pkgs}
export MAMBA_ROOT_PREFIX=$PREFIX/mamba TMPDIR=$PREFIX/tmp
export PIP_CACHE_DIR=$PREFIX/../.openfold-pip   # $HOME is small on some sites
# No device on the build node to probe, so target every datacenter GPU these
# sites schedule: 7.0 V100, 8.0 A100, 8.6 A40/A6000, 9.0 H100. Miss one and the
# fold dies "no kernel image is available for execution on the device".
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;8.0;8.6;9.0}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
# A CPU build node has no driver, so conda sees no __cuda and resolves pytorch-gpu
# down to an ancient CPU build that no longer exists. Assert the version instead.
export CONDA_OVERRIDE_CUDA=${OPENFOLD_MAX_CUDA:-12.8}

MIRROR=$([ -n "$AF2" ] && [ -d "$AF2" ] && echo yes || echo no)
REQUIRED=("$REPO/openfold/resources/params/params_model_1_ptm.npz" "$STEREO"
          "$DATA/pdb_mmcif/mmcif_files")
[ "$MIRROR" = yes ] && REQUIRED+=(
    "$DATA/uniref90/uniref90.fasta"
    "$DATA/mgnify/mgy_clusters_2022_05.fa"
    "$DATA/pdb70/pdb70"
    "$UNICLUST/uniclust30_2018_08"
    "$DATA/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
)
BINARIES=(jackhmmer hhblits hhsearch)

step() { echo "== $* (+$((SECONDS))s)"; }
have() { test -e "$1" || compgen -G "${1}_*.ffindex" >/dev/null; }   # ffindex sets are prefixes

mkdir -p "$PREFIX/bin" "$TMPDIR" "$DATA" "$REPO/openfold/resources"
hostname
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo "no GPU on this node"
echo "prefix=$PREFIX repo=$REPO env=$ENV_NAME max_cuda=$MAX_CUDA mirror=$MIRROR${AF2:+ ($AF2)}"
test -f "$REPO/setup.py" || die "$REPO is not an OpenFold checkout"

step micromamba
# linux-64 or linux-aarch64 (Grace-Hopper), whichever this node is.
case $(uname -m) in aarch64|arm64) MM_ARCH=linux-aarch64 ;; *) MM_ARCH=linux-64 ;; esac
[ -x "$MM" ] || curl -Ls "https://micro.mamba.pm/api/micromamba/$MM_ARCH/latest" |
    tar -xj -C "$PREFIX" bin/micromamba

step "conda env $ENV_NAME"
# By path and --no-rc: a ~/.condarc envs_dirs outranks MAMBA_ROOT_PREFIX and would
# put a 12 GB environment wherever it points, typically a small $HOME, and its
# channels join the solve. Neither belongs in a reproducible install.
[ -d "$ENV_DIR" ] ||
    "$MM" create -y --no-rc -p "$ENV_DIR" -f "$REPO/environment.yml" "cuda-version<=$MAX_CUDA"

set +u   # the conda gcc hook reads SYS_SYSROOT unset
eval "$("$MM" shell hook --shell bash)"
micromamba activate "$ENV_DIR"
set -u

step "third-party dependencies"
# scripts/install_third_party_dependencies.sh, whose `conda env config vars set`
# micromamba does not implement.
[ -d "$CUTLASS/.git" ] ||
    git clone -q https://github.com/NVIDIA/cutlass --branch v3.6.0 --depth 1 "$CUTLASS"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/openfold.sh" <<ACTIVATE
export CUTLASS_PATH=$CUTLASS
export KMP_AFFINITY=none
export LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}
ACTIVATE
. "$CONDA_PREFIX/etc/conda/activate.d/openfold.sh"

step nvrtc
# OpenMM JITs its kernels through NVRTC, and a driver refuses PTX emitted by a
# newer toolkit than its own -- CUDA_ERROR_UNSUPPORTED_PTX_VERSION, which surfaces
# only as "Minimization failed after 100 attempts". torch is immune, shipping
# prebuilt cubins, so pin NVRTC alone, beside the env: installed into it, the
# solver drags torch down with it (43 downgrades on a 12.8 env).
# LD_PRELOAD, not LD_LIBRARY_PATH: OpenMM's plugin carries DT_RPATH $ORIGIN/..,
# and RPATH outranks LD_LIBRARY_PATH, so the env's own copy would still win.
DRIVER_CUDA=${OPENFOLD_DRIVER_CUDA:-$(python3 -c "
import ctypes
v = ctypes.c_int()
ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(v))
print(f'{v.value // 1000}.{v.value % 1000 // 10}')" 2>/dev/null)} || true
# A CPU-only build queue has no driver to probe, but the GPU node's may still be
# older than the toolkit (Bridges-2 V100 = 12.6). Can't confirm it's new enough,
# so assume old and pin conservatively -- 12.2 PTX loads on any >=12.2 driver and
# still compiles OpenMM's kernels (proven on nexus). A site can set the exact one.
: "${DRIVER_CUDA:=${OPENFOLD_FALLBACK_CUDA:-12.2}}"
ENV_CUDA=$(ls "$CONDA_PREFIX"/lib/libnvrtc.so.*.*.* 2>/dev/null |
    sed 's/.*so\.//; s/\.[0-9]*$//' | head -1) || true
older() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$1" ] && [ "$1" != "$2" ]; }
export OPENFOLD_DRIVER_CUDA=$DRIVER_CUDA

if [ -n "${DRIVER_CUDA:-}" ] && [ -n "${ENV_CUDA:-}" ] && older "$DRIVER_CUDA" "$ENV_CUDA"; then
    NVRTC=$PREFIX/nvrtc-$DRIVER_CUDA
    [ -d "$NVRTC" ] ||
        "$MM" create -y --no-rc -p "$NVRTC" -c conda-forge "cuda-nvrtc<=$DRIVER_CUDA"
    LIB=$(ls "$NVRTC"/lib/libnvrtc.so.* 2>/dev/null | sort -V | tail -1)
    test -n "$LIB" || die "no libnvrtc in $NVRTC"
    echo "export LD_PRELOAD=$LIB\${LD_PRELOAD:+:\$LD_PRELOAD}" \
        >> "$CONDA_PREFIX/etc/conda/activate.d/openfold.sh"
    . "$CONDA_PREFIX/etc/conda/activate.d/openfold.sh"
    echo "driver CUDA $DRIVER_CUDA is older than NVRTC $ENV_CUDA; preloading ${LIB##*/}"
else
    echo "driver CUDA ${DRIVER_CUDA:-unknown}, NVRTC ${ENV_CUDA:-unknown}; no pin needed"
fi

step openfold
# No build isolation: the extension must link against this env's torch. Build in a
# curated env -- a site's modules leak CC/CXX/CPATH/LD_LIBRARY_PATH for their own
# gcc, and nvcc then cannot drive this env's toolchain (cc1plus/crt not found).
# env -i drops all of that; CC/CXX name this env's gcc 12 explicitly, or torch falls
# back to a RHEL8 system gcc 8 that is too old for it. Runtime is untouched.
CONDA_CC=$(echo "$CONDA_PREFIX"/bin/*-conda-linux-gnu-gcc)
CONDA_CXX=$(echo "$CONDA_PREFIX"/bin/*-conda-linux-gnu-g++)
python3 -c 'import torch, openfold, attn_core_inplace_cuda' 2>/dev/null ||
    env -i HOME="$HOME" PATH="$CONDA_PREFIX/bin:/usr/bin:/bin" \
        CC="$CONDA_CC" CXX="$CONDA_CXX" \
        CUDA_HOME="$CONDA_PREFIX" TMPDIR="$TMPDIR" MAX_JOBS="$MAX_JOBS" \
        TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
        pip install --no-build-isolation -e "$REPO"

step datasets
if [ "$MIRROR" = yes ]; then
    ln -sfn "$AF2"/* "$DATA/"
    ln -sfn "$AF2/params" "$REPO/openfold/resources/params"
    # hhblits wants uniclust30_2018_08. A mirror that ships it (PACE) is linked
    # above; one that ships only its successor uniref30 (Delta) is aliased here.
    # Guard on have(), not the symlink: writing into a linked read-only mirror fails.
    if ! have "$UNICLUST/uniclust30_2018_08"; then
        mkdir -p "$UNICLUST"
        for f in "$AF2"/uniref30/UniRef30_[0-9][0-9][0-9][0-9]_[0-9][0-9]*; do
            [ -e "$f" ] || continue
            ln -sfn "$f" "$UNICLUST/uniclust30_2018_08${f##*/UniRef30_[0-9][0-9][0-9][0-9]_[0-9][0-9]}"
        done
    fi
else
    # No mirror: fetch the parameters, and the templates the bundled examples cite.
    # Into the prefix, not the checkout: these are 4 GB, and a site puts its
    # prefix on the volume meant for bulk data.
    [ -f "$PREFIX/params/params_model_1_ptm.npz" ] ||
        bash "$REPO/scripts/download_alphafold_params.sh" "$PREFIX"
    ln -sfn "$PREFIX/params" "$REPO/openfold/resources/params"
    mkdir -p "$DATA/pdb_mmcif/mmcif_files"
    # env -u LD_LIBRARY_PATH: the activated env prepends $CONDA_PREFIX/lib, and system
    # curl then binds conda's libcurl, which on some sites lacks a needed feature and
    # fails every fetch. || true so a single obsolete PDB does not abort the install;
    # the count assert below catches a total failure loudly instead of silently.
    grep -ohE "^ *[0-9]+ [0-9A-Za-z]{4}_" "$REPO"/examples/monomer/alignments/*/*.hhr |
        awk '{ print tolower(substr($2, 1, 4)) }' | sort -u |
        xargs -P 8 -I{} sh -c \
            '[ -s "$1/{}.cif" ] || env -u LD_LIBRARY_PATH curl -Lsf -o "$1/{}.cif" https://files.rcsb.org/download/{}.cif' _ \
            "$DATA/pdb_mmcif/mmcif_files" || true
    n=$(ls "$DATA/pdb_mmcif/mmcif_files" | wc -l)
    echo "fetched $n template mmCIFs"
    [ "$n" -gt 0 ] || die "no template mmCIFs fetched; check outbound https from the compute node"
fi
[ -f "$STEREO" ] || { env -u LD_LIBRARY_PATH curl -Lsf -o "$STEREO.part" \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt &&
    mv "$STEREO.part" "$STEREO"; }
mkdir -p "$REPO/tests/test_data/alphafold/common"
ln -sfn "$STEREO" "$REPO/tests/test_data/alphafold/common/stereo_chemical_props.txt"
step verify
python3 - <<'PY'
import importlib.util as util, os, torch, attn_core_inplace_cuda, openfold
from openfold.model.primitives import Linear
print("torch", torch.__version__, "cuda_devices", torch.cuda.device_count())
print("openfold", openfold.__file__)
assert util.find_spec("flash_attn"), "flash_attn is not importable"
assert os.path.isdir(os.environ.get("CUTLASS_PATH", "")), "CUTLASS_PATH is unset"
print("flash_attn ok, CUTLASS_PATH", os.environ["CUTLASS_PATH"])
PY
for b in "${BINARIES[@]}"; do command -v "$b" >/dev/null || die "missing binary: $b"; done
for p in "${REQUIRED[@]}"; do have "$p" || die "missing: $p"; done

# The site script names its GPU queue and, where the hardware constrains it, a
# smaller example or extra fold arguments.
# --mem matters: the scheduler default is per-CPU and Amber relaxation is OOM-killed under it.
GPU_RES=${OPENFOLD_GPU_RESOURCES:---cpus-per-task=8 --mem=32G}
GPU_GRES=${OPENFOLD_GPU_GRES:-gpu:1}   # a heterogeneous queue needs a type, e.g. gpu:a100:1
LAUNCH="${OPENFOLD_GPU_PARTITION:+srun ${OPENFOLD_GPU_ACCOUNT:+-A $OPENFOLD_GPU_ACCOUNT }-p $OPENFOLD_GPU_PARTITION --gres=$GPU_GRES $GPU_RES -t 00:30:00 }"
EXAMPLE=${OPENFOLD_EXAMPLE:-6KWC_1}
FOLD_ARGS=${OPENFOLD_FOLD_ARGS:-}
STRUCTURE=relaxed
case $FOLD_ARGS in *skip_relaxation*) STRUCTURE=unrelaxed ;; esac

step config
# The resolved value, not the caller's: a default that stayed a default is still
# what this install uses, and a consumer should not have to know our fallbacks.
export OPENFOLD_HOME=$REPO OPENFOLD_PREFIX=$PREFIX OPENFOLD_ENV_NAME=$ENV_NAME
export OPENFOLD_ENV_PREFIX=$CONDA_PREFIX OPENFOLD_DATA_DIR=$DATA OPENFOLD_MAX_CUDA=$MAX_CUDA
export OPENFOLD_GPU_RESOURCES=$GPU_RES OPENFOLD_EXAMPLE=$EXAMPLE OPENFOLD_GPU_GRES=$GPU_GRES
config::save OPENFOLD_HOME OPENFOLD_PREFIX OPENFOLD_ENV_NAME OPENFOLD_ENV_PREFIX \
    OPENFOLD_DATA_DIR OPENFOLD_SITE OPENFOLD_AF2_ROOT OPENFOLD_MAX_CUDA \
    OPENFOLD_DRIVER_CUDA OPENFOLD_GPU_ACCOUNT OPENFOLD_GPU_PARTITION \
    OPENFOLD_GPU_RESOURCES OPENFOLD_GPU_GRES OPENFOLD_EXAMPLE OPENFOLD_FOLD_ARGS

cat <<EOF
== ready (+$((SECONDS))s)

Check it works -- fold the bundled example and count the atoms:

  ${LAUNCH}env OPENFOLD_PREFIX=$PREFIX $REPO/run/fold.sh $EXAMPLE${FOLD_ARGS:+ $FOLD_ARGS}
  grep -c '^ATOM' $PREFIX/outputs/$EXAMPLE/predictions/${EXAMPLE}_model_1_ptm_$STRUCTURE.pdb

A few thousand atoms means it worked. To use the environment directly:

  export MAMBA_ROOT_PREFIX=$PREFIX/mamba
  eval "\$($MM shell hook --shell bash)" && micromamba activate $ENV_DIR
  export OPENFOLD_HOME=$REPO OPENFOLD_DATA_DIR=$DATA
EOF
