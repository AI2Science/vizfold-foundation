#!/bin/bash

# Bootstrap the vizfold platform CLI into ~/.local/bin; then `vizfold init` installs a model backend (OpenFold today; the same install/ scripts would host openfold3/boltz/esmfold).
set -euo pipefail

bootstrap::config() {
    REPO_URL=${OPENFOLD_REPO_URL:-https://github.com/AI2Science/vizfold-foundation.git}
    BRANCH=${OPENFOLD_BRANCH:-main}
    SRC=${OPENFOLD_SRC:-$HOME/openfold-src}   # the checkout the CLI is built from; `vizfold init` reuses it
    BIN=$HOME/.local/bin
}

# Piped into bash BASH_SOURCE is unusable; under sbatch it is the spool copy.
bootstrap::find_repo() {
    if [ -n "${OPENFOLD_HOME:-}" ]; then
        REPO=$OPENFOLD_HOME
    elif [ -f "${SLURM_SUBMIT_DIR:-$PWD}/setup.py" ]; then
        REPO=${SLURM_SUBMIT_DIR:-$PWD}
    else
        # Walk up from this file; a copy saved outside a checkout finds nothing.
        REPO=$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null &&
            until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)
    fi
}

# No checkout found: fall back to our own clone and keep it current, so a re-run after a fix doesn't reuse stale code.
bootstrap::sync_checkout() {
    [ -f "$REPO/setup.py" ] && return
    REPO=$SRC
    if [ -d "$REPO/.git" ]; then
        # Point origin at REPO_URL so OPENFOLD_REPO_URL applies to re-runs, not just the first clone.
        git -C "$REPO" remote set-url origin "$REPO_URL" 2>/dev/null
        git -C "$REPO" fetch -q origin "$BRANCH" &&
            git -C "$REPO" reset -q --hard FETCH_HEAD ||
            echo "warning: could not update $REPO, using it as-is" >&2
    else
        git clone -q --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO"   # tip only; the build needs no history
    fi
}

# Minimal rustup into ~/.cargo so the build needs nothing preinstalled on the node.
bootstrap::rust() {
    command -v cargo >/dev/null && return
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    . "$HOME/.cargo/env"
}

bootstrap::build() {
    cargo build --release --manifest-path "$REPO/science-gateway/apps/executor/Cargo.toml"
    install -Dm755 "$REPO/science-gateway/apps/executor/target/release/vizfold" "$BIN/vizfold"
    echo "installed vizfold to $BIN/vizfold"
}

# Put ~/.local/bin on PATH for future shells (idempotent), and note it for this one.
bootstrap::path() {
    case ":$PATH:" in *":$BIN:"*) return ;; esac
    local line="export PATH=\"$BIN:\$PATH\""
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [ "$rc" = "$HOME/.zshrc" ] && [ ! -f "$rc" ]; then continue; fi
        grep -qsF "$line" "$rc" 2>/dev/null || echo "$line" >> "$rc"
    done
    echo "added $BIN to PATH in your shell rc; restart your shell or run: $line"
}

main() {
    bootstrap::config
    bootstrap::find_repo
    bootstrap::sync_checkout
    bootstrap::rust
    bootstrap::build
    bootstrap::path
    echo "vizfold installed at $BIN/vizfold. Run \`vizfold init\` to install a model backend (OpenFold)."
}
main
