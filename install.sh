#!/bin/bash

# Bootstrap the vizfold binary and micromamba into ~/.local/bin; everything else installs from there.
set -euo pipefail

die() { echo "FATAL: $*" >&2; exit 1; }

bootstrap::config() {
    REPO=${VIZFOLD_REPO:-AI2Science/vizfold-foundation}
    VERSION=${VIZFOLD_VERSION:-latest}   # a release tag (e.g. v0.1.0) or "latest"
    BIN=${VIZFOLD_BIN_DIR:-$HOME/.local/bin}
    mkdir -p "$BIN"
}

# Linux only: what release.yml publishes, and a backend needs CUDA and a scheduler anyway.
bootstrap::arch() {
    [ "$(uname -s)" = Linux ] || die "vizfold releases are Linux-only (this is $(uname -s))"
    case "$(uname -m)" in
        x86_64|amd64)  ARCH=x86_64;  MAMBA_BUILD=linux-64 ;;
        aarch64|arm64) ARCH=aarch64; MAMBA_BUILD=linux-aarch64 ;;
        *) die "unsupported architecture: $(uname -m)" ;;
    esac
}

bootstrap::asset() {
    ASSET="vizfold-linux-${ARCH}"
    if [ "$VERSION" = latest ]; then
        URL="https://github.com/$REPO/releases/latest/download/$ASSET"
    else
        URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET"
    fi
}

bootstrap::download() {
    echo "downloading $ASSET ($VERSION) from $REPO ..."
    curl -fsSL "$URL" -o "$BIN/vizfold" ||
        die "download failed: $URL -- check that a release with this asset exists (set VIZFOLD_VERSION to pin one)"
    chmod +x "$BIN/vizfold"
    echo "installed vizfold to $BIN/vizfold"
}

# Everything downstream creates and runs its envs through it, so it has to be on PATH.
bootstrap::micromamba() {
    if [ -x "$BIN/micromamba" ] || command -v micromamba >/dev/null; then return; fi
    echo "downloading micromamba ($MAMBA_BUILD) ..."
    # A failed fetch leaves a non-executable file, so the guard above self-heals next run.
    curl -fsSL "https://micro.mamba.pm/api/micromamba/$MAMBA_BUILD/latest" |
        tar -xj -O bin/micromamba > "$BIN/micromamba" || die "micromamba download failed"
    chmod +x "$BIN/micromamba"
    echo "installed micromamba to $BIN/micromamba"
}

# .bashrc always -- bash reads it on every interactive shell; .zshrc only where it exists, since
# writing one invents a config for a shell that may not be installed.
bootstrap::rc() {   # $1 shell, $2 line
    local rc=$HOME/.${1}rc
    [ "$1" = bash ] || [ -f "$rc" ] || return 0
    grep -qsF "$2" "$rc" && return 0
    # Else ours fuses onto an rc whose last line carries no newline.
    [ ! -s "$rc" ] || [ -z "$(tail -c1 "$rc")" ] || echo >> "$rc"
    echo "$2" >> "$rc"
}

bootstrap::path() {
    case ":$PATH:" in *":$BIN:"*) return ;; esac
    local line="export PATH=\"$BIN:\$PATH\""
    bootstrap::rc bash "$line"
    bootstrap::rc zsh "$line"
    echo "added $BIN to PATH in your shell rc; restart your shell or run: $line"
}

# Eval'd from the binary, not a written file, so a self-update leaves nothing stale. Absolute path, never PATH:
# Ubuntu's and RHEL's profiles source .bashrc *before* prepending ~/.local/bin, so a lookup there fails exactly here.
# -x quiets an uninstalled binary, 2>/dev/null one too old for the subcommand, *i* the non-interactive ssh shells bash reads .bashrc in too.
bootstrap::completions() {
    local shell
    for shell in bash zsh; do
        bootstrap::rc "$shell" "case \$- in *i*) [ -x \"$BIN/vizfold\" ] && eval \"\$(\"$BIN/vizfold\" completions $shell 2>/dev/null)\" ;; esac"
    done
    echo "enabled tab completion in your shell rc"
}

main() {
    bootstrap::config
    bootstrap::arch
    bootstrap::asset
    bootstrap::download
    bootstrap::micromamba
    bootstrap::path
    bootstrap::completions
    echo "vizfold installed at $BIN/vizfold. Run \`vizfold install src\` for the checkout, then \`vizfold install openfold\` (or \`esmfold\`)."
}

# Not the backend installers' `BASH_SOURCE = $0` guard: under `curl | bash` there is no BASH_SOURCE at
# all, which that guard would read as "sourced" and bootstrap nothing. Sourced by tests/install_rc.sh.
if [ -z "${BASH_SOURCE[0]:-}" ] || [ "${BASH_SOURCE[0]}" = "$0" ]; then main; fi
