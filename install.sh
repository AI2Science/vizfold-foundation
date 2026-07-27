#!/bin/bash

# Bootstrap vizfold's core dependencies into ~/.local/bin -- the release binary and micromamba, which every environment is created and run through. Then `vizfold install base` fetches the checkout those live in, and `vizfold install <backend>` installs a model backend (OpenFold, ESMFold; each a pip/conda-installable package under backends/<name>/ with its own installer).
set -euo pipefail

die() { echo "FATAL: $*" >&2; exit 1; }

bootstrap::config() {
    REPO=${VIZFOLD_REPO:-AI2Science/vizfold-foundation}
    VERSION=${VIZFOLD_VERSION:-latest}   # a release tag (e.g. v0.1.0) or "latest"
    BIN=${VIZFOLD_BIN_DIR:-$HOME/.local/bin}
    mkdir -p "$BIN"
}

# One OS/arch detection for every binary this bootstrap installs. Linux only: that is what
# release.yml publishes, and a model backend needs CUDA and a scheduler anyway.
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

# micromamba creates and runs every environment; everything downstream assumes it is on PATH.
bootstrap::micromamba() {
    if [ -x "$BIN/micromamba" ] || command -v micromamba >/dev/null; then return; fi
    echo "downloading micromamba ($MAMBA_BUILD) ..."
    # A failed fetch leaves a non-executable file, so the guard above self-heals on the next run.
    curl -fsSL "https://micro.mamba.pm/api/micromamba/$MAMBA_BUILD/latest" |
        tar -xj -O bin/micromamba > "$BIN/micromamba" || die "micromamba download failed"
    chmod +x "$BIN/micromamba"
    echo "installed micromamba to $BIN/micromamba"
}

# Append a line to one shell's rc, once. .zshrc only if it exists (zsh may not be installed);
# .bashrc regardless, since bash reads it on every interactive shell even on a fresh account.
bootstrap::rc() {   # $1 shell, $2 line
    local rc=$HOME/.${1}rc
    [ "$1" = bash ] || [ -f "$rc" ] || return 0
    grep -qsF "$2" "$rc" || echo "$2" >> "$rc"
}

# Put ~/.local/bin on PATH for future shells (idempotent), and note it for this one.
bootstrap::path() {
    case ":$PATH:" in *":$BIN:"*) return ;; esac
    local line="export PATH=\"$BIN:\$PATH\""
    bootstrap::rc bash "$line"
    bootstrap::rc zsh "$line"
    echo "added $BIN to PATH in your shell rc; restart your shell or run: $line"
}

# Tab completion, eval'd from the binary rather than written out as a file: it then follows whatever
# vizfold is on PATH, so a self-update cannot leave a stale script behind. After the PATH line, which
# is what puts vizfold there; the guard keeps an uninstall from erroring on every shell start.
bootstrap::completions() {
    local shell
    for shell in bash zsh; do
        bootstrap::rc "$shell" "command -v vizfold >/dev/null && eval \"\$(vizfold completions $shell)\""
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
    echo "vizfold installed at $BIN/vizfold. Run \`vizfold install base\` for the checkout, then \`vizfold install openfold\` (or \`esmfold\`)."
}

# Sourced (tests/install_rc.sh) this file is just its definitions. Not the backend installers'
# `BASH_SOURCE = $0` guard: piped through `curl | bash` this script has no BASH_SOURCE at all, and
# that guard would read the absence as "sourced" and bootstrap nothing.
if [ -z "${BASH_SOURCE[0]:-}" ] || [ "${BASH_SOURCE[0]}" = "$0" ]; then main; fi
