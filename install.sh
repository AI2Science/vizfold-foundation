#!/bin/bash

# Bootstrap the vizfold platform CLI: download the release binary into ~/.local/bin, then `vizfold install` installs a model backend (OpenFold, ESMFold; each a pip/conda-installable package under backends/<name>/ with its own installer).
set -euo pipefail

die() { echo "FATAL: $*" >&2; exit 1; }

bootstrap::config() {
    REPO=${VIZFOLD_REPO:-AI2Science/vizfold-foundation}
    VERSION=${VIZFOLD_VERSION:-latest}   # a release tag (e.g. v0.1.0) or "latest"
    BIN=${VIZFOLD_BIN_DIR:-$HOME/.local/bin}
}

# Map uname to the release asset name the workflow publishes (vizfold-<os>-<arch>).
bootstrap::asset() {
    local os arch
    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)
    case "$arch" in
        x86_64|amd64)  arch=x86_64 ;;
        aarch64|arm64) arch=aarch64 ;;
        *) die "unsupported architecture: $arch" ;;
    esac
    ASSET="vizfold-${os}-${arch}"
    if [ "$VERSION" = latest ]; then
        URL="https://github.com/$REPO/releases/latest/download/$ASSET"
    else
        URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET"
    fi
}

bootstrap::download() {
    mkdir -p "$BIN"
    echo "downloading $ASSET ($VERSION) from $REPO ..."
    curl -fSL "$URL" -o "$BIN/vizfold" ||
        die "download failed: $URL -- check that a release with this asset exists (set VIZFOLD_VERSION to pin one)"
    chmod +x "$BIN/vizfold"
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
    bootstrap::asset
    bootstrap::download
    bootstrap::path
    echo "vizfold installed at $BIN/vizfold. Run \`vizfold install\` to install a model backend (OpenFold)."
}
main
