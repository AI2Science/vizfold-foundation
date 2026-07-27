#!/bin/bash
# What the bootstrap writes into a user's shell rc, and what it refuses to write twice.
# Run: bash tests/install_rc.sh
#
# `curl install.sh | bash` is documented as re-runnable, so every line it appends is appended again
# on the next run unless the guard holds -- and these land in a file the user did not ask us to edit.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SANDBOX=${TMPDIR:-/tmp}/vizfold-install-rc-$$
trap 'rm -rf "$SANDBOX"' EXIT
fail=0

. "$REPO/install.sh"                          # definitions only; its guard runs main on execution
set +e                                        # the installer's own set -e, not this harness's

run_case() { # $1 case name, $2... rc files to pre-create -- how "zsh is installed" is expressed
    CASE=$1; shift
    rm -rf "$SANDBOX"; mkdir -p "$SANDBOX"
    HOME=$SANDBOX BIN=$SANDBOX/.local/bin
    for rc; do : > "$SANDBOX/$rc"; done
    bootstrap::path >/dev/null
    bootstrap::completions >/dev/null
}

# grep -c prints 0 itself on no match, so only an absent file needs answering for.
count() { if [ -f "$SANDBOX/$1" ]; then grep -cF "$2" "$SANDBOX/$1"; else echo 0; fi; }

want() { # $1 rc, $2 fragment, $3 expected count, $4 what it means
    local got; got=$(count "$1" "$2")
    if [ "$got" = "$3" ]; then echo "ok   $4"; else
        echo "FAIL [$CASE] $4"; echo "  want $3 x '$2' in $1"; echo "  got  $got"; fail=1
    fi
}

PATH_LINE='export PATH='
BASH_EVAL='completions bash'
ZSH_EVAL='completions zsh'

run_case fresh
want .bashrc "$PATH_LINE" 1 "a fresh account gets the PATH line in .bashrc"
want .bashrc "$BASH_EVAL" 1 "and the completion eval for its own shell"
if [ -e "$SANDBOX/.zshrc" ]; then
    echo "FAIL [fresh] .zshrc was created for a shell that may not be installed"; fail=1
else
    echo "ok   no .zshrc is invented where none existed"
fi

# Ubuntu's and RHEL's stock profiles source .bashrc before prepending ~/.local/bin, so a line that
# looked vizfold up on PATH would find nothing exactly when it runs, on the machines this targets.
run_case absolute
want .bashrc "$SANDBOX/.local/bin/vizfold" 1 "the eval names the binary by absolute path"
want .bashrc "command -v vizfold"          0 "and never resolves it through PATH"

# bash sources .bashrc for non-interactive shells too, under ssh -- where there is no completion to
# register and forking the binary is pure latency on every scp, rsync and `ssh host cmd`.
want .bashrc 'case $- in *i*)' 1 "the eval runs only in an interactive shell"
# An older binary answers on stderr, which unsilenced lands on the user's prompt.
want .bashrc '2>/dev/null'     1 "and says nothing when the binary is too old to know it"

# An rc whose last byte is not a newline: ours would fuse onto the user's last statement, and since
# grep -F matches a fragment anywhere in a line the idempotency guard would then hide it forever.
CASE=no-trailing-newline
rm -rf "$SANDBOX"; mkdir -p "$SANDBOX"; HOME=$SANDBOX BIN=$SANDBOX/.local/bin
printf 'export EDITOR=vim' > "$SANDBOX/.bashrc"
bootstrap::path >/dev/null; bootstrap::completions >/dev/null
got=$(HOME=$SANDBOX bash -c '. "$HOME/.bashrc" 2>/dev/null; echo "$EDITOR"')
if [ "$got" = vim ]; then
    echo "ok   an rc with no trailing newline keeps its last statement intact"
else
    echo "FAIL [no-trailing-newline] EDITOR became '$got'; the appended line fused onto it"; fail=1
fi

run_case rerun .zshrc
bootstrap::path >/dev/null; bootstrap::completions >/dev/null
want .bashrc "$PATH_LINE" 1 "a re-run appends no second PATH line"
want .bashrc "$BASH_EVAL" 1 "nor a second completion eval"
want .zshrc  "$ZSH_EVAL"  1 "and the same holds for .zshrc"

# A zsh script eval'd by bash completes nothing.
run_case per-shell .zshrc
want .zshrc  "$ZSH_EVAL"  1 "an existing .zshrc gets the zsh script"
want .zshrc  "$BASH_EVAL" 0 "and not the bash one"
want .bashrc "$ZSH_EVAL"  0 "nor .bashrc the zsh one"

exit $fail
