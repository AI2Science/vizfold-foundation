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
BASH_EVAL='vizfold completions bash'
ZSH_EVAL='vizfold completions zsh'

run_case fresh
want .bashrc "$PATH_LINE" 1 "a fresh account gets the PATH line in .bashrc"
want .bashrc "$BASH_EVAL" 1 "and the completion eval for its own shell"
if [ -e "$SANDBOX/.zshrc" ]; then
    echo "FAIL [fresh] .zshrc was created for a shell that may not be installed"; fail=1
else
    echo "ok   no .zshrc is invented where none existed"
fi

# Against main, not against a file this test wrote: that would only check the order this test chose.
CASE=ordering
if [ "$(grep -n '^ *bootstrap::path$' "$REPO/install.sh" | cut -d: -f1)" -lt \
     "$(grep -n '^ *bootstrap::completions$' "$REPO/install.sh" | cut -d: -f1)" ]; then
    echo "ok   main writes the PATH line before the eval that depends on it"
else
    echo "FAIL [ordering] main appends the completion eval above the PATH line it needs"; fail=1
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
