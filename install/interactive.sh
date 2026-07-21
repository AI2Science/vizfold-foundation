# interactive.sh -- ask before guessing. Library; source it, do not run it.
#
#   . "$REPO/install/interactive.sh"
#   prefix=$(interactive::resolve OPENFOLD_PREFIX "install prefix" "$HOME/openfold")
#   alloc=$(interactive::choose OPENFOLD_ALLOCATION allocation bbol cqj)
#
# Both echo the chosen value and return non-zero if there is none, leaving the
# caller's `set -e` to stop. Prompts read /dev/tty, not stdin, which under
# `curl ... | bash` is the script itself.

[ "${BASH_SOURCE[0]}" = "$0" ] && { echo "interactive.sh is a library" >&2; exit 1; }
[ -n "${INTERACTIVE_SH:-}" ] && return 0
INTERACTIVE_SH=1

# `test -r /dev/tty` passes in a batch job, where opening it fails.
interactive::available() { { : <"/dev/tty"; } 2>/dev/null; }

interactive::resolve() {
    local var=$1 label=$2 value=${!1:-$3} reply
    if [ -n "${!1:-}" ]; then
        echo "$value"
    elif interactive::available; then
        printf '%s [%s]: ' "$label" "$value" >&2
        read -r reply <"/dev/tty" || reply=
        echo "${reply:-$value}"
    else
        echo "$label: $value (set $var to override)" >&2
        echo "$value"
    fi
}

interactive::choose() {
    local var=$1 label=$2 choice
    shift 2
    [ -n "${!var:-}" ] && { echo "${!var}"; return 0; }
    [ $# -eq 0 ] && { echo "no $label found; set $var" >&2; return 1; }
    [ $# -eq 1 ] && { echo "$label: $1" >&2; echo "$1"; return 0; }
    interactive::available || { echo "several $label ($*); set $var" >&2; return 1; }
    echo "select $label:" >&2
    # The >&2 covers the whole block, so the answer is echoed after it, not inside.
    select choice in "$@"; do
        [ -n "$choice" ] && break
    done <"/dev/tty" >&2
    [ -n "$choice" ] || { echo "no $label chosen" >&2; return 1; }   # EOF
    echo "$choice"
}
