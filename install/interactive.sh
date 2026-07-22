# interactive.sh -- ask before guessing. Library; source it, do not run it.
#
#   . "$REPO/install/interactive.sh"
#   prefix=$(interactive::resolve OPENFOLD_PREFIX "install prefix" "$HOME/openfold")
#
# Echoes the value, so a caller can always proceed: a site works out a sensible
# default and this only reports it, or takes an answer when someone is there.
# Prompts read /dev/tty, not stdin, which under `curl ... | bash` is the script.

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

