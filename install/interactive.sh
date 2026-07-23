# interactive.sh -- library. resolve() echoes a value (the site default, or a /dev/tty answer), so callers always proceed.

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

