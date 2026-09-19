#!/usr/bin/env bash
# Install (or update) StreamJanitor on a Raspberry Pi running Mopidy with an S/PDIF HAT.
#
# Run as the user that will own the service (not root); sudo is used where needed.
# Safe to run again: it updates the program and keeps the existing config and library.
#
#   deploy/install-pi.sh [options]
#
#   --source SRC        what to install: a git URL (git+https://...) or a local checkout
#                       (default: the checkout this script belongs to)
#   --hat ID            ALSA card id of the HAT (default: detected, asked if unclear)
#   --host ADDR         web interface listen address (default: as installed, else 0.0.0.0 = whole LAN)
#   --port N            web interface port (default: as installed, else 8080)
#   --delay S           delay_s for a new config (default: the example's)
#   --rate R            sample rate of the whole chain, 96000 or 48000 (default: the rate your
#                       Mopidy output already forces, else 96000)
#   --patch-mopidy      point Mopidy at the loopback without asking
#   --no-patch-mopidy   leave the Mopidy config alone (the needed change is printed)
#   --yes               don't ask; take the defaults
#   --dry-run           show what would be done, change nothing
#   --uninstall         remove service, loopback setup and program; restore Mopidy's config
#                       (config and library are kept)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

SOURCE=""
HAT=""
HOST=""
PORT=""
DELAY=""
RATE=""
PATCH_MOPIDY="ask"
ASSUME_YES=0
DRY_RUN=0
UNINSTALL=0

SERVICE_FILE=/etc/systemd/system/streamjanitor.service
ALOOP_CONF=/etc/modprobe.d/snd-aloop.conf
ALOOP_LOAD=/etc/modules-load.d/snd-aloop.conf
MOPIDY_CONF=/etc/mopidy/mopidy.conf
MOPIDY_BACKUP=/etc/mopidy/mopidy.conf.streamjanitor-backup

usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) SOURCE="$2"; shift 2 ;;
        --hat) HAT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --delay) DELAY="$2"; shift 2 ;;
        --rate) RATE="$2"; shift 2 ;;
        --patch-mopidy) PATCH_MOPIDY="yes"; shift ;;
        --no-patch-mopidy) PATCH_MOPIDY="no"; shift ;;
        --yes|-y) ASSUME_YES=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

# --- helpers ---

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# Run a command that changes the system (printed instead in --dry-run).
run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '[dry-run] %s\n' "$*"
    else
        "$@"
    fi
}

# Write stdin to a root-owned file.
write_root_file() {
    local path="$1" content
    content="$(cat)"
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '[dry-run] write %s:\n%s\n' "$path" "$content" | sed 's/^/    /'
    else
        printf '%s\n' "$content" | sudo tee "$path" >/dev/null
    fi
}

ask() {  # ask "question" default(y|n) -> returns 0 for yes
    local answer default="$2"
    if [[ $ASSUME_YES -eq 1 ]]; then
        [[ "$default" == y ]]
        return
    fi
    read -r -p "$1 [$([[ $default == y ]] && echo Y/n || echo y/N)] " answer
    answer="${answer:-$default}"
    [[ "$answer" =~ ^[Yy] ]]
}

uv_bin() {
    command -v uv 2>/dev/null || { [[ -x "$HOME/.local/bin/uv" ]] && echo "$HOME/.local/bin/uv"; } || true
}

# The `output` value of Mopidy's [audio] section in a config file (continuation lines joined).
mopidy_output_of() {
    [[ -f "$1" ]] || return 0
    sudo cat "$1" | python3 -c '
import re, sys
section, value = None, None
for raw in sys.stdin.read().splitlines():
    header = re.fullmatch(r"\s*\[([^]]+)\]\s*", raw)
    if header:
        section = header[1].strip().lower()
    elif value is not None and raw[:1] in (" ", "\t") and raw.strip():
        value += " " + raw.strip()
    elif value is not None:
        break
    elif section == "audio" and (m := re.match(r"\s*output\s*[=:]\s*(.*)", raw)):
        value = m[1].strip()
print(value or "")
'
}

# streamjanitor.mopidy from this checkout (pure Python: works before the program is installed).
#   mopidy_py rate ORIGINAL [RATE]           the chain's rate: RATE, else the original's, else 96000
#   mopidy_py line ORIGINAL RATE LATENCY_MS  the `output =` value for the loopback
mopidy_py() {
    PYTHONPATH="$REPO_DIR/src" python3 -c 'import sys
from streamjanitor.mopidy import caps_rate, loopback_output
cmd, original, *rest = sys.argv[1:]
if cmd == "rate":
    print(rest[0] if rest and rest[0] else caps_rate(original) or 96000)
else:
    print(loopback_output(original, int(rest[0]), int(rest[1])))' "$@"
}

# Card ids from /proc/asound/cards, e.g. " 0 [vc4hdmi ]: ..." -> vc4hdmi
card_ids() {
    sed -n 's/^ *[0-9]\+ \[\([^] ]*\) *\].*/\1/p' /proc/asound/cards 2>/dev/null || true
}

# --- uninstall ---

if [[ $UNINSTALL -eq 1 ]]; then
    say "Removing the service"
    if [[ -f $SERVICE_FILE ]]; then
        run sudo systemctl disable --now streamjanitor.service || true
        run sudo rm -f "$SERVICE_FILE"
        run sudo systemctl daemon-reload
    fi
    say "Removing the loopback setup"
    run sudo rm -f "$ALOOP_CONF" "$ALOOP_LOAD"
    if [[ -f $MOPIDY_BACKUP ]]; then
        say "Restoring Mopidy's original config"
        run sudo mv "$MOPIDY_BACKUP" "$MOPIDY_CONF"
    fi
    UV="$(uv_bin)"
    if [[ -n "$UV" ]]; then
        say "Uninstalling the program"
        run "$UV" tool uninstall streamjanitor || true
    fi
    say "Done. Config and library were kept in ~/.config/streamjanitor and ~/.local/share/streamjanitor."
    say "Reboot to unload the loopback and restart Mopidy on the HAT: sudo reboot"
    exit 0
fi

# --- checks ---

[[ $EUID -ne 0 ]] || die "run this as your normal user, not root (sudo is used where needed)"
[[ -z "$RATE" || "$RATE" == 96000 || "$RATE" == 48000 ]] || die "--rate must be 96000 or 48000"
[[ "$(uname -m)" == aarch64 ]] || warn "this is not a 64-bit system ($(uname -m)): numpy may have to be compiled, which is slow. Raspberry Pi OS 64-bit is recommended."

if [[ -z "$SOURCE" ]]; then
    [[ -f "$REPO_DIR/pyproject.toml" ]] || die "no --source given and $REPO_DIR is not a checkout"
    SOURCE="$REPO_DIR"
fi

# --- 1. program ---

UV="$(uv_bin)"
if [[ -z "$UV" ]]; then
    say "Installing uv (Python package manager)"
    command -v curl >/dev/null || die "curl is needed to install uv: sudo apt install curl"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] curl -LsSf https://astral.sh/uv/install.sh | sh"
        UV="$HOME/.local/bin/uv"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        UV="$HOME/.local/bin/uv"
    fi
fi

say "Installing streamjanitor from $SOURCE"
run "$UV" tool install --force --reinstall "$SOURCE"
BIN="$("$UV" tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")/streamjanitor"
[[ $DRY_RUN -eq 1 || -x "$BIN" ]] || die "streamjanitor was not installed at $BIN"

# --- 2. HAT ---

mapfile -t CARDS < <(card_ids)
if [[ -z "$HAT" ]]; then
    CANDIDATES=()
    for id in "${CARDS[@]}"; do
        [[ "$id" =~ ^(vc4hdmi.*|Headphones|Loopback|b1|b2)$ ]] || CANDIDATES+=("$id")
    done
    echo "Sound cards found:"
    sed 's/^/    /' /proc/asound/cards 2>/dev/null || echo "    (none: is this a Pi?)"
    DEFAULT="${CANDIDATES[0]:-}"
    if [[ ${#CANDIDATES[@]} -eq 1 && $ASSUME_YES -eq 1 ]]; then
        HAT="$DEFAULT"
    elif [[ $ASSUME_YES -eq 1 ]]; then
        die "can't tell which card is the HAT: pass --hat ID"
    else
        read -r -p "ALSA id of the S/PDIF HAT (the word in [brackets])${DEFAULT:+ [$DEFAULT]}: " HAT
        HAT="${HAT:-$DEFAULT}"
    fi
fi
[[ -n "$HAT" ]] || die "no HAT card id given"
if [[ ${#CARDS[@]} -gt 0 ]] && ! printf '%s\n' "${CARDS[@]}" | grep -qx "$HAT"; then
    warn "card '$HAT' is not in /proc/asound/cards right now"
fi
say "Using HAT card '$HAT'"

# --- 3. loopback clocked by the HAT ---

KERNEL="$(uname -r)"
KMAJ="${KERNEL%%.*}"; KMIN="${KERNEL#*.}"; KMIN="${KMIN%%.*}"
ALOOP_OPTS="id=Loopback pcm_substreams=1"
if (( KMAJ > 5 || (KMAJ == 5 && KMIN >= 8) )); then
    ALOOP_OPTS+=" timer_source=$HAT"
else
    warn "kernel $KERNEL is older than 5.8: the loopback can't follow the HAT's clock (drift is corrected in software)"
fi
say "Configuring the ALSA loopback ($ALOOP_OPTS)"
write_root_file "$ALOOP_CONF" <<EOF
# Written by streamjanitor's install-pi.sh: loopback for Mopidy -> streamjanitor,
# clocked by the S/PDIF HAT so both sides share one clock.
options snd-aloop $ALOOP_OPTS
EOF
write_root_file "$ALOOP_LOAD" <<< "snd-aloop"
NEED_REBOOT=0
if ! printf '%s\n' "${CARDS[@]}" | grep -qx Loopback; then
    NEED_REBOOT=1
elif [[ "$(cat /sys/module/snd_aloop/parameters/timer_source 2>/dev/null || true)" != *"$HAT"* && "$ALOOP_OPTS" == *timer_source* ]]; then
    NEED_REBOOT=1  # loaded earlier with other options
fi

run sudo usermod -aG audio "$USER"

# --- 4. config ---

# Mopidy's own chain is kept: take it from the pre-streamjanitor backup if there is one.
ORIG_FILE="$MOPIDY_CONF"
[[ -f $MOPIDY_BACKUP ]] && ORIG_FILE="$MOPIDY_BACKUP"
ORIG_OUTPUT="$(mopidy_output_of "$ORIG_FILE")"
RATE="$(mopidy_py rate "$ORIG_OUTPUT" "$RATE")"
say "Sample rate of the whole chain: $RATE Hz"

CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/streamjanitor/config.toml"
if [[ -f "$CONFIG" ]]; then
    say "Keeping the existing config $CONFIG"
    grep -q "CARD=$HAT" "$CONFIG" || warn "its output_device doesn't mention '$HAT': check it"
    if ! grep -qE "^sample_rate *= *$RATE\b" "$CONFIG"; then
        say "Setting sample_rate = $RATE in it"
        run sed -i -E "s/^(sample_rate *= *)[0-9]+/\1$RATE/" "$CONFIG"
    fi
else
    say "Writing $CONFIG"
    run "$BIN" init --config "$CONFIG" --output-device "plughw:CARD=$HAT,DEV=0" --rate "$RATE" ${DELAY:+--delay "$DELAY"}
fi
ALSA_LATENCY_MS="$(python3 -c 'import sys, tomllib
try:
    print(tomllib.load(open(sys.argv[1], "rb")).get("audio", {}).get("alsa_latency_ms", 100))
except OSError:
    print(100)' "$CONFIG")"

# --- 5. service ---

# An update keeps the address and port of the installed service unless given again.
if [[ -f $SERVICE_FILE ]]; then
    INSTALLED="$(grep -m1 '^ExecStart=' "$SERVICE_FILE" || true)"
    [[ -n "$HOST" ]] || HOST="$(sed -n 's/.* --host \([^ ]*\).*/\1/p' <<< "$INSTALLED")"
    [[ -n "$PORT" ]] || PORT="$(sed -n 's/.* --port \([^ ]*\).*/\1/p' <<< "$INSTALLED")"
fi
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

say "Installing the systemd service (web interface on $HOST:$PORT)"
sed -e "s|@USER@|$USER|g" -e "s|@HOME@|$HOME|g" -e "s|@BIN@|$BIN|g" -e "s|@HOST@|$HOST|g" \
    -e "s|@PORT@|$PORT|g" -e "s|@CONFIG@|$CONFIG|g" "$SCRIPT_DIR/streamjanitor.service" \
    | sed '1,/^$/d' | write_root_file "$SERVICE_FILE"
run sudo systemctl daemon-reload
run sudo systemctl enable streamjanitor.service

# --- 6. Mopidy ---

# Same chain, same samples; only the sink changes: the loopback, with the HAT's ALSA period.
MOPIDY_OUTPUT="output = $(mopidy_py line "$ORIG_OUTPUT" "$RATE" "$ALSA_LATENCY_MS")"
PATCHED_MOPIDY=0
if [[ -f $MOPIDY_CONF ]]; then
    if sudo grep -qxF "$MOPIDY_OUTPUT" "$MOPIDY_CONF"; then
        say "Mopidy already plays into the loopback"
    else
        if [[ $PATCH_MOPIDY == ask ]]; then
            echo "Mopidy must play into the loopback instead of the HAT. Your chain stays as it is;"
            echo "only the sink changes. From ($ORIG_FILE):"
            echo "    output = ${ORIG_OUTPUT:-(none)}"
            echo "to:"
            echo "    $MOPIDY_OUTPUT"
            ask "Change $MOPIDY_CONF now (a backup is kept)?" y && PATCH_MOPIDY=yes || PATCH_MOPIDY=no
        fi
        if [[ $PATCH_MOPIDY == yes ]]; then
            say "Pointing Mopidy at the loopback (backup: $MOPIDY_BACKUP)"
            [[ -f $MOPIDY_BACKUP ]] || run sudo cp -p "$MOPIDY_CONF" "$MOPIDY_BACKUP"
            # Replace output (and its continuation lines) in [audio], or add it.
            sudo cat "$MOPIDY_CONF" | python3 -c '
import re, sys
line = sys.argv[1]
out, section, done, skipping = [], None, False, False
for raw in sys.stdin.read().splitlines():
    if skipping and raw[:1] in (" ", "\t") and raw.strip():
        continue
    skipping = False
    header = re.fullmatch(r"\s*\[([^]]+)\]\s*", raw)
    if header:
        if section == "audio" and not done:  # end of [audio]: add before its trailing blank lines
            i = len(out)
            while i and not out[i - 1].strip():
                i -= 1
            out.insert(i, line); done = True
        section = header[1].strip().lower()
    elif section == "audio" and re.match(r"\s*output\s*[=:]", raw):
        if not done:
            out.append(line); done = True
        skipping = True
        continue
    out.append(raw)
if not done:
    if section != "audio":
        out += ["", "[audio]"]
    out.append(line)
print("\n".join(out))
' "$MOPIDY_OUTPUT" | write_root_file "$MOPIDY_CONF"
            PATCHED_MOPIDY=1
        else
            warn "Mopidy not changed. In $MOPIDY_CONF, under [audio], set:"
            echo "    $MOPIDY_OUTPUT"
        fi
    fi
else
    warn "$MOPIDY_CONF not found (Mopidy running as a user service?). Under [audio] set:"
    echo "    $MOPIDY_OUTPUT"
fi

# --- 7. start ---

ADDR="$(hostname).local"
if [[ $NEED_REBOOT -eq 1 ]]; then
    say "Done. Reboot to load the loopback: sudo reboot"
else
    run sudo systemctl restart streamjanitor.service
    [[ $PATCHED_MOPIDY -eq 1 ]] && run sudo systemctl restart mopidy.service
    say "Done: streamjanitor is running."
fi
cat <<EOF

  Web interface: http://$ADDR:$PORT/  (On Air)
  Next: in the Studio on your PC, "Export for On Air", then import the zip here.
  Logs:   journalctl -u streamjanitor -f
  Update: run this script again (with the same --source)
  Undo:   $0 --uninstall
EOF
