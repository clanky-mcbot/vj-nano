#!/usr/bin/env bash
# Launch vj-nano on Jetson Nano with everything configured~!
# Usage:  ./run.sh [extra-args]

set -euo pipefail

# ── config ──────────────────────────────
VENV="/home/gigant/vj-nano/.venv"
LOG="/tmp/vj-nano.log"
PIDFILE="/tmp/vj-nano.pid"

export DISPLAY=":1"
export OPENBLAS_CORETYPE=ARMV8
export PYTHONUNBUFFERED=1

# ── helpers ─────────────────────────────
log()    { echo "[$(date '+%H:%M:%S')] $*"; }
kill_old() {
    if [[ -f "$PIDFILE" ]]; then
        OLD=$(cat "$PIDFILE" 2>/dev/null || true)
        if [[ -n "${OLD:-}" ]] && kill -0 "$OLD" 2>/dev/null; then
            log "Killing old vj-nano (PID $OLD)..."
            kill -9 "$OLD" 2>/dev/null || true
            sleep 1
        fi
        rm -f "$PIDFILE"
    fi
    pkill -9 -f "python3 -m vj.main" 2>/dev/null || true
}

rotate_log() {
    if [[ -f "$LOG" && $(stat -c%s "$LOG" 2>/dev/null || echo 0) -gt 10485760 ]]; then
        mv "$LOG" "${LOG}.old" 2>/dev/null || true
    fi
}

# ── main ────────────────────────────────
log "╔══════════════════════════════════════╗"
log "║         vj-nano launcher             ║"
log "╚══════════════════════════════════════╝"

kill_old
rotate_log

log "Activating venv: $VENV"
source "$VENV/bin/activate"

find /home/gigant/vj-nano/src -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

DEFAULT_ARGS=(
    --flip-webcam
    --audio line
    --audio-device 19
    --sensitivity 1.0
    --webcam-device auto
    --debug
    --fps
)

# FIX: only pass extra args if there actually are some
if [[ $# -gt 0 ]]; then
    log "Launching vj-nano with extras: $*"
    nohup python3 -m vj.main "${DEFAULT_ARGS[@]}" "$@" >> "$LOG" 2>&1 &
else
    log "Launching vj-nano (no extra args)"
    nohup python3 -m vj.main "${DEFAULT_ARGS[@]}" >> "$LOG" 2>&1 &
fi

NEWPID=$!
echo "$NEWPID" > "$PIDFILE"

log "Started with PID $NEWPID"
log "Log file: $LOG"
log "Tail:     tail -f $LOG"
log "Stop:     kill \$(cat $PIDFILE)"
log ""

sleep 3
if kill -0 "$NEWPID" 2>/dev/null; then
    log "✅ vj-nano is running~!"
else
    log "❌ vj-nano died immediately. Last log lines:"
    tail -n 30 "$LOG" || true
    rm -f "$PIDFILE"
    exit 1
fi
