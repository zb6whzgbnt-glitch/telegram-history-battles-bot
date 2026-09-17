#!/usr/bin/env bash
set -u
cd /workspace/telegram-history-battles-bot
mkdir -p data
export PYTHONUNBUFFERED=1
PIDFILE=data/bot.pid
LOG=data/bot.log

while true; do
  echo "$(date -Is) starting history_bot" >> "$LOG"
  .venv/bin/python -m history_bot >> "$LOG" 2>&1 &
  PID=$!
  echo "$PID" > "$PIDFILE"
  wait "$PID"
  CODE=$?
  echo "$(date -Is) history_bot exited code=$CODE — restart in 5s" >> "$LOG"
  sleep 5
done
