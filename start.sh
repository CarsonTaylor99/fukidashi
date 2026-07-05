#!/usr/bin/env bash
cd "$(dirname "$0")"
exec .venv/bin/uvicorn fukidashi.server:app --host 0.0.0.0 --port 8014 "$@"
