#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
exec "$PY" "$ROOT/scripts/build_editions.py"
