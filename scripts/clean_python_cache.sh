#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

find "$REPO_ROOT" \
  -type d \
  -name '__pycache__' \
  -prune \
  -exec rm -rf {} +

find "$REPO_ROOT" \
  -type f \
  \( -name '*.pyc' -o -name '*.pyo' \) \
  -delete

echo "Python bytecode caches removed from: $REPO_ROOT"