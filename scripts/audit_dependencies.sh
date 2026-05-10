#!/usr/bin/env bash
set -euo pipefail

audit_requirements="$(mktemp "${TMPDIR:-/tmp}/mamut-audit.XXXXXX.txt")"
audit_cache="$(mktemp -d "${TMPDIR:-/tmp}/mamut-audit-cache.XXXXXX")"
trap 'rm -rf "$audit_requirements" "$audit_cache"' EXIT

uv export \
  --locked \
  --no-dev \
  --no-emit-project \
  --output-file "$audit_requirements" \
  > /dev/null

uv run --locked --only-group security pip-audit \
  --requirement "$audit_requirements" \
  --require-hashes \
  --disable-pip \
  --cache-dir "$audit_cache" \
  --progress-spinner off \
  "$@"
