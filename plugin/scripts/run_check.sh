#!/usr/bin/env bash
# Stop-hook entrypoint. Invokes the compliance-check-hook console script
# from the published GitHub repo via uvx.

set -e

if ! command -v uvx >/dev/null 2>&1; then
  echo "compliance-checker: uvx not found on PATH" >&2
  exit 0
fi

# Override the source (e.g. during local dev) by setting COMPLIANCE_CHECKER_SRC
# to a path or VCS URL understood by `uvx --from`.
SRC="${COMPLIANCE_CHECKER_SRC:-git+https://github.com/shakahs/compliance-checker@main}"

exec uvx --from "$SRC" compliance-check-hook
