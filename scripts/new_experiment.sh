#!/usr/bin/env bash
# new_experiment.sh — scaffold an experiments/ entry.
#
# Usage:  ./scripts/new_experiment.sh d4-adjacency-rate
#
# Creates experiments/YYYY-MM-DD-<slug>/README.md from TEMPLATE.md with the date,
# branch and current commit already filled in. Stamping the commit at creation is
# the point: a metric whose code you cannot identify is not reproducible, and the
# SHA is the one field nobody remembers to backfill afterwards.
set -euo pipefail

SLUG="${1:-}"
if [[ -z "${SLUG}" ]]; then
  echo "usage: $0 <short-slug>     e.g. $0 d4-adjacency-rate" >&2
  exit 1
fi
# Keep folder names sortable and shell-safe.
if [[ ! "${SLUG}" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]]; then
  echo "slug must be lowercase letters, digits and hyphens: '${SLUG}'" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${ROOT}/experiments/$(date +%F)-${SLUG}"

if [[ -e "${DIR}" ]]; then
  echo "already exists: ${DIR}" >&2
  exit 1
fi

# This node ships git 1.8.3.1, which predates `git -C` (added in 1.8.5). Using it
# here failed silently and stamped every scaffold with SHA=UNCOMMITTED, so run git
# from a subshell cd instead. Keep it that way unless the server's git is upgraded.
git_at() { ( cd "${ROOT}" && git "$@" ); }

# Uncommitted changes mean the recorded SHA would not describe what actually ran.
# Warn rather than block — you often scaffold before the code is final.
if ! git_at diff --quiet HEAD 2>/dev/null; then
  echo "note: working tree is dirty; commit before running, then update the SHA" >&2
fi

SHA="$(git_at rev-parse --short HEAD 2>/dev/null || echo UNCOMMITTED)"
BRANCH="$(git_at rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

mkdir -p "${DIR}"
sed -e "s|YYYY-MM-DD|$(date +%F)|" \
    -e "s|\`<sha>\` on \`<branch>\`|\`${SHA}\` on \`${BRANCH}\`|" \
    "${ROOT}/experiments/TEMPLATE.md" > "${DIR}/README.md"

echo "created ${DIR#"${ROOT}"/}/README.md"
echo
echo "next:"
echo "  1. fill in Question and Prediction BEFORE running anything"
echo "  2. run it, paste the exact command into Setup"
echo "  3. fill in Results / Interpretation / Decision / Threats"
echo "  4. add a row to experiments/README.md"
