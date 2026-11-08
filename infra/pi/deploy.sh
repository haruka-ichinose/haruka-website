#!/usr/bin/env bash
# Deploy the RAG service on the Pi: pull, rebuild, restart, health-check.
#
#   infra/pi/deploy.sh            # deploy origin/main
#   infra/pi/deploy.sh my-branch  # deploy another branch (e.g. to test a PR)
#
# Run on the Pi from anywhere inside the repo checkout. Needs, next to this
# script: backend.env (copy backend.env.example) and .env with POSTGRES_PASSWORD
# and CLOUDFLARE_TUNNEL_TOKEN (read by compose for variable substitution).
#
# Not done here: rebuilding the RAG index. That runs from the laptop
# (`cd llm && uv run python -m mecha_llm.build`) against the Pi's Postgres, and
# should run *before* deploying a backend that expects new source types.

set -euo pipefail

BRANCH="${1:-main}"
PI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(git -C "$PI_DIR" rev-parse --show-toplevel)"
COMPOSE=(docker compose --project-directory "$PI_DIR" -f "$PI_DIR/compose.yaml")
HEALTH_TIMEOUT=180  # seconds; first start downloads + loads the embedding model

log() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# --- Preflight -------------------------------------------------------------
[[ -f "$PI_DIR/backend.env" ]] || die "missing $PI_DIR/backend.env (copy backend.env.example)"
[[ -f "$PI_DIR/.env" ]] || die "missing $PI_DIR/.env (POSTGRES_PASSWORD, CLOUDFLARE_TUNNEL_TOKEN)"
for var in POSTGRES_PASSWORD CLOUDFLARE_TUNNEL_TOKEN; do
  grep -qE "^${var}=.+" "$PI_DIR/.env" || die "$var is not set in $PI_DIR/.env"
done
grep -qE '^RAG_PROXY_TOKEN=.+' "$PI_DIR/backend.env" \
  || die "RAG_PROXY_TOKEN is empty in backend.env — /chat would be open to the internet"

cd "$REPO_DIR"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] \
  || die "uncommitted changes in $REPO_DIR; commit or stash them first"

# --- Update code -----------------------------------------------------------
PREV_COMMIT="$(git rev-parse --short HEAD)"
log "Fetching origin/$BRANCH"
git fetch --prune origin "$BRANCH"
git checkout -q "$BRANCH" 2>/dev/null || git checkout -q -b "$BRANCH" --track "origin/$BRANCH"
git merge --ff-only -q "origin/$BRANCH" || die "local $BRANCH has diverged from origin/$BRANCH"
NEW_COMMIT="$(git rev-parse --short HEAD)"
log "Deploying $BRANCH @ $NEW_COMMIT (was $PREV_COMMIT)"

# --- Build + restart -------------------------------------------------------
# `up` alone never rebuilds, so build explicitly first.
log "Building backend image"
"${COMPOSE[@]}" build backend
log "Starting services"
"${COMPOSE[@]}" up -d --remove-orphans

# --- Health check ----------------------------------------------------------
# The backend publishes no ports, so probe from inside its container.
log "Waiting for /health (up to ${HEALTH_TIMEOUT}s)"
deadline=$((SECONDS + HEALTH_TIMEOUT))
until "${COMPOSE[@]}" exec -T backend python -c \
  "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" \
  >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    "${COMPOSE[@]}" logs --tail 50 backend
    die "backend not healthy after ${HEALTH_TIMEOUT}s. To roll back:
    git -C $REPO_DIR checkout $PREV_COMMIT && ${COMPOSE[*]} up -d --build"
  fi
  sleep 3
done

# --- Report ----------------------------------------------------------------
logs="$("${COMPOSE[@]}" logs --no-log-prefix backend 2>&1 | tail -200)"
grep -q "Content Safety: ENABLED" <<<"$logs" \
  && log "Content Safety: ENABLED" \
  || printf '\033[33mwarning: Content Safety is not enabled (check AZURE_CONTENT_SAFETY_* in backend.env)\033[0m\n'

log "Pruning dangling images"
docker image prune -f >/dev/null

"${COMPOSE[@]}" ps
log "Deployed $NEW_COMMIT. Previous: $PREV_COMMIT"
