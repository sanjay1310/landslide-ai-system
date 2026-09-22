#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example — review settings before production use."
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Start Docker Desktop, then rerun this script."
  exit 1
fi

docker compose build
docker compose up -d

API_PORT="${LANDSLIDE_API_PORT:-8000}"
DASHBOARD_PORT="${LANDSLIDE_DASHBOARD_PORT:-8501}"

echo
echo "Landslide AI System is deploying..."
echo "  API:       http://localhost:${API_PORT}"
echo "  API docs:  http://localhost:${API_PORT}/docs"
echo "  Dashboard: http://localhost:${DASHBOARD_PORT}"
echo
echo "Check status: docker compose ps"
echo "View logs:    docker compose logs -f"
