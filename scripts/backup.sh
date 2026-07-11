#!/usr/bin/env sh
set -eu

BACKUP_DIR="${BACKUP_DIR:-./backups}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$BACKUP_DIR"
docker compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U "${POSTGRES_USER:-mental_state_bot}" "${POSTGRES_DB:-mental_state_bot}" > "$BACKUP_DIR/db-$TIMESTAMP.sql"
docker compose -f "$COMPOSE_FILE" run --rm --no-deps bot tar -czf - /app/data/media > "$BACKUP_DIR/media-$TIMESTAMP.tar.gz"
echo "Backup written to $BACKUP_DIR"
