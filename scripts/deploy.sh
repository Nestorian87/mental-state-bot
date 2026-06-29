#!/usr/bin/env sh
set -eu

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"

docker compose -f "$COMPOSE_FILE" pull
docker compose -f "$COMPOSE_FILE" up -d
docker compose -f "$COMPOSE_FILE" exec bot mental-state-bot migrate
docker compose -f "$COMPOSE_FILE" restart bot
docker compose -f "$COMPOSE_FILE" ps
