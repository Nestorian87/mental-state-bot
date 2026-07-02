#!/usr/bin/env sh
set -eu

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
DOCKER_CONFIG="${DOCKER_CONFIG:-$(pwd)/.docker}"
mkdir -p "$DOCKER_CONFIG"
export DOCKER_CONFIG

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

repository_from_git_remote() {
  if ! command -v git >/dev/null 2>&1; then
    return 1
  fi
  remote_url="$(git remote get-url origin 2>/dev/null || true)"
  if [ -z "$remote_url" ]; then
    return 1
  fi
  case "$remote_url" in
    git@github.com:*)
      repo="${remote_url#git@github.com:}"
      repo="${repo%.git}"
      ;;
    https://github.com/*)
      repo="${remote_url#https://github.com/}"
      repo="${repo%.git}"
      ;;
    *)
      return 1
      ;;
  esac
  printf '%s' "$repo"
}

IMAGE_TAG="${IMAGE_TAG:-latest}"

if [ -z "${BOT_IMAGE:-}" ]; then
  if [ -n "${GITHUB_REPOSITORY:-}" ]; then
    BOT_IMAGE="ghcr.io/$(lowercase "$GITHUB_REPOSITORY"):$IMAGE_TAG"
  elif repository="$(repository_from_git_remote)"; then
    BOT_IMAGE="ghcr.io/$(lowercase "$repository"):$IMAGE_TAG"
  else
    echo "BOT_IMAGE is not set and repository name could not be detected." >&2
    echo "Set BOT_IMAGE or GITHUB_REPOSITORY, for example: GITHUB_REPOSITORY=owner/mental-state-bot sh scripts/deploy.sh" >&2
    exit 1
  fi
fi

export BOT_IMAGE

echo "Deploying $BOT_IMAGE"

docker compose -f "$COMPOSE_FILE" pull
docker compose -f "$COMPOSE_FILE" up -d --remove-orphans
docker compose -f "$COMPOSE_FILE" ps
