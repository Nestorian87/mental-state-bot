# Deployment

The intended production target is a small VPS running Docker and Docker Compose.

## GitHub Container Registry

When the repository is pushed to GitHub, `.github/workflows/docker-publish.yml` builds and
publishes images to GitHub Container Registry. The workflow lowercases the repository name
before using it as a package name, so `Owner/Mental-State-Bot` becomes:

```text
ghcr.io/owner/mental-state-bot:latest
ghcr.io/owner/mental-state-bot:sha-...
ghcr.io/owner/mental-state-bot:vX.Y.Z
```

On pushes to `main`, and on manual `workflow_dispatch` runs, the workflow can also deploy
to the VPS automatically.

## GitHub Secrets And Variables

Required repository secrets:

- `VPS_HOST` - VPS hostname or IP.
- `VPS_USER` - SSH user on the VPS.
- `VPS_SSH_KEY` - private SSH key that can connect to the VPS.

Optional repository secret:

- `GHCR_USERNAME` and `GHCR_TOKEN` - only needed if the GHCR package is private.
  `GHCR_TOKEN` should be a GitHub token with `read:packages`.

Optional repository variables:

- `DEPLOY_PATH` - remote directory, default: `mental-state-bot`.
- `VPS_PORT` - SSH port, default: `22`.

## VPS Setup

Install Docker and Docker Compose on the VPS. The GitHub workflow will copy these files
on every deploy:

- `docker-compose.prod.yml`
- `scripts/deploy.sh`
- `scripts/backup.sh`

Create `.env` once on the VPS. The workflow checks that this file exists, but it does not
write or overwrite it. App secrets such as Telegram, AI, database password, and embeddings
stay only in the VPS `.env`.

`BOT_IMAGE` is derived automatically from the lowercased GitHub repository name by
`scripts/deploy.sh`, so you do not need to put it in `.env` unless you want to override it.

If the GitHub Container Registry package is public, the VPS can pull it without login.
If it is private, set `GHCR_USERNAME` and `GHCR_TOKEN`; the workflow will run `docker login`
on the VPS before pulling the image.

## Running Beside Another Project

This bot uses Telegram long polling, so it does not need a domain, HTTPS certificate,
reverse proxy route, or open inbound HTTP port. The VPS only needs outbound internet
access so the bot container can call Telegram and the AI APIs.

It can run beside another Docker project on the same VPS. Production compose sets an
explicit project name, `mental-state-bot`, and does not publish any ports. PostgreSQL is
only reachable inside the Compose network. Persistent data is kept in this project's own
Docker volumes:

- `mental-state-bot_postgres_data`
- `mental-state-bot_media_data`

If another project already uses the same Compose project name, change `name:` in
`docker-compose.prod.yml` before deploying.

Example VPS `.env`:

```text
POSTGRES_DB=mental_state_bot
POSTGRES_USER=mental_state_bot
POSTGRES_PASSWORD=<strong-password>
TELEGRAM_BOT_TOKEN=<telegram-token>
TELEGRAM_ALLOWED_USER_IDS=<your-telegram-user-id>
APP_TIMEZONE=Europe/Kyiv
AI_PROVIDER=deepseek
AI_BASE_URL=https://api.deepseek.com
AI_API_KEY=<deepseek-or-compatible-key>
AI_LIVE_MODEL=deepseek-v4-flash
AI_HEAVY_MODEL=deepseek-v4-pro
AI_LIVE_THINKING=false
AI_HEAVY_THINKING=false
EMBEDDINGS_ENABLED=false
EMBEDDING_API_KEY=<embedding-key>
```

`EMBEDDING_API_KEY` is optional for the first run. If it is not configured, set
`EMBEDDINGS_ENABLED=false`; the bot will still collect entries, analyze them, and generate
summaries, but semantic search/memory will stay off until embeddings are enabled.

Start:

```bash
GITHUB_REPOSITORY=<owner>/mental-state-bot sh scripts/deploy.sh
```

Update:

```bash
GITHUB_REPOSITORY=<owner>/mental-state-bot sh scripts/deploy.sh
```

If the directory is a git checkout with an `origin` pointing to GitHub, `scripts/deploy.sh`
can infer the repository name from that remote. Otherwise pass `GITHUB_REPOSITORY`.

Backup:

```bash
sh scripts/backup.sh
```

## Data Persistence

Production compose uses two persistent volumes:

- `postgres_data` for PostgreSQL;
- `media_data` for Telegram photos/media.

Secrets are read from `.env` and must never be baked into the image.

## Telegram Mode

The first production version uses long polling. This avoids requiring a public HTTPS endpoint. Webhooks can be added later if the VPS gets a reverse proxy and TLS setup.
