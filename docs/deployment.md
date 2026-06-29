# Deployment

The intended production target is a small VPS running Docker and Docker Compose.

## GitHub Container Registry

When the repository is public on GitHub, `.github/workflows/docker-publish.yml` publishes images to:

```text
ghcr.io/<owner>/<repo>:latest
ghcr.io/<owner>/<repo>:sha-...
ghcr.io/<owner>/<repo>:vX.Y.Z
```

For this repository the expected default is:

```text
ghcr.io/<owner>/mental-state-bot:latest
```

## VPS Setup

Install Docker and copy these files to the server:

- `docker-compose.prod.yml`
- `.env`
- `scripts/deploy.sh`
- `scripts/backup.sh`

Set in `.env`:

```text
BOT_IMAGE=ghcr.io/<owner>/mental-state-bot:latest
POSTGRES_PASSWORD=<strong-password>
TELEGRAM_BOT_TOKEN=<telegram-token>
TELEGRAM_ALLOWED_USER_IDS=<your-telegram-user-id>
AI_API_KEY=<deepseek-or-compatible-key>
EMBEDDING_API_KEY=<embedding-key>
```

`EMBEDDING_API_KEY` is optional for the first run. If it is not configured, set
`EMBEDDINGS_ENABLED=false`; the bot will still collect entries, analyze them, and generate
summaries, but semantic search/memory will stay off until embeddings are enabled.

Start:

```bash
docker compose -f docker-compose.prod.yml up -d
```

Update:

```bash
sh scripts/deploy.sh
```

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
