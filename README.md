# Mental State Bot

Personal Telegram bot for a living diary: short snapshots during the day, AI interpretation, summaries with optional semantic memory, exportable raw data, and embeddings.

The product is intentionally not a therapist, coach, or motivational companion. It quietly preserves moments and helps make days visible.

## Current Status

This repository is being built as a working personal tool. The initial architecture includes:

- Telegram bot interface;
- PostgreSQL as source of truth;
- JSONB AI analyses;
- DeepSeek/OpenAI-compatible chat client abstraction;
- embedding provider abstraction with `pgvector`;
- scheduled snapshots, reminders, and morning fallback summaries;
- daily, weekly, and monthly summaries;
- Docker/VPS deployment path;
- GitHub Container Registry publishing path.

See [docs/project-plan.md](docs/project-plan.md) for the product and technical plan.

Default AI routing uses DeepSeek `deepseek-v4-flash` for live interactions and `deepseek-v4-pro` for heavier summaries. Live tasks explicitly disable DeepSeek thinking mode to keep latency and cost low.

## Local Development

1. Copy environment example:

   ```bash
   cp .env.example .env
   ```

2. Fill in at least:

   ```text
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_ALLOWED_USER_IDS=<your-telegram-user-id>
   AI_API_KEY=...
   ```

   For the first playable run, `EMBEDDING_API_KEY` is optional. If you do not have one yet,
   set `EMBEDDINGS_ENABLED=false` to keep `doctor` quiet.

3. Start PostgreSQL:

   ```bash
   docker compose up -d postgres
   ```

4. Install and run:

   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -e ".[dev]"
   mental-state-bot migrate
   mental-state-bot run
   ```

5. In Telegram, try:

   ```text
   /start
   /snapshot
   лежу, залип, важко почати
   /today
   /gaps
   лягаю спати
   ```

### Local PostgreSQL Troubleshooting

If `mental-state-bot migrate` says `role "mental_state_bot" does not exist`, your
`DATABASE_SYNC_URL` points to a PostgreSQL server that does not have the app role/database.
This often happens when a local PostgreSQL is already listening on `localhost:5432`.
The compose database is exposed on `localhost:5433` by default to avoid that conflict:

```bash
docker compose up -d postgres
```

Make sure local app URLs use port `5433`:

```text
DATABASE_URL=postgresql+asyncpg://mental_state_bot:mental_state_bot@localhost:5433/mental_state_bot
DATABASE_SYNC_URL=postgresql+psycopg://mental_state_bot:mental_state_bot@localhost:5433/mental_state_bot
```

If you want to use an existing local PostgreSQL instead, create the role and database first:

```bash
createuser mental_state_bot
createdb mental_state_bot -O mental_state_bot
```

The initial migration also needs the `pgvector` extension to be available in that PostgreSQL
installation. The compose image already includes it.

## Docker

Local Docker run for the full bot:

```bash
docker compose up --build
```

The container runs migrations before starting the bot. For a faster first test, start only
PostgreSQL and run the app locally as shown above.

Production is intended to use `docker-compose.prod.yml` on a VPS with an image published to
GitHub Container Registry. The GitHub Actions workflow can build, push, copy deployment files
to the VPS, and run the deploy script automatically. Runtime secrets stay in `.env` on the VPS; see
`docs/deployment.md`.

## Telegram Commands

- `/start` - register the personal bot user and show main actions.
- `/help` - show available commands.
- `/status` - check that the bot is alive.
- `/snapshot` - start a snapshot right now.
- `/today` - show today's raw timeline-style view.
- `/metrics` - show extracted mood/energy sparklines, labels, and data quality.
- `/photos` - show today's photo moments as a separate lightweight photo strip.
- `/gaps` - show today's coverage, notable pauses, and missed prompts.
- `/raw` - show today's raw entries.
- `/report` - show timeline and metrics together.
- `/costs` - show model usage and estimated cost for the last 7 days.
- `/audit` - show archive coverage and maintenance suggestions.
- `/settings` - show active hours, snapshot frequency, reminder delay, and command hints.
- `стиль: коротко, без підбадьорювання, але не сухо` - save a custom natural-language interaction style.
- `скинути стиль` - remove the custom interaction style.
- `/pause` - pause automatic snapshots without disabling manual entries or summaries.
- `/resume` - resume automatic snapshots.
- `/set_active 09:00 23:30` - update active hours.
- `/set_frequency 30 70` - update floating snapshot interval in minutes.
- `/set_reminder 25` - update soft reminder delay in minutes.
- `/summary` - generate a summary for today.
- `/sleep` - close the day and generate a summary. The exact message `лягаю спати` does the same.
- `/week` - generate a summary for the current week.
- `/month` - generate a summary for the current month.
- `/export` - export the archive as JSON.
- `/export_md` - export a readable Markdown archive.
- `/export_csv` - export entry-level AI metrics as CSV.
- `/export_zip` - export a portable bundle with JSON, Markdown, CSV, and local media files.

When a snapshot reminder appears, the bot can ask you to write a short free-text reason for the
missed moment. Send it as `причина: ...`; it is stored as diary data and appears in `/gaps`
and daily summaries. The bot does not force a fixed reason category.

## Maintenance Commands

- `mental-state-bot migrate` - apply database migrations.
- `mental-state-bot doctor` - check local configuration readiness.
- `mental-state-bot export <telegram-user-id>` - export archive JSON.
- `mental-state-bot export <telegram-user-id> --output ./data/export.md --format markdown` - export readable Markdown.
- `mental-state-bot export <telegram-user-id> --output ./data/metrics.csv --format csv` - export flat metrics CSV.
- `mental-state-bot export <telegram-user-id> --output ./data/archive.zip --format zip` - export a portable bundle with data and media.
- `mental-state-bot user-audit <telegram-user-id>` - print archive coverage and maintenance suggestions.
- `mental-state-bot features-backfill <telegram-user-id> --limit 100` - generate missing AI feature analyses for old or incomplete entries.
- `mental-state-bot features-backfill <telegram-user-id> --limit 100 --force` - re-run AI feature extraction for existing entries, useful after changing extraction prompts.
- `mental-state-bot embed-backfill <telegram-user-id> --limit 100` - generate missing embeddings for old entries.
- `mental-state-bot embed-backfill <telegram-user-id> --limit 100 --force` - rebuild entry embeddings after re-analysis or corrections.

Telegram voice messages are supported when `TRANSCRIPTION_API_KEY` is configured. The default
model is `gpt-4o-mini-transcribe`; the bot stores the original voice note, asks you to confirm
or edit the transcript first, and only then saves the diary entry and runs the normal AI reply.

## Data Ownership

Raw text, photos, AI analysis, embeddings metadata, model usage, and summaries are stored separately so exports can preserve both original data and model-derived interpretation.

JSON exports use `archive.v2` and include raw entries, days, snapshots, prompts, missed prompts, media metadata, AI analyses, summaries, model runs/cost metadata, embedding metadata, retrieval logs, user settings, and export history. Markdown exports render the same archive into a readable day-by-day document. CSV exports flatten entry-level AI metrics for spreadsheets and later charts. ZIP exports bundle `archive.json`, `archive.md`, `metrics.csv`, an export manifest, and any locally available media files. Embedding vectors themselves are not exported by default; their source text, model, dimensions, and source hash are included so they can be regenerated.

When embeddings are enabled, daily, weekly, and monthly summaries can include a compact `semantic_memory` section with similar older moments. Current-period entries are filtered out so this context acts as memory rather than duplication.

The settings panel controls snapshot question behavior too: body-signal prompts can be disabled, and photo prompts are treated as occasional optional alternatives rather than required input. The AI receives an occasional photo opportunity flag, so it can sometimes ask for a photo in natural language without turning every snapshot into a photo request.

Interaction style can be customized in natural language instead of only choosing presets. Open settings and use `Власний стиль`, or send a message like `стиль: коротко, без підбадьорювання, але не сухо; якщо відповідь нечітка — уточнюй м’яко`. This text is passed to AI question generation, clarifications, and micro-summaries. Use `скинути стиль` to remove it.

The main actions use a compact Telegram reply keyboard at the bottom of the chat. Inline buttons are reserved for short-lived contextual actions such as an active snapshot, summary details, or settings. Metrics are shown in Ukrainian and, when enough data exists, `/metrics` also sends a small PNG chart for mood and energy. Photos are kept as a separate view through `/photos`, the `Фото дня` keyboard action, or the `Фото дня` button in summary details, so they do not get mixed into text summaries or metric charts.
