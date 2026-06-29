# Mental State Bot: Product And Technical Plan

Version: 0.1  
Date: 2026-06-29  
Status: draft

## 1. Goal

Build a personal Telegram bot for a "living diary": short snapshots throughout the day, automatic AI interpretation, long-term memory, and calm summaries that help understand time, state, activities, gaps, and patterns.

This is not a commercial product and not a universal mental health app. It is a personal system optimized for one user.

The main product promise:

> Small moment snapshots during the day make the day visible again.

## 2. Product Principles

- The bot is not a therapist, coach, motivational companion, or AI friend.
- The bot should feel quiet, precise, contextual, and non-intrusive.
- The user should not manually rate mood, anxiety, energy, or other scales.
- Natural language is the primary input.
- Raw text must always be preserved.
- AI interpretation must be stored separately from raw data.
- Missing responses are data, not failure.
- Short responses such as "не знаю", "лежу", or "нічого" are valid.
- The bot may ask clarifying questions, but must keep interactions short.
- The bot must clearly signal when a snapshot session is complete.
- The user can always stop the current interaction.
- The bot should avoid repetitive questions, repetitive phrases, and robotic behavior.
- The bot should not give advice unless this becomes an explicit future feature.
- Live snapshot questions, clarifications, and micro-summaries should be AI-generated.
  Hardcoded text is acceptable only for explicit commands, button labels, system status,
  and emergency fallbacks when AI is unavailable.

## 3. Core UX Loop

Typical snapshot flow:

1. The scheduler chooses a floating time window.
2. The bot asks one natural-language question.
3. The question usually touches both:
   - current activity;
   - current state.
4. The user responds in natural language.
5. If the response is too vague, the bot may ask 1-2 clarifying questions.
6. Before clarification, the bot should make the boundary clear:
   - "Уточню одне і закрию..."
   - "Одне коротке уточнення..."
7. The user can stop with:
   - "записати як є";
   - "не хочу зараз";
   - "пізніше";
   - free-text equivalent.
8. The bot generates a short human micro-summary.
9. The bot clearly closes the session.

Example:

```text
Bot:
Що зараз відбувається і що було з минулого запису? Можна кількома словами.

User:
лежу, залип, нічого не зробив

Bot:
Уточню одне і закрию: це більше про втому, порожнечу, думки по колу, уникання чи щось інше?

User:
порожнеча і не можу почати

Bot:
Я почув, що ти зараз лежиш, з минулого запису був провал у дію, і це більше схоже на порожнечу та незапуск, ніж на конкретну тривогу. Зберіг цей зріз, на цьому все.
```

## 4. Interaction Types

### 4.1 Scheduled Snapshot

Triggered automatically 1-2 times per hour during active hours.

Important behavior:

- Timing should float, not happen at exact fixed minutes.
- The bot should consider recent responses, missed prompts, and time of day.
- The bot should avoid asking the same kind of question repeatedly.
- Snapshot frequency should be configurable.

### 4.2 Clarification

Used when the first response has too little information.

Clarification should try to extract maximum useful information quickly:

- activity;
- state;
- cause of stuckness;
- body signals;
- transition since previous snapshot;
- whether the answer describes literal inactivity or subjective "nothing happened".

Clarification should not become a conversation.

### 4.3 Manual Entry

The user can write to the bot at any time.

Behavior:

- store the entry;
- run the same AI analysis pipeline;
- optionally return a micro-summary;
- do not turn it into an open-ended chat.

### 4.4 Missed Prompt

If the user does not answer:

- store the prompt as unanswered;
- store time and prompt text;
- send one soft reminder later;
- ask for a free-text reason if useful, without forcing a fixed reason category;
- do not spam;
- represent the missing response as a data gap in summaries.

### 4.5 Sleep Marker

The bot should support a "лягаю спати" action.

Behavior:

- close the current day;
- generate or schedule the daily summary;
- mark the day boundary.

Edge cases:

- if the user falls asleep without pressing the button, generate the summary next morning;
- if the day is auto-closed in the morning, mark the boundary as `auto_morning`
  and `day_boundary_uncertain`;
- do not mechanically pretend the day is normal when the night was unusual.

### 4.6 Photo Attachment

Photos are optional memory artifacts, not primarily AI-analysis inputs.

Behavior:

- store photo metadata;
- store local or remote file reference;
- link photo to snapshot or manual entry;
- include photos in timeline/day archive.

## 5. Summaries

### 5.1 Daily Summary

Daily summary should be a calm analytical note.

Default short version:

- short story of the day;
- what actually happened;
- state changes;
- hardest interval;
- best or most stable interval;
- small pleasant/living moments;
- data gaps;
- cautious observations;
- data quality statement.

Detailed sections should be available behind buttons:

- timeline;
- coverage/gaps;
- charts;
- photos;
- patterns;
- raw entries;
- extracted features.

### 5.2 Weekly Summary

Weekly summary should be more analytical and less emotional.

It should include:

- repeated patterns;
- changes compared with previous week;
- sleep and day-boundary irregularities;
- repeated activity/state combinations;
- what seemed to help;
- what seemed to worsen the state;
- amount and distribution of missing data;
- cautious confidence notes.

### 5.3 Monthly Summary

Monthly summary should focus on larger trends:

- state trajectory;
- recurring loops;
- activity distribution;
- meaningful changes;
- repeated sources of improvement or decline;
- data quality across the month;
- notable days.

## 6. AI Task Design

The app should not call "one general AI chatbot". It should define explicit AI tasks.

Core tasks:

- `generate_snapshot_question`
- `generate_clarification_question`
- `extract_entry_features`
- `generate_micro_summary`
- `generate_daily_summary`
- `generate_weekly_summary`
- `generate_monthly_summary`
- `generate_semantic_memory_text`
- `find_relevant_context`

Each task should have:

- explicit input contract;
- explicit output contract;
- model selection;
- temperature/style settings;
- retry policy;
- token/cost logging;
- version identifier.

## 7. Model Strategy

Use model routing instead of one model for everything.

Initial default:

- DeepSeek `v4-flash` non-thinking for most live tasks.
- DeepSeek `v4-pro` for heavier summaries and difficult interpretation.
- Thinking mode off by default.
- Thinking mode only for slow background tasks where extra reasoning is useful.
- Separate embedding model for semantic memory.
- DeepSeek OpenAI-format requests should explicitly send thinking mode controls, because thinking mode can be provider-default enabled.

Suggested routing:

| Task | Model | Thinking |
| --- | --- | --- |
| Generate snapshot question | DeepSeek `deepseek-v4-flash` | off |
| Generate clarification | DeepSeek `deepseek-v4-flash` | off |
| Extract features | DeepSeek `deepseek-v4-flash` | off |
| Micro-summary | DeepSeek `deepseek-v4-flash` | off |
| Difficult/low-confidence extraction | DeepSeek `deepseek-v4-pro` | optional |
| Daily summary | DeepSeek `deepseek-v4-flash` first, `deepseek-v4-pro` if needed | off by default |
| Weekly/monthly summary | DeepSeek `deepseek-v4-pro` | optional |
| Embeddings | separate embedding provider | n/a |

Important:

- Do not send the entire diary to the model for every snapshot.
- Use compact context windows.
- Log cost and token usage per task from day one.
- Keep provider abstraction so DeepSeek can be replaced later.

## 8. Embeddings And Semantic Memory

Embeddings should be included in the architecture from the start.

Purpose:

- find similar past states;
- discover repeated patterns described in different words;
- support future questions such as "when was this similar before?";
- help summaries understand recurring states beyond keyword search;
- retrieve relevant context without sending the full diary.

Embeddings are not the source of truth. They are a memory/retrieval layer.

### 8.1 What To Embed

Do not embed only raw text. Generate a compact semantic representation first.

For each snapshot, create `semantic_memory_text`, for example:

```text
Time: afternoon.
Raw: лежу, залип, нічого не зробив.
Activity: lying down, inactive, phone unknown.
State: low energy, emptiness, inability to start.
Context: no clear anxiety, possible avoidance, data partial.
Micro-summary: ...
```

Embed:

- completed snapshots;
- manual entries;
- daily summaries;
- weekly summaries;
- later, stable patterns if needed.

### 8.2 Storage

Use PostgreSQL with `pgvector`.

Store:

- target type: entry, snapshot, day, week, pattern;
- target id;
- embedding model;
- dimensions;
- vector;
- source hash;
- created timestamp;
- embedding task version.

`source_hash` is required so changed semantic text can be re-embedded safely.

### 8.3 Retrieval Rules

Do not use embeddings in every live response.

Use retrieval when:

- generating daily summaries;
- generating weekly/monthly summaries;
- the current state resembles known recurring patterns;
- the user explicitly asks for similar past moments;
- AI needs historical context but full diary context is too large.

Retrieval should be logged:

- query text;
- embedding model;
- retrieved target ids;
- similarity scores;
- task that used retrieval.

### 8.4 Backfill

Embedding generation should support backfill.

Reason:

- the bot may be used before `EMBEDDING_API_KEY` is configured;
- embedding provider/model may change later;
- older entries should still become searchable.

Required behavior:

- command-line backfill for one Telegram user;
- limit parameter for safe batch sizes;
- skip entries that already have an embedding for the configured model;
- fail clearly if embeddings are disabled or the key is missing.

AI feature extraction should also support a repair/backfill command so older button actions,
imported entries, or entries created during AI outages can still appear in metrics, CSV exports,
and summaries.

## 9. Data Architecture

Use PostgreSQL as source of truth.

Use relational structure for time, identity, links, and analysis history. Use JSONB for flexible AI interpretation.

Main data concepts:

- user settings;
- days;
- snapshots;
- prompts;
- responses;
- manual entries;
- media;
- missed prompts;
- AI analyses;
- summaries;
- model runs;
- embeddings;
- retrieval logs;
- exports.

### 9.1 Suggested Tables

Initial tables:

- `users`
- `user_settings`
- `days`
- `snapshots`
- `snapshot_prompts`
- `entries`
- `media`
- `missed_prompts`
- `ai_analyses`
- `summaries`
- `model_runs`
- `embedding_records`
- `retrieval_logs`
- `exports`

### 9.2 Raw Entry

Store raw user input exactly as received:

- text;
- Telegram message id;
- timestamp;
- reply context;
- media references;
- source: scheduled snapshot, clarification, manual entry, sleep marker, button.

### 9.3 AI Analysis

Store AI output separately:

- target entry/snapshot/day;
- task name;
- schema version;
- model/provider;
- structured JSON result;
- confidence;
- uncertainty notes;
- created timestamp.

Example JSON shape:

```json
{
  "activity": {
    "labels": ["lying_down", "inactive"],
    "confidence": 0.74
  },
  "state": {
    "mood": {"value": "low", "confidence": 0.68},
    "energy": {"value": "very_low", "confidence": 0.81},
    "anxiety": {"value": "unclear", "confidence": 0.33},
    "emptiness": {"present": true, "confidence": 0.79}
  },
  "patterns": {
    "avoidance": {"present": true, "confidence": 0.61},
    "rumination": {"present": false, "confidence": 0.45},
    "inability_to_start": {"present": true, "confidence": 0.86}
  },
  "data_quality": "partial",
  "uncertainty": ["unclear whether phone was involved"]
}
```

## 10. Backend Components

### 10.1 Telegram Bot Layer

Responsibilities:

- receive Telegram updates;
- send prompts;
- send reminders;
- receive text/buttons/photos;
- route messages to the interaction engine;
- keep Telegram-specific logic out of business logic.

Recommended framework:

- Python + aiogram.

### 10.2 Interaction Engine

Responsibilities:

- determine current session state;
- decide whether message is a response, clarification, manual entry, or command;
- close sessions;
- create micro-summaries;
- avoid chat-like behavior.

### 10.3 Scheduler

Responsibilities:

- floating snapshot scheduling;
- reminder scheduling;
- sleep marker handling;
- morning fallback summaries;
- daily/weekly/monthly summary jobs.

### 10.4 AI Service

Responsibilities:

- model routing;
- prompt construction;
- structured output validation;
- retry/fallback;
- token/cost logging;
- provider abstraction.

### 10.5 Memory Service

Responsibilities:

- generate semantic memory text;
- create embeddings;
- store embeddings;
- retrieve similar moments;
- expose context snippets to summaries.

### 10.6 Summary Service

Responsibilities:

- generate daily/weekly/monthly summaries;
- include raw facts, AI features, gaps, and retrieved memory;
- produce short and detailed versions;
- maintain calm analytical style.

### 10.7 Export Service

Responsibilities:

- export raw entries;
- export media metadata;
- export AI analyses;
- export summaries;
- export as JSON first;
- later support Markdown/CSV bundles.

### 10.8 Deployment Layer

Deployment should be designed from the beginning, not added as an afterthought.

Primary target:

- Docker-based deployment;
- VPS hosting;
- public GitHub repository;
- public container image published through GitHub Packages / GitHub Container Registry.

The app should be easy to run in three modes:

- local development;
- local Docker Compose;
- production Docker Compose on a VPS.

Production deployment should not require manual Python setup on the server. The VPS should only need:

- Docker;
- Docker Compose;
- environment variables or `.env`;
- persistent volumes;
- access to the published container image.

The first production deployment can use long polling for Telegram updates. Webhooks can be added later if needed, especially if a reverse proxy and HTTPS setup becomes useful for other features.

Important deployment principles:

- one command should start the bot and required services;
- database data must live in a persistent Docker volume;
- media/photo files must live in a persistent Docker volume or mounted directory;
- secrets must not be baked into the image;
- logs should go to stdout/stderr so Docker can collect them;
- migrations should be easy to run during deploy;
- the container image should be reproducible from git state.

Suggested deployment artifacts:

- `Dockerfile`;
- `.dockerignore`;
- `docker-compose.yml` for local/dev;
- `docker-compose.prod.yml` for VPS;
- `.env.example`;
- GitHub Actions workflow for building and publishing image;
- optional deploy script for VPS pull/restart;
- basic healthcheck command;
- backup script for PostgreSQL and media files.

Container image naming convention:

```text
ghcr.io/<github-username>/mental-state-bot:<version>
ghcr.io/<github-username>/mental-state-bot:latest
```

The exact GitHub username/package path should stay configurable until the repository is created.

## 11. First Implementation Milestone

Goal: working personal bot that can collect snapshots and store data correctly.

Scope:

- project skeleton;
- configuration;
- PostgreSQL connection;
- migrations;
- Telegram bot basic loop;
- manual entry saving;
- scheduled prompt sending;
- response saving;
- minimal micro-summary;
- basic AI extraction;
- token/cost logging;
- Dockerfile and local Docker Compose;
- `.env.example`;
- no advanced charts yet.

Definition of done:

- bot can send a scheduled prompt;
- user can respond;
- raw entry is saved;
- AI analysis is saved separately;
- bot replies with a micro-summary;
- session closes;
- data can be inspected from database.
- project can be started locally with Docker Compose.

## 12. Milestone 2: Snapshot Quality

Goal: make interactions feel useful and non-robotic.

Scope:

- prompt intent system;
- varied question generation;
- clarification logic;
- "record as is" / "not now" / "later" buttons;
- missed prompt tracking;
- soft reminder;
- session closure variations;
- user settings for tone/frequency.
- quick pause/resume and frequency/reminder controls.
- body/photo prompt preferences are included in AI question context.

Definition of done:

- bot can run realistic 1-2 minute snapshot sessions;
- vague answers trigger bounded clarification;
- skipped prompts are stored as data;
- repeated questions are avoided.

## 13. Milestone 3: Daily Summary

Goal: make the day visible.

Scope:

- day model;
- sleep marker;
- morning fallback summary;
- daily summary generation;
- data gaps;
- hardest/stablest interval detection;
- pleasant/living moments;
- short summary in Telegram;
- detailed sections behind buttons.

Definition of done:

- pressing "лягаю спати" generates a useful daily summary;
- if the button is not pressed, summary appears next morning and the day boundary is marked uncertain;
- summary separates facts, interpretation, and uncertainty.

## 14. Milestone 4: Embeddings

Goal: add semantic memory.

Scope:

- `pgvector` setup;
- embedding provider interface;
- semantic memory text generation;
- background embedding jobs;
- embedding storage;
- similarity search;
- retrieval logs;
- first use in daily/weekly summaries.

Definition of done:

- each snapshot gets an embedding asynchronously;
- similar entries can be retrieved;
- summary generation can use retrieved context;
- current-period entries are excluded from retrieved memory context to avoid duplication;
- failures in embedding generation do not break normal bot behavior.

## 15. Milestone 5: Deployable VPS Version

Goal: make the bot easy to deploy and update on a VPS.

Scope:

- production Docker Compose;
- persistent PostgreSQL volume;
- persistent media volume;
- migration command;
- application healthcheck;
- structured logging;
- restart policy;
- GitHub Actions build;
- publish container image to GitHub Packages / GHCR;
- version tags;
- VPS deployment notes;
- backup/restore notes for database and media.

Definition of done:

- a clean VPS can run the bot without installing Python manually;
- a public GitHub package/container image is built automatically;
- production config uses environment variables;
- database and media survive container restarts/rebuilds;
- there is a documented update flow: pull image, run migrations, restart.

## 16. Milestone 6: Weekly And Monthly Analysis

Goal: make patterns visible.

Scope:

- weekly summary;
- monthly summary;
- comparison with previous periods;
- recurring activity/state patterns;
- "what helped" and "what worsened" extraction;
- confidence and data quality statements;
- use daily summaries + selected raw entries + semantic retrieval.

Definition of done:

- weekly summary is automatically generated;
- monthly summary is automatically generated;
- summaries avoid overclaiming when data is sparse.

## 17. Milestone 7: Export And Review

Goal: make the archive portable and inspectable.

Scope:

- JSON export;
- Markdown export;
- CSV export for entry-level metrics;
- ZIP media export bundle;
- archive/data coverage audit;
- raw text + AI analyses + summaries;
- model run/cost export;
- simple local review scripts or later web dashboard.

Definition of done:

- all personal data can be exported;
- raw and interpreted data remain distinguishable;
- exports are not tied to one model provider.

## 17. Cost Control

Cost must be measured, not guessed.

Implement from the beginning:

- model run table;
- input tokens;
- output tokens;
- reasoning tokens if available;
- provider;
- model;
- task name;
- estimated cost;
- latency;
- success/failure.

Rules:

- use non-thinking mode by default;
- do not send unnecessary historical context;
- use compact day state for live prompts;
- use full-day context only for daily summaries;
- use daily summaries instead of all raw entries for weekly/monthly summaries;
- use embeddings for retrieval when history grows.

## 18. Open Decisions

Still to decide:

- exact embedding provider;
- exact daily active hours;
- exact default snapshot frequency;
- exact reminder delay;
- exact initial AI schemas;
- whether to include a small local dashboard in early versions;
- where photos are stored in the first deployed version;
- deployment target.
- exact VPS provider;
- exact GitHub repository/package name;
- whether production uses long polling first or webhooks immediately;
- backup destination for database and media;
- whether the public repository should contain only app code or also deployment docs.

## 19. Immediate Next Steps

1. Create repository structure.
2. Add configuration template.
3. Add PostgreSQL migration setup.
4. Define initial database schema.
5. Implement Telegram bot shell.
6. Implement manual entry saving.
7. Implement scheduled prompt skeleton.
8. Implement AI provider abstraction.
9. Implement DeepSeek client.
10. Implement structured feature extraction.
11. Implement micro-summary generation.
12. Add model run/cost logging.
13. Add initial daily summary job.
14. Add embedding architecture and tables.
15. Add background embedding job.
16. Add Dockerfile and Docker Compose.
17. Add GitHub Actions image publishing.
18. Add VPS deployment notes.
19. Add backup/restore notes.

## 20. Guiding Constraint

The bot should remain boring in the right way:

> It should quietly preserve reality, not perform care.
