# Completed Work

This is a concise implementation record. It intentionally contains no diary content or personal
examples. The active backlog lives in [`ai-analysis-improvement-plan.md`](ai-analysis-improvement-plan.md).

## Diary Interaction

- Natural-language snapshots, manual entries, photos, voice transcription confirmation, sleep and
  wake handling, journal-day boundaries, quiet pause and missed-prompt behavior.
- Contextual snapshot questions with variable wording, day context and anti-spam scheduling.
- One-step-at-a-time conversational follow-ups that survive restarts and do not reset the snapshot
  rhythm for bookkeeping messages.
- Corrections rewrite the target interpretation instead of creating a second diary event; free text,
  menus and voice paths are supported where applicable.

## Analysis And Calibration

- Evidence-based mood, energy and emotion extraction with uncertainty, graphability rules and
  manual calibration that survives reanalysis.
- Controlled current-emotion vocabulary, separate broader affective states, per-emotion intensity,
  current-vs-mentioned distinction and targeted emotion correction.
- Contextual clarification chain, AI-generated options, free-text/voice answers and serialized
  follow-up delivery.
- Evening heavy-model review and structured daily, weekly and monthly analysis.

## Reports And Data

- Daily timelines, histories, metrics, photos, gaps, graphs, turning points and selected-day views.
- Weekly/monthly summaries with grouped views, period analysis, emotion dynamics and visual PDF
  reports.
- Raw exports, archive audit, emotion audit, controlled reanalysis workflow and AI usage recording.

## Semantic Memory And Graph

- Contextual memory capsules, embeddings, selective semantic retrieval and retrieval logs.
- Evidence-linked relational graph with nodes, edges, aliases, personal lexicon candidates,
  decay/staleness, visualization, JSON export/import and maintenance.
- Local/daily/weekly bounded duplicate review, embedding-assisted candidates and a shared
  confirmation queue for ambiguous merges.
- Manual and automatic graph-confirmation delivery only when the chat is free, with a cooldown and
  natural AI-generated options or free-text interpretation.

## Operations

- Docker/VPS deployment path, database migrations, backups, GitHub package deployment, diagnostics
  and focused test coverage for the interaction and analysis paths.
