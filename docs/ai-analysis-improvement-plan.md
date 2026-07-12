# Active Improvement Backlog

This is the active backlog only. Completed work is recorded in
[`completed-work.md`](completed-work.md); it is not duplicated here.

## Operating Constraints

- Natural-language diary entries remain the primary input. Numeric controls are optional
  clarification fallbacks, never a required survey.
- AI works evidence-first, preserves uncertainty and must not invent current emotions,
  activities, metrics or personal facts.
- No deterministic diary-text keyword/root interpretation.
- Raw entries, corrections, photos, timestamps and prior analyses remain exportable source data.
- A new feature should reuse an existing AI call where possible and state its extra cost when it
  cannot.

## 1. Archive Refresh And Quality Audit

- Run a controlled archive reanalysis after material schema or prompt changes: small sample first,
  then selected days, then the full archive only after inspection.
- Build a repeatable audit of metric coverage, current-vs-mentioned emotions, corrections, and
  disagreement between manual calibration and AI interpretation.
- Use the audit to tune prompts and validators from evidence, not by adding word-based rules.
- Backfill older contextual capsules, embeddings and graph candidates after the new memory pipeline
  is stable enough to make doing so worthwhile.

## 2. Clarification Quality From Real Use

- Audit chains for repetition, stale references and low information gain.
- Improve the AI decision of whether another clarification is worth asking, using the full compact
  chain and current journal-day context.
- Keep one open conversational step at a time while allowing a useful adaptive chain; skipping must
  always remain easy and must not create a diary entry.
- Review the older clarification queue and keep only questions that can still improve a historical
  interpretation.

## 3. Metric And Emotion Reliability

- Inspect real examples where energy or mood stays unknown despite meaningful context or manual
  calibration, then improve evidence handling and contextual follow-ups.
- Audit free-text emotion corrections. Add a dedicated calibration call only if target-entry
  reanalysis demonstrably cannot normalize them safely.
- Review the controlled emotion and broader-affect vocabularies deliberately after archive audits;
  do not accept arbitrary labels into charts.
- Continue validating the readability and coverage semantics of emotion lanes and the day spectrum
  with real journal days.

## 4. Reports And Period Analysis

- Audit weekly and monthly reports for over-interpretation, weak comparisons and overly technical
  wording; tune thresholds and prompts from observed failures.
- Expose the grouped period views for custom date ranges directly, not only through the visual PDF.
- Consider a dedicated period turning-points visualization only after structured turning points have
  consistently reliable entry/time references.
- Add graph-backed situation candidates to deterministic period patterns only after their evidence
  quality has been audited.

## 5. Memory And Graph Follow-Up

- Backfill the historical archive through the current capsule/embedding/graph pipeline in a
  controlled, resumable run.
- Extend weekly graph review from duplicate candidates to evidence-backed contradiction candidates,
  while keeping uncertain changes in the same confirmation queue.
- Measure whether retrieved graph and semantic-memory context actually improves questions,
  summaries and corrections; remove or deprioritize paths that do not.
- Keep graph decay, duplicate consolidation and confirmation behavior under real-use audit so the
  graph stays an index of evidence rather than a second, stale biography.

## 6. Operations And Cost Visibility

- Keep the in-bot AI-cost report visible under `Дані` and distinguish estimated known-price calls
  from calls for which the configured provider/model has no pricing rule.
- Refresh pricing tables whenever the configured provider or model changes, and verify token
  semantics against provider responses before changing billing formulas.
- Monitor high-cost tasks after real usage and compact their payloads before introducing additional
  model calls.

## Done Criteria

A backlog item is complete only when its behavior is covered by focused tests, does not add a
hardcoded interpretation path, and has been checked against realistic exported data where the item
affects analysis quality.
