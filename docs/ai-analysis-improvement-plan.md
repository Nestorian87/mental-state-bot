# AI Analysis Improvement Plan

This document tracks remaining improvements for making diary analysis more accurate, less noisy,
and more useful without turning the bot into a manual survey.

## Product Goal

The bot should keep accepting natural language entries, while AI-derived metrics stay evidence-based:

- do not treat activity alone as certain mood or energy evidence; allow only explicitly low-confidence energy hypotheses when the described effort, pace, or ability supports one;
- preserve uncertainty instead of guessing;
- make graphs reflect only reliable points;
- ask for clarification only when it will materially improve the day picture;
- keep interaction lightweight and non-therapeutic;
- avoid deterministic keyword/root interpretation of diary text.

## Already Implemented

These are no longer backlog items, but they matter as constraints for future work.

### Evidence-Based Entry Features

Implemented:

- `entry_type` for entry analysis;
- evidence fields for mood, energy, anxiety;
- reasoning types for metric inference;
- `should_graph_mood` and `should_graph_energy`;
- conservative post-processing that clears unsupported metrics;
- graphable-vs-total metric reporting;
- separation between moment metrics and day-story material.

Current behavior:

- activity-only, photo-only, sleep/dream fragments, reply fragments, and command/system-like entries are much less likely to create graph points;
- mood/energy graph points require direct or manual evidence;
- reports explain when graphs are sparse.

### Emotion Intensity Foundation

Implemented:

- structured `emotions` with `label`, `intensity_level`, `intensity`, `confidence`, `evidence`, and `time_scope`;
- `non_emotional_states` and `mentioned_but_not_felt`;
- extraction prompt guidance for distinguishing current emotion from story/dream/remembered emotion;
- coarse intensity buckets: `trace`, `mild`, `moderate`, `strong`, `overwhelming`;
- post-processing that drops current emotions without evidence or with too-low confidence;
- compatibility fallback for older `emotion_labels`;
- manual emotion correction now writes structured emotion records;
- manual emotion correction uses the controlled emotion list, can save a separate intensity for each selected emotion, and can mark emotions as mentioned rather than current;
- entry analysis records whether the emotional moment is observed, explicitly without a current emotion, or genuinely unclear;
- new analyses receive a compact context of recent evidenced emotions from the same journal day, only to assess a possible transition;
- when AI judges a meaningful emotional transition unclear, it can ask one contextual follow-up while the moment is still current instead of silently losing the point;
- contextual clarification questions can carry AI-generated answer buttons while retaining free text and voice as equivalent inputs;
- an answer is replayed against the original entry together with the exact clarification context, so it updates the target analysis rather than becoming a separate diary record;
- the emotion chart draws each selected emotion through observed moments, uses zero only for emotions absent in an observed moment, and leaves genuinely unknown moments as visible gaps.
- strict emotions are now separated from controlled broader affective states; the latter stay visible in reports but do not become emotion-chart series.

Important remaining limitation:

- old entries still need deliberate reanalysis before the new emotion model becomes broadly useful.

### Backfill and Audit Foundation

Implemented:

- entry feature backfill exists and is resumable by count/date-like operational usage;
- old analyses remain readable;
- tests cover conservative metric post-processing and structured emotion filtering.
- guided menu workflow supports a small recent sample, recent journal days, an explicit date range, or the full archive;
- every run shows the selected count before confirmation and keeps older AI analyses as historical versions;
- completion compares before/after coverage of mood, energy, and observed emotion points, plus the number of updated interpretations.
- explicit user-calibrated mood and energy values survive later AI corrections and archive reanalysis, including when an intermediate AI version was already saved.

### Semantic Memory Foundation

Implemented foundation:

- entries can be embedded;
- new entry embeddings are generated from contextual memory capsules rather than raw text alone;
- capsules include entry text, snapshot prompt, local day window, AI features, micro-summary, corrections, and live context;
- new capsules can also update a relational life graph with nodes, edges, weights, confidence, status, and evidence;
- similar records can be retrieved for some bot contexts and manual search;
- memory is available as context without being treated as current-day fact.

Remaining:

- rebuild/backfill old embeddings so existing records also use contextual capsules;
- rebuild/backfill old records so the graph is populated from the archive;
- automatic situation clusters beyond single-entry graph candidates;
- explicit usefulness/decay scoring;
- debug view showing which memories influenced a question or summary.

Detailed lifecycle, cost boundaries, live-context relationship, and implementation stages are tracked in
[`context-memory-graph-plan.md`](context-memory-graph-plan.md).

### Contextual Life Graph Foundation

Implemented foundation:

- PostgreSQL-backed graph tables: `memory_nodes`, `memory_edges`, `memory_evidence`;
- soft node model with `label`, `kind`, `aliases`, `summary`, `weight`, `confidence`, `status`;
- soft edge model with free-form `relation_label`, `summary`, `weight`, `confidence`, `status`;
- evidence records linked back to source entries;
- AI graph extraction from contextual memory capsules;
- conservative upsert by normalized labels to avoid simple duplicates;
- bot menu action to rebuild memory, which recreates capsules, embeddings, and graph candidates for recent entries.

Remaining:

- embedding-assisted node resolution for near-duplicates, not only exact normalized labels;
- context lookup by uncertainty during question/micro-summary generation;
- full AI-assisted graph merge/contradiction review;
- user-facing confirmation flow for sensitive or uncertain graph merges.

Implemented for inspection:

- memory menu can export the current graph as an interactive HTML file;
- the view supports filtering, zooming, dragging, edge labels, and a plain-text fallback list;
- the visualization is explicitly diagnostic and does not present graph candidates as verified facts.

## Active Backlog

## Priority 1: Clarification Queue

Current implementation:

- uncertainty about the entry that has just arrived is handled immediately in the same composed reply, rather than being delayed until the moment is stale;
- the bot keeps at most one active clarification at a time; after an answer it may ask another only when AI finds a different, material uncertainty;
- no fixed per-day clarification cap is used;
- historical/reviewer questions stay in the visible queue and are opened deliberately from the menu, never pushed by a timer as if they described the current moment;
- a deferred question can be answered by text or confirmed voice transcription, or skipped without creating an entry;
- meaningful uncertainty is evaluated for both snapshot answers and deliberately saved free-form entries;
- once the system has decided a clarification is useful, the question model only formulates it and cannot silently veto it a second time;
- question options are AI-generated when natural, including for a grouped queue question, while text and voice remain equivalent answers;
- reanalysis does not create an automatic conversational follow-up;
- queued items retain their status and target entry in user settings for export/debugging.
- the main menu has a visible clarification queue/review screen;
- queue creation skips already-active or recently-resolved similar questions.
- one entry response now produces one composed message with at most one active next step; a contextual clarification takes priority over dry interpretation and numeric calibration;
- numeric metric buttons are a fallback when contextual AI clarification finds no useful natural question;
- an active clarification or metric/emotion follow-up is persisted in settings, so scheduled snapshots wait for it, including across a bot restart/deploy;
- scheduled snapshots and active follow-ups are serialized per user in the bot process, preventing competing prompts at the same time;
- each entry-linked clarification keeps its question, answer, source, focus and expected information gain; later AI steps receive this compact chain rather than only the newest reply;
- the same AI clarification call decides whether another question has material new value; it can end the chain without an arbitrary numerical cap;
- entry-feature reanalysis receives the full ordered correction history, where newer user corrections override earlier interpretations and the original wording;
- correction/profile bookkeeping entries are excluded from the cadence activity clock, so answering a deferred clarification does not falsely reset the rhythm of new snapshots.

Target behavior:

- ask contextual questions while the referenced moment is still current;
- allow an adaptive chain of questions, but only one open question and only while each answer reveals a new important gap;
- collect older, reviewer-created uncertainty into a queue for deliberate review rather than surprise delivery;
- always allow "не хочу уточнювати" / "пропустити";
- preserve uncertainty when skipped instead of guessing.

Remaining refinement:

- tune contextual question generation from real usage where questions still feel generic or repetitive;
- consider a lightweight review view that explains why an older question remains in the queue.

Examples:

> Сьогодні було кілька записів про прогулянки без опису сил. Це були прогулянки з нормальними силами, чи радше попри втому?

> Ти пишеш, що "не дуже сильно тривожить". Це радше легка тривога фоном, чи вона все ж помітно впливає на тіло/дії?

Design constraints:

- no generic repeated "яка зараз емоція?";
- generate questions from the entry, day context, corrections, and personal context;
- buttons are useful when there are obvious options, but free text must always work;
- do not ask only because one graph metric is unclear; do ask when a meaningful, sufficiently described moment has a gap that a short contextual answer could resolve.

## Priority 2: Personalized Emotion Calibration

Current state:

- AI can extract structured emotion intensity;
- user can correct multiple emotion labels from the controlled list and set a separate intensity level for each one;
- graph can use structured intensity.
- the emotion calibration screen also offers `Описати словами`; this opens a targeted correction for the same entry rather than creating a new diary record, and the existing evidence-aware AI reanalysis normalizes the explanation into the controlled model or keeps it outside current emotions.

Remaining refinement:

- audit real free-text emotion corrections; add a separate calibration-only AI call only if the existing target-entry reanalysis is demonstrably not precise enough to normalize ambiguous affective descriptions.

Target UX:

- correction UI supports multi-select emotions;
- selected emotions can optionally get intensity: `ледь фоном`, `слабко`, `помірно`, `сильно`, `дуже сильно`;
- user can mark "це не зараз, я лише згадував/розповідав про це";
- if the user explains a more complex affective correction in words, AI should normalize it into the controlled affective vocabulary or mark it as adjacent/non-emotional before storing.

Good calibration moments:

- after repeated uncertainty;
- when AI cannot confidently separate current emotion from remembered/story emotion;
- when emotion is clear but intensity would materially affect graph/summaries;
- during evening review;
- not after most entries.

Avoid:

- daily mandatory ratings;
- long questionnaires;
- asking the same generic emotion/intensity question repeatedly.

## Priority 3: Evening Reviewer

Implemented as a daily reviewer step immediately before daily summary generation.

Current behavior:

- entries with low confidence;
- entries where mood/energy/anxiety were cleared by validators;
- contradictions;
- activity-energy confusion;
- dream/sleep entries;
- reply fragments without clear context;
- labels that look too broad, duplicated, or outside the controlled affective vocabulary;
- emotion entries where current/story scope may be wrong.
- it runs as one dedicated evening AI call and uses the configured heavy/thinking route;
- it can patch only a small allowlisted subset of uncertain emotion/affective fields, only with direct evidence and high confidence;
- applied patches replace the stale entry-feature view in the same daily-summary context, so the summary actually reflects the review;
- it may return up to two evidence-backed, entry-linked question candidates in the same call, without another question-generation request;
- only candidates with direct evidence, high confidence and an existing target entry enter the existing deferred clarification queue; they remain optional and can use generated buttons or free text;
- its raw result, patches and unresolved items remain stored as a day-level AI analysis for audit.

Structured output:

```json
{
  "corrections": [],
  "uncertain_items": [],
  "questions_for_user": [],
  "daily_metric_suggestions": {}
}
```

The reviewer patches only uncertain fields rather than blindly regenerating the whole day.

Model choice:

- use the heavy/thinking route because this is the final quality gate for a day;
- keep its input compact and reuse its structured question candidates instead of making extra calls.

Remaining tuning:

- inspect real reviewer patches and queued questions for overreach, repetition or low practical value;
- connect its evidence-backed memory-maintenance notes to the existing graph maintenance path only after auditing that the notes are useful enough to justify influence on memory;
- add a compact audit/debug view if real-use diagnosis needs it.

## Priority 4: Personal Lexicon

Implemented as a special `lexicon` node type in the contextual life graph, rather than a separate dictionary.

Implemented behavior:

- the existing capsule-and-graph AI call may create a lexicon node only for a short phrase that appears in the user's text and has direct contextual/correction evidence;
- nodes store the literal phrase, a conditional context-specific summary, evidence links, confidence, weight and recency;
- structural validation caps automatic confidence/weight and prevents a lexicon node from becoming `confirmed` through AI alone;
- relevant lexical nodes are returned as a clearly marked conditional part of the small graph context for questions, entry analysis and summaries;
- normal graph decay makes unused candidates stale without deleting their evidence;
- `Пам’ять → Фрази й значення` exposes the current candidates without dumping raw diary entries.

This is not a universal translation table and does not use keyword/root rules to infer diary meaning.

Sources:

- user corrections;
- repeated clarification answers;
- explicit calibration;
- recurring phrases that often produce uncertainty.

Example:

```json
{
  "phrase": "порожньо",
  "meaning": "often means emptiness / low emotional presence, not neutral mood",
  "confidence": 0.8,
  "needs_context": true
}
```

Remaining refinement:

- do not hardcode this as permanent truth;
- let stale lexicon items decay if unused or contradicted;
- ask before adding ambiguous interpretations;
- keep it small and useful;
- use real data after a memory rebuild to decide whether candidate promotion or a user confirmation flow is needed.

## Priority 5: Reanalysis Workflow (Implemented)

After schema changes, old analyses should be upgradable in a controlled way.

Workflow:

1. Reanalyze recent entries first.
2. Compare old vs new on a small audit CSV/report.
3. Check for metric drift, hallucinated emotions, and context contamination.
4. Only then reanalyze the full archive.

Implemented flow:

- menu action with a confirmation after the exact selected count is known;
- sample of the 10 most recent entries, recent journal-day presets, custom date range, or explicit full archive;
- limited sample mode for checking a schema change before a larger run;
- completion report with changed interpretations and before/after graph coverage;
- warning that graph values can change after reanalysis.

Still deliberately outside this flow:

- a qualitative, human/AI audit of whether the *new* interpretations are correct. The existing archive audit and CSV export remain the right tools for that separate judgement.

## Priority 6: Contextual Life Graph V2

Build on the implemented v1 graph foundation and make it useful during actual bot reasoning.

Core idea:

- live context should not be one large text blob that AI reads in full;
- memory should be a selective network of concepts, relations, evidence, weights, and decay;
- AI should not read all memory; AI should query memory and receive a small relevant subgraph;
- raw diary entries remain the source of truth; the graph is an index of meanings above them.

This is broader than short-term context:

- the current manual user profile stays as explicit user-authored context;
- the life graph is dynamic long-term memory inferred from entries, corrections, summaries, and confirmations;
- graph facts are not hard truth unless confirmed and supported by evidence.

Soft graph model:

Do not force everything into rigid tables like `people`, `places`, `projects`, `emotions`.
The graph should support living concepts:

- `важлива людина`
- `особистий проєкт`
- `регулярна подія`
- `чутлива особиста тема`
- `очікування відповіді від інших людей`
- `зміна стану після значущої події`
- `важливе заняття як джерело енергії і ризику`
- `рутина як спроба стабілізації`

Nodes should have a soft kind for orientation, but the kind should not define the whole model:

```json
{
  "label": "очікування відповіді від інших людей",
  "kind": "theme",
  "aliases": ["ще не відповіли", "чекаю на уточнення", "відповідь затримується"],
  "summary": "Ситуації, де стан змінюється через невизначеність або затримку відповіді від іншої людини.",
  "confidence": 0.72,
  "status": "hypothesis"
}
```

Edges should also be soft and evidence-backed:

```json
{
  "source": "очікування відповіді від інших людей",
  "relation_label": "often_precedes",
  "target": "розчарування / спад настрою",
  "summary": "Коли відповідь від іншої людини зависає, це часто пов'язано зі спадом настрою.",
  "weight": 0.68,
  "confidence": 0.61,
  "evidence_entry_ids": ["..."],
  "last_seen_at": "...",
  "status": "hypothesis"
}
```

Node resolution:

- do not create duplicates like `проєкт`, `мій проєкт`, `творчий проєкт`, `робоча справа` unless they really differ;
- when AI proposes a new node/edge, search existing nodes by embeddings, aliases, and graph neighborhood;
- AI then chooses: merge, link, create new, ignore as trivial, or ask the user;
- uncertain merges should become explicit questions, not silent overwrites.

Example:

> Маю припущення, що це чутливе переживання пов'язане з ширшою особистою темою. Це так?

Possible choices:

- `так, це частина цього`
- `ні, це окрема тема`
- `скоріше про маму`
- `не хочу зараз розбирати`

Context lookup by uncertainty:

- AI should be able to say, structurally: "I do not understand this reference";
- the system then searches the graph/capsules for relevant context;
- only the small relevant subgraph goes back into the prompt.

Examples:

- User says `"Назва" прямо зараз` -> lookup can resolve a title as a project item, not a location.
- User mentions that a reply has not arrived -> lookup returns only the relevant waiting context.
- User mentions a recurring event -> lookup returns only related patterns, not the entire life context.

Weights, decay, and status:

- repeated supported relation -> weight grows;
- old unused relation -> weight decays;
- user correction -> confidence drops or edge changes;
- user confirmation -> status may become `confirmed`;
- contradiction -> status may become `contradicted`;
- stale relation -> remains available but low priority.

Possible statuses:

- `candidate`
- `hypothesis`
- `confirmed`
- `stale`
- `contradicted`
- `rejected`

Design constraints:

- graph memory must not become a dumping ground for every tiny detail;
- every durable node/edge needs evidence;
- sensitive assumptions should remain tentative unless confirmed;
- do not use deterministic root/keyword rules to infer meaning;
- make graph influence inspectable: show which nodes/edges affected a question or summary;
- keep raw entries exportable and separate from graph hypotheses.

Implementation stages:

1. Add embedding-assisted node resolution: merge/link/create/ignore/ask.
2. Add retrieval: given a current entry and uncertainty, return a small relevant subgraph.
3. Feed retrieved subgraphs into question generation, micro-summaries, and daily summaries.
4. Add decay and usefulness scoring.
5. Add a "what the bot remembers and why" debug/admin view.

Storage note:

- a graph database may help later, but it is not mandatory for the first serious version;
- PostgreSQL can store graph-like nodes and edges in normal tables, with pgvector for node/capsule similarity;
- start with a relational graph representation unless graph traversal needs become painful.

## Priority 7: Semantic Situation Memory

Use embeddings as a practical memory layer, not only as manual `/similar` search.

Important correction:

- raw diary text alone is usually not enough for useful memory;
- short entries like "пізніше", "знову засумував", "нормально", "йду" only make sense through the prompt, previous entries, day context, corrections, and live context;
- embeddings should operate on contextual memory capsules, not just `raw_text`.

Target behavior:

- treat embeddings as memory of recurring life situations, not just similar wording;
- cluster repeated situations such as "expectation from someone -> disappointment -> mood drop", "contact with mother", "therapy before/after", "album work", "going outside";
- when generating a question, use similar past situations to ask a more specific and useful question;
- when summarizing a day/week/month, compare current patterns with similar previous patterns without presenting retrieved memories as current-day facts;
- reduce repetitive questions by checking whether a similar uncertainty has already been asked recently;
- explain retrieved memory in human terms: why this moment seems similar and how it differs.

Implemented first practical retrieval layer:

- automatic snapshot-question generation now returns a structured, evidence-linked semantic-memory insight in the same AI call;
- an insight is stored only when its evidence ids are present in the retrieved records, so invented memory references do not become debug data;
- snapshot context keeps the cautious hypothesis and evidence count, while the user-facing question remains grounded in the current day;
- the memory menu has `Пам’ять у питаннях`, a compact debug view of when retrieval actually changed a question, without dumping raw similar diary records.
- the same evidence-bound `semantic_memory_insight` contract now applies to micro-summaries and daily/period summaries in their existing AI calls; an insight is stored only when every cited entry id belongs to the retrieved capsule set.
- prompts explicitly distinguish similarity confidence from a claim about the current moment/day/period, so memory can shape a cautious observation without overwriting current evidence.
- retrieved embedding capsules are enriched with up to three evidence-linked `situation` graph nodes for the same past entry; this connects semantic retrieval to durable graph hypotheses without an additional AI call;
- situation nodes retain their own evidence count, recency, confidence and normal graph decay, while prompts treat them as context for why past moments resemble each other rather than proof about the present.
- when an evidence-bound memory insight is stored for a snapshot question, its debug view can also show only the situation labels attached to the cited retrieved records.

Still remaining:

- durable situation clusters and their usefulness/decay scores;
- compact user-facing use of summary/life-context memory insights when it has clear practical value, not merely diagnostic value;
- AI-mediated comparison of current and past situations beyond the question-generation call.

Possible UX:

> Це схоже на кілька минулих моментів, де очікування від людини різко впливало на стан. Тут зараз більше розчарування, самотність, чи щось інше?

> Схожі вечори раніше часто закінчувалися читанням або прогулянкою. Сьогодні це радше допомагає, чи вже не дуже?

Design constraints:

- do not let semantic memory override the current entry;
- keep retrieved memories clearly separated from facts of the current day;
- avoid deterministic hardcoded rules based on word roots or keyword fragments;
- use AI to interpret recurring patterns, but require evidence from retrieved entries;
- decay or deprioritize noisy memory records over time;
- keep manual search available, but make automatic memory useful enough that manual search is not the main value.

Contextual memory capsule:

Instead of embedding only the entry text, build a compact AI-authored memory text for each meaningful entry.

Possible capsule contents:

- raw user text;
- bot question / active snapshot prompt if this entry was an answer;
- micro-summary;
- corrected interpretation if the user corrected the bot;
- structured features: entry type, activities, current emotions with intensity, mood/energy/anxiety evidence, uncertainty notes;
- local day context before/after the entry;
- relevant confirmed live context about people, places, projects, terms;
- whether the entry is current state, story, dream, activity-only, or reply fragment.

Example:

```text
Current moment: user felt sad after Max did not ask his brother about the tour idea.
Day context: before this, therapy increased hope about moving out and creative work.
People/projects: Max is connected with music/tour uncertainty; album is an important creative project.
Current emotions: sadness, loneliness, disappointment. Mood dropped.
Important pattern candidate: expectation from another person did not resolve and the state changed quickly.
```

Retrieval UX:

- do not show raw similar records as the main value;
- retrieved capsules should be passed through an AI "why this is similar" step;
- output should be a cautious pattern hypothesis, not a list of records;
- every pattern should keep evidence entry ids and confidence.

Implementation ideas:

1. Backfill/rebuild old embeddings so recent archive uses contextual capsules.
2. Add a "memory insight" step that turns retrieved capsules into compact pattern candidates.
3. Store lightweight situation clusters with evidence entry ids, confidence, last_seen_at, and usefulness score.
4. Use clusters in question generation, daily summaries, and life-context review.
5. Add a debug/admin view showing which memories influenced a question or summary.
6. Add tests that ensure semantic memory is used as context, not copied into current facts.

## Priority 8: Adaptive Observation Frequency (Implemented)

Move beyond a mostly time-randomized schedule. The bot should adapt the next snapshot interval
to how quickly the day seems to be changing.

Core idea:

- do not make frequency depend simply on "good" or "bad" state;
- adapt based on volatility, eventfulness, uncertainty, and likely near-future change;
- ask more densely when the day is actively unfolding, not when the user is merely distressed;
- ask less often when the state is stable, the day is already well covered, or the user has ignored/paused prompts.

Implemented internal AI-derived fields after each meaningful entry:

```json
{
  "state_change_likelihood": "low|medium|high",
  "eventfulness": "low|medium|high",
  "volatility": "stable|moving|volatile|sensitive",
  "next_checkin_window_minutes": {"min": 25, "max": 55},
  "reason": "коротке пояснення, засноване на поточному записі й контексті дня"
}
```

Examples:

- stable, ordinary activity, enough recent data -> ask later;
- therapy just ended, important contact is pending, or mood shifted sharply -> ask sooner;
- very sensitive/overloaded state -> do not automatically increase pressure; maybe ask later or use a very small prompt;
- positive uplift can also justify a follow-up, because positive dynamics are important too.

Safety and UX constraints:

- user messages and manual entries reset the next-question timer;
- open unanswered prompts block new prompts unless a gentle follow-up is clearly useful;
- ignored/deferred prompts should lower frequency for a while;
- never let adaptive frequency bypass quiet mode, sleep, or explicit pauses;
- keep the behavior explainable in debug/logs, but do not show the reasoning every time;
- avoid hardcoded keyword/root rules; use AI analysis plus structural anti-spam constraints.

Implemented behavior:

- `observation_cadence` витягується тим самим AI-викликом, що й аналіз запису, без додаткової вартості;
- AI повертає вікно наступної перевірки, волатильність, ймовірність змін, подієвість і службову причину;
- планувальник приймає рекомендацію лише за достатньої впевненості, затискає її в межі користувацького мінімуму/максимуму та стабільно вибирає момент для конкретного запису;
- новий ручний запис або відповідь все одно відсуває наступний зріз, бо відлік іде від останньої активності;
- жодна рекомендація не обходить активні години, сон, тиху паузу, відкритий зріз або відкрите уточнення;
- вибір і причина зберігаються в `Snapshot.context_json.scheduling` та пишуться в журнал для діагностики;
- у «Ритмі зрізів» є перемикач адаптивної частоти.

Remaining tuning:

- після кількох тижнів реальних даних перевірити, чи AI не пропонує надто вузькі або надто широкі вікна.

Possible setting:

> Адаптивна частота: бот питає частіше, коли день швидко змінюється, і рідше, коли все стабільно.

## Priority 9: Reports and Emotion Graph Polish

Current state:

- daily metrics explain graphable mood/energy coverage;
- emotion charts use strict/current emotion signals and separate broader affective states from strict emotions;
- daily emotion charts use emotion lanes instead of forcing missing emotions to zero;
- weekly/monthly period charts can include a separate emotion dynamics image;
- period PDF reports exist separately from this plan.
- metrics also send a separate `Спектр стану дня`: a broad visual trajectory based only on
  graphable mood points, with current-emotion color used as a secondary accent. It gently
  interpolates between observed points, making long or under-observed intervals visually
  quieter rather than presenting them as additional measurements; it does not replace the
  strict emotion lanes.
- the existing heavy daily-summary call can return up to four evidence-bound turning points,
  each anchored to a specific entry; no additional AI call is needed;
- the spectrum marks only turning points that also have a graphable spectrum point, using
  small numbers rather than prose on the image;
- `Повороти дня` opens a Telegram list, a detail card, and then the original supporting entry.

Remaining:

- consider stacked/ribbon variants after enough reanalyzed data exists;
- ensure PDF reports include the emotional story of each day, not only aggregate charts.

## Priority 10: Controlled Affective Vocabulary Review

Implemented:

- a strict controlled vocabulary for `emotions` and a separate controlled vocabulary for `affective_states`;
- validator-level transfer of a controlled affective label mistakenly emitted as an emotion, without relying on diary-text keyword matching;
- rejection of out-of-vocabulary emotion labels instead of letting them silently enter charts;
- exports and metric views preserve broader affective states separately from strict emotions.
- `Дані → Аудит емоцій` shows strict emotions, broader states, and legacy/out-of-vocabulary labels without another AI call.

Design note:

- this is an operational taxonomy for this diary, not a claim that science has one universally settled finite emotion list;
- body, energy, attention and medication states remain outside both lists.

Remaining:

- inspect reanalysed real data and adjust the two vocabularies only deliberately, rather than accepting free-form model labels;
- consider a separate compact affective-state visualization only if its use becomes clear from actual diary history.

## Priority 11: Period State and Emotion Analysis

Turn weekly and monthly reports into an evidence-based view of change over time, rather than a
short AI-written recap or a collection of unrelated averages.

Implemented foundation:

- `period_analysis.v1` is a deterministic, reusable aggregate over already saved entry analyses and journal-day boundaries;
- it records coverage, daily mood/energy trajectory, strict emotion frequency and mean intensity, repeated co-occurrence, time-of-day rhythm, cautious repeated activity/state observations, and daily-summary turning points;
- it builds the same aggregate for the preceding comparable period and stores raw coverage alongside metric differences, so the AI can decline a comparison when the periods are not actually comparable;
- weekly/monthly AI prompts receive the compact aggregate and are explicitly forbidden from inventing numeric trends or treating repeated observations as causation;
- period details now expose grouped `Емоції`, `Патерни`, and `Повороти` views, with rhythm included in `Метрики` to keep the menu compact;
- visual PDF reports now include a separate high-resolution emotion-dynamics page based on strict current emotion signals and intensity, while preserving the existing per-day stories and contents.

### Product questions

The report should help answer:

- what emotional and energy trajectory was visible across the period;
- which emotions were frequent, intense, co-occurring, or changing over time;
- whether there were meaningful turning points and which concrete diary moments support them;
- which activities, situations, people, places, or events were repeatedly associated with a state;
- whether the current period differs from the preceding comparable period;
- how much of the apparent pattern is supported by reliable observations versus gaps.

### Report sections

- `Огляд`: a compact narrative of the period grounded in computed observations and selected evidence;
- `Динаміка`: mood and energy trajectories by journal day, with medians/ranges and visible uncertainty;
- `Емоції`: frequency, intensity, co-occurrence, and day-to-day trajectories for the strict emotion vocabulary;
- `Ритм`: observed morning/day/evening patterns using journal-day time, not calendar-midnight assumptions;
- `Патерни`: cautious repeated associations between state and activities, events, graph context, or situations;
- `Повороти`: a small selection of important state changes, each linked to entries and evidence;
- `Дані`: coverage, confidence, gaps, and reasons why a conclusion is weak or absent.

The bot should expose these as grouped report actions rather than a single oversized message. The
same sections should be available for an explicitly selected week, month, or custom range, and
the PDF export should preserve the per-day stories and navigable structure.

### Analysis boundaries

- Compute counts, distributions, time windows, comparisons, co-occurrence candidates, and coverage deterministically from saved analyses.
- Let AI explain and prioritize already computed candidates; it must not invent numerical trends or causal conclusions.
- Phrase associations as observations, not proof of causation, and show the number of supporting moments where useful.
- Do not produce a pattern from a single event or from weak/unknown signals.
- Compare with a previous period only when the observation coverage is reasonably comparable; otherwise explain why comparison is unreliable.
- Preserve journal-day boundaries, including activity after midnight before sleep.
- Use graph and semantic memory only as retrieval aids for relevant evidence; raw similar records never become evidence by themselves.

### Visual direction

- Keep mood/energy and strict-emotion charts separate so neither becomes unreadable.
- Prefer a small number of legible charts with real gaps over dense dashboards.
- Use a separate turning-points layer or list; do not place long prose labels directly over a chart.
- Show intensity, variability, and coverage alongside frequency so "often" is not confused with "strong".

Remaining refinement:

1. Audit real weekly/monthly reports for over-interpretation, then tune thresholds and wording rather than adding diary-text keyword rules.
2. Extend custom-range report navigation to expose the same grouped period views directly, not only the existing visual PDF.
3. Add graph-backed situation/context candidates to the deterministic pattern input once their reliability is audited; raw graph nodes must not become facts by themselves.
4. Consider a dedicated period-turning-points chart only after structured turning points can carry reliable entry/time references.

## Success Criteria

The next phase worked if:

- clarification questions feel situational and useful, not like repeated scale questions;
- manual corrections decrease over time;
- emotion labels stop mixing current emotions with activities, body states, topics, and remembered/story emotions;
- emotion intensity graphs show a readable emotional shape of the day;
- AI analysis audit CSV shows fewer hallucinated metrics and less context contamination;
- semantic memory helps the bot notice recurring situations without confusing past and present;
- adaptive frequency improves timing of questions without increasing pressure or notification noise.
