# Memory Graph: Current Design And Remaining Work

This document describes the current memory boundary and only its remaining work. Completed graph
features are listed in [`completed-work.md`](completed-work.md).

## Current Boundary

- Raw diary records, corrections, timestamps and photos are the source of truth.
- The graph is an evidence-linked index of tentative concepts and relations, not a biography or a
  source of facts about the present moment.
- User-authored profile context stays separate from dynamic inferred memory.
- AI receives a small relevant subgraph, never the whole graph; retrieved memory is reference
  context, not evidence of a current state.
- Nodes and edges retain evidence, confidence, weight, recency and status. Unused hypotheses can
  decay into `stale` without destroying their source evidence.
- Ambiguous merges remain separate until an AI-generated, evidence-bound confirmation is answered
  with a contextual option or free text.

## Remaining Work

1. Backfill the old archive through the current capsule, embedding and graph pipeline in a bounded,
   resumable job with before/after audit.
2. Add evidence-backed contradiction candidates to the weekly review, using the same confirmation
   queue as ambiguous duplicate candidates.
3. Audit graph retrieval logs against real questions and summaries. Keep only retrieval paths that
   demonstrably add useful context without introducing stale assumptions.
4. Review decay and consolidation on real graph exports, especially whether noisy candidates become
   stale soon enough and whether meaningful aliases are preserved.
5. Consider a graph database only if relational traversal and retrieval become an observed
   bottleneck; it is not a prerequisite for these steps.

## Cost Rule

Local graph updates should reuse the entry analysis/capsule call. Separate AI work is reserved for
bounded duplicate or contradiction review; embeddings are reused as candidate signals rather than
recomputed solely for graph maintenance.
