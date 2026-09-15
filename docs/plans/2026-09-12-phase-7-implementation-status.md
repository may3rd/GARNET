# Phase 7 implementation status

Phase 7 completes the review-to-release boundary for single-sheet and
multi-sheet P&ID graphs.

## Implemented

- Stage 8 labels review items with `release_blocking` and
  `release_relevance`, exposes existing decision targets, and reports blocking
  and informational counts.
- Stage 9 treats absent decisions as unresolved. Explicit `accept_as_is` and
  `false_positive` decisions resolve an item without changing graph geometry;
  `defer` keeps it open.
- Stage 9 supports atomic topology decisions:
  - `merge_nodes`
  - `reconnect_edge`
  - `split_edge`
  - `delete_edge`
  - `set_node_type`
- Existing `set_line_number` and `set_flow_direction` decisions participate in
  the same release decision. Invalid or stale batch targets do not partially
  mutate the graph.
- Applied topology decisions retain deterministic before/after snapshots and
  affected entity IDs. Split routes retain ordered pixel polylines, updated
  route lengths, and non-duplicated physical attachment observations.
- Stage 9 writes `stage9_release_gate.json`. Duplicate decisions, decisions for
  unknown review items, and unresolved release-blocking items keep the gate
  blocked.
- The API withholds graph-v1, Stage 10 process exports, the Stage 11 connection
  overlay, and multi-sheet merge input until the single-sheet gate is ready.
  Review items, resolutions, correction audits, summaries, and the gate remain
  accessible while blocked.
- Multi-sheet graph release requires an explicit connector-review revision and
  zero merge issues. Released `cross_sheet_continues` relationships are marked
  reviewed and retain the review revision and reviewer provenance.

## Compatibility

The Stage 7 graph stays unchanged and Stage 9 remains the corrected revision.
Existing graph-v1 fields and process exports remain additive. Completed legacy
Stage 9 jobs without a gate remain readable; newly generated jobs use the gate.
Legacy system manifests whose merge status says connector review is pending are
blocked even when they predate the explicit `release_ready` field.

## Next phase

Phase 8 should use only released graph revisions to create process boundaries,
test-package candidates, and LLM-oriented process and HAZOP projections. Those
views must retain pixel routes, cut points, exclusions, provenance, confidence,
and unresolved evidence.
