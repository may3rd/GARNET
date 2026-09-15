# Phase 4 implementation checkpoint

Status: implementation and review complete; changes remain uncommitted. Luna
performed the implementation and regression work. Terra reviewed the
implementation in multiple passes, its findings were addressed, and its final
read-only re-check reported no actionable findings.

## Saved work

- Stage 7 graph construction now creates drawing-scoped equipment and physical
  port identities. Edge endpoints reference those ports while legacy node
  references remain available for graph-v1 consumers.
- Canonical line identities are separate from OCR occurrences. Repeated
  reviewed text can support one drawing-scoped line without losing occurrence
  evidence.
- Inline equipment keeps drawing-scoped identity and ordered occurrences along
  each pixel route.
- Instruments are canonicalized separately from OCR occurrences. Functional
  `measures`, `controls`, and `actuates` relationships require explicit
  evidence; proximity-only associations remain unresolved.
- Stage 10 adds canonical line, equipment-connectivity, inline-MTO, and
  instrument projections while retaining its legacy fields.
- Graph-v1 catalog repair prevents dangling equipment/port references,
  preserves conflicting endpoint evidence for review, deduplicates semantic
  relationships, and emits strict JSON-safe data.

## Review closure

- Port identity is deterministic across trace ordering and small terminal
  jitter, with explicit port indices preferred when present.
- Skipped or malformed traces cannot seed canonical ports. Non-finite source,
  terminal, segment, or interior route coordinates exclude the whole trace and
  create blocking review evidence.
- Equipment, port, edge endpoint, and relationship references are
  self-consistent, including repaired legacy or malformed persisted catalogs.
- Repeated catalog rows and attachment observations retain occurrence evidence
  without duplicating graph relationship IDs.
- Graph-v1 sanitizes non-finite values across geometry, catalogs, and copied
  evidence, and pipeline JSON persistence rejects non-standard NaN/Infinity
  output.
- Terra's final closure review reported no actionable findings and marked
  Phase 4 ready.

## Verification at checkpoint

- Backend root discovery: 287 tests successful, one skipped.
- Focused Phase 4 and graph export tests: 45 successful.
- Backend Python compilation and `git diff --check`: successful.

No real-drawing extraction benchmark was run because this slice did not execute
the heavyweight OCR/detection pipeline with project weights. Phase 5 should add
flow-direction inference from arrow evidence and geometry while retaining
unknown and conflicting cases.
