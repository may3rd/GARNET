# Phase 1–3 implementation checkpoint

Status: implementation and review complete; changes remain uncommitted. Luna
performed the implementation and regression work. Terra completed the final
read-only review with no actionable findings.

## Saved work

- Phase 1: restored `MASTER_PLAN.md`, documented the target contract, added
  synthetic fixtures, and reconciled historical stage-numbering guidance.
- Phase 2: route comparison now considers route geometry and length; split
  attachments are distributed by location; the production association path no
  longer calls simulated HITL line assignment. Added regression coverage.
- Phase 3: review corrections now rebuild the public graph-v1 export and final
  overlay from the corrected graph revision. Line identities, evidence,
  assignment state, line groups, process exports, resume fingerprints, stale
  artifact invalidation, and parent-system invalidation stay synchronized.

## Review closure

- Route deduplication preserves close parallel and bypass paths while collapsing
  reversed and resampled representations of the same route.
- Split-point line evidence is localized or recorded as ambiguous junction
  evidence instead of being assigned arbitrarily to one branch.
- Unsupported line-number assignments remain unresolved in the production path.
- Rejected review evidence cannot be promoted to accepted during duplicate
  merging, and structured provenance remains JSON serializable.
- Review edits invalidate and regenerate all corrected downstream artifacts.
  Stale graphs are withheld from job responses, direct artifact reads, and
  single- or multi-sheet merge operations until Stage 9 completes.
- Resume rejects immutable run-signature changes without mutating the manifest,
  and handles changed review inputs, legacy fingerprints, and interrupted Stage
  9 execution conservatively.
- Terra's closure review found no actionable issues in the final runtime and
  regression-test diff.

## Verification at checkpoint

- Backend root discovery: 256 tests, successful, one skipped.
- Separate `tests/test_path_tracer` discovery: 21 tests, successful.
- Backend Python compilation and `git diff --check`: successful.
- Terra performed a final static, read-only review after the expanded regression
  suite and reported no actionable findings.

No real-drawing extraction benchmark was run because this slice did not execute
the heavyweight OCR/detection pipeline with project weights. Phases 4–9 remain
outside this work slice; flow direction, cross-sheet semantic strengthening,
test-package generation, and representative benchmark validation are still
roadmap work.
