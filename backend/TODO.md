# Backend TODO

## Current gaps to master plan

### Highest priority
- Add a dedicated equipment detection pass as planned `Stage 4.1`.
- Extend text-to-graph attachment beyond the current validated scope:
  - instrument tags to nodes/symbols where appropriate, not only nearest-edge semantics
  - notes to review queue or semantic targets
  - utility/process labels to graph semantics when confidence is sufficient
- Improve crossing semantics:
  - true junctions
  - non-connecting crossings
  - unresolved crossing queue
- Add flow direction / arrow-to-edge assignment.

### Graph / export gaps
- Align provisional graph payload to [`backend/schema/graph_v1.json`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/schema/graph_v1.json).
- Add provenance bundles and confidence bundles.
- Add export-ready polyline simplification.
- Add direction state and richer review state.

### QA / review gaps
- Prioritize review queue items by severity and semantics.
- Add orphan-terminal checks and path-based checks.
- Add targeted retry hooks for OCR, attachment, crossing, and morphology issues.
- Expose review queue and graph details more deeply in frontend/API.

## Current validated state
- Stage 4 line-number fusion is validated on the PPCL set:
  - fuller crop OCR is preferred over partial sheet OCR
  - bogus OD-only page-border/title artifacts are rejected
  - `0` OD-only line-number survivors across the 9-image PPCL sweep
- Stage 4 instrument semantic fusion is validated on the PPCL set:
  - balloon/tight crop OCR is working
  - `0` detection-only instrumentation survivors across the 9-image PPCL sweep
- Stage 12 semantic attachment is validated on the PPCL set:
  - adaptive line-number attachment is working
  - small instrumentation-specific tolerance is working
  - all confirmed line numbers and confirmed instrumentation semantics attached in the 9-image PPCL sweep
- Gemini OCR route now explicitly falls back to the repo root `.env` for `OPENROUTER_API_KEY`.

## Current highest-value next work
1. Add dedicated equipment detection as `Stage 4.1`.
2. Improve crossing-vs-junction semantics in the geometry path.
3. Add richer human-in-the-loop review for Stage 4 and Stage 12:
   - per-item provenance
   - accept/reject actions
   - unresolved item queues
4. Align graph/export payloads to `graph_v1.json`.

## Equipment attachment note
- Equipment attachment is integrated into the staged pipeline as provisional semantics.
- Dedicated equipment detection is still on hold.
- Planned placement for dedicated equipment detection: `Stage 4.1`.
- Until `Stage 4.1` exists, equipment attachment should use the current Stage 4 detections only as provisional evidence.
