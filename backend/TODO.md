# Backend TODO

## Current gaps to master plan

### Highest priority
- Integrate equipment attachment into the staged `pid_extractor` flow.
- Add a dedicated equipment detection pass as planned `Stage 4.1`.
- Add text-to-graph attachment:
  - line numbers to edges
  - tags to nodes/symbols
  - notes to review queue or semantic targets
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

## Equipment attachment note
- Equipment attachment implementation is proceeding now.
- Dedicated equipment detection is still on hold.
- Planned placement for dedicated equipment detection: `Stage 4.1`.
- Until `Stage 4.1` exists, equipment attachment should use the current Stage 4 detections only as provisional evidence.
