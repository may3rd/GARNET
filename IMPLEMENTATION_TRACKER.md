# P&ID Digitizing Implementation Tracker

## Goal
- Turn the current `backend/garnet` pipeline into the geometry-first, confidence-scored P&ID graph system described in [`MASTER_PLAN.md`](/Users/maetee/Code/GARNET/MASTER_PLAN.md).
- Keep delivery practical: each sprint should end in a runnable artifact, not just refactoring.
- Keep delivery traceable: every work item maps to master-plan references, repo paths, and verification evidence.

## Scope
- In scope: backend pipeline stages, graph model, export contract, API integration needed to expose pipeline outputs, and review-ready anomaly outputs.
- Out of scope for this plan: model retraining programs, production deployment hardening, frontend redesign, and full multi-sheet merge UI.

## Tracking model
- Status values: `TODO`, `DOING`, `BLOCKED`, `DONE`.
- Update each task with:
  - status
  - owner
  - commit or branch reference
  - evidence path under `backend/output/` or `backend/tests/`
  - verification command and result
- Use task IDs in commit messages or PR titles where possible, for example `S2-03 geometry: add skeleton node clustering`.

## Current baseline (repo reality on 2026-03-09)

| Area | Repo evidence | Status | Gap to master plan |
|------|---------------|--------|--------------------|
| Staged orchestrator | [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py) currently implements `stage1_input_normalization`, `stage2_ocr_discovery`, `stage4_object_detection`, `stage4_line_number_fusion`, `stage5_pipe_mask`, `stage6_morphological_sealing`, `stage7_skeleton_generation`, `stage8_skeleton_node_detection`, `stage9_node_clustering`, `stage10_edge_tracing`, `stage11_junction_review`, `stage12_graph_assembly`, and `stage13_graph_qa` | Active | Stage numbering is intentionally sparse and the rebuild now reaches provisional graph + QA with first-pass semantic attachment, but export refinement, dedicated equipment detection, and directionality are still incomplete |
| OCR ingestion | [`backend/garnet/text_ocr.py`](/Users/maetee/Code/GARNET/backend/garnet/text_ocr.py), [`backend/garnet/easyocr_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/easyocr_sahi.py), [`backend/garnet/gemini_ocr_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/gemini_ocr_sahi.py), [`backend/garnet/paddle_ocr_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/paddle_ocr_sahi.py), [`backend/garnet/ocrmac_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/ocrmac_sahi.py) | Active | Selectable OCR routing is implemented, but comparative route evaluation and route-specific tuning are still incomplete |
| Object detection | `/api/detect` flow in [`backend/api.py`](/Users/maetee/Code/GARNET/backend/api.py), [`backend/garnet/object_detection_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/object_detection_sahi.py), and Stage 4 in [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py) | Active | Detection is part of the staged rebuild and now feeds dedicated line-number fusion, but dedicated equipment detection remains planned for `Stage 4.1` |
| Pipe geometry | [`backend/garnet/pipe_mask.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_mask.py), [`backend/garnet/pipe_seal.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_seal.py), [`backend/garnet/pipe_skeleton.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_skeleton.py), [`backend/garnet/pipe_nodes.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_nodes.py), [`backend/garnet/pipe_node_clusters.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_node_clusters.py), [`backend/garnet/pipe_edges.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_edges.py), [`backend/garnet/pipe_junctions.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_junctions.py) | Active | Geometry path now exists end-to-end, but crossing semantics and cleanup are still heuristic and need later refinement |
| Graph assembly | [`backend/garnet/pipe_graph.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_graph.py), [`backend/garnet/pipe_graph_qa.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_equipment_attachment.py), [`backend/garnet/pipe_text_attachment.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_text_attachment.py) | Active | Graph JSON and QA artifacts now include provisional equipment, line-number, and instrumentation attachment with validated Stage 12 thresholding on sampled PPCL drawings, but export refinement, directionality, and richer per-item provenance exposure still need work |
| Export | [`backend/schema/graph_v1.json`](/Users/maetee/Code/GARNET/backend/schema/graph_v1.json) remains a reference artifact, while `stage12_graph.json` is the current provisional graph payload | Early | Confidence bundles, provenance, direction state, and simplified polylines are not yet aligned to the target export schema |
| Verification | `py_compile`, stage helper tests, runner tests, and API contract tests now cover the staged rebuild through Stage 13 | Moderate | API tests still depend on optional local dependencies and there is still no scorecard or artifact-diff acceptance harness |
| Code health | `pid_extractor.py` no longer carries the old duplicated line-graph stages, but it still has sparse stage numbering (`1,2,4,5`) and evolving defaults that must stay aligned with the API | Moderate | Keep the stage manifest, tracker, and shared model defaults aligned as new stages land |

## Suggested cadence
- Use 1-week sprints.
- Group sprints into master-plan phases so progress stays visible at both levels:
  - Phase 0: baseline and guardrails
  - Phase A: evidence extraction
  - Phase B: geometry engine
  - Phase C: attachment and semantic association
  - Phase D/E: graph QA, export, recovery, and review boundary

## Sprint roadmap

| Sprint | Master-plan refs | Outcome | Exit evidence |
|--------|------------------|---------|---------------|
| Sprint 0 | Foundation before A-E | Stable baseline, repeatable sample run, task-level verification harness | Baseline run folder + metrics manifest |
| Sprint 1 | A1-A6 | Evidence extraction split into clear OCR and detection sub-stages | Structured OCR/detection tables with confidence and provenance |
| Sprint 2 | B1-B7 | Explicit geometry engine with audited mask -> skeleton -> nodes -> traced edges flow | Geometry artifact set and edge extraction metrics |
| Sprint 3 | C1-C5 | Multi-signal attachment, text binding, arrow-driven directionality | Association report + directed edge coverage metrics |
| Sprint 4 | D1-D4, E1 | Graph schema, graph-native QA, simplified export | Graph QA report + smaller export payloads |
| Sprint 5 | E2-E3, V3 prep | Recovery loop, review queue, API-ready unresolved outputs | Retry queue + manual-review queue exposed in backend outputs/API |

## Sprint 0 - Baseline and guardrails

**Objective**
- Make the current pipeline repeatable enough to measure progress.

**Definition of done**
- A single sample command runs end-to-end and produces a versioned output folder.
- Stage outputs are documented and diffable.
- Immediate code-structure risks are reduced before new behavior lands.

| ID | Task | Master-plan refs | Repo targets | Verification | Status |
|----|------|------------------|--------------|--------------|--------|
| S0-01 | Freeze a baseline sample set and canonical run command using existing local assets | Foundation | `backend/sample.png`, `backend/coco_annotations.json`, `backend/ocr_results.json`, `backend/coco_arrows.json` | `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --coco coco_annotations.json --ocr ocr_results.json --arrow-coco coco_arrows.json --out output/baseline_s0` | DONE |
| S0-02 | Add a stage manifest file that records stage names, output filenames, and timing/metrics JSON contracts | Foundation | `backend/garnet/pid_extractor.py`, new tracker helper file if needed | Baseline run writes `output/baseline_s0/stage_manifest.json` | DONE |
| S0-03 | Remove duplicated `stage6_line_graph` logic and align `--stop-after` semantics with real stage count | Foundation | `backend/garnet/pid_extractor.py` | `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py` plus targeted `unittest` coverage and baseline run | DONE |
| S0-04 | Add a thin regression harness for staged sample runs and artifact existence checks | Foundation | new tests under `backend/tests/` or lightweight runner under `backend/tools/` | Run harness against the baseline sample and confirm expected artifacts exist | TODO |
| S0-05 | Define a scorecard for every sprint: graph node count, edge count, open ends, isolated equipment count, directed-edge count, export size | Foundation | new markdown or JSON scorecard file at repo root or `backend/` | Scorecard produced from the baseline run | TODO |

## Sprint 1 - Evidence extraction

**Objective**
- Make OCR and detection outputs explicit, reviewable, and confidence-scored before topology work.

**Definition of done**
- OCR discovery and OCR refinement are separate artifacts.
- Stage 2 uses EasyOCR on overlapped sheet tiles as the primary OCR detector.
- OCR must be user-selectable per run, with one route only:
  - EasyOCR route
  - Gemini route
- Small objects, arrows, equipment, and off-page connectors are represented as structured evidence tables.

| ID | Task | Master-plan refs | Repo targets | Verification | Status |
|----|------|------------------|--------------|--------------|--------|
| S1-01 | Split input normalization into explicit working views with preprocessing metadata | A1 | `backend/garnet/pid_extractor.py` | Baseline run writes normalized image bundle + metadata JSON | TODO |
| S1-02 | Add Stage 2 OCR discovery using SAHI-style tiling with EasyOCR as the primary detector | A2 | `backend/garnet/easyocr_sahi.py`, `backend/garnet/text_ocr.py`, `backend/garnet/pid_extractor.py` | OCR stage writes canonical sheet-level `text_regions` JSON, overlay, summary, and exception candidates | DONE |
| S1-03 | Add selectable OCR routes so a pipeline run uses one OCR route per job, with Gemini using `1024x1024` patch full-page prompts first and crop prompts only as route-local fallback for missed or `<0.3` confidence candidates | A2-A3 | `backend/garnet/gemini_ocr_sahi.py`, `backend/garnet/pid_extractor.py`, `backend/api.py`, `frontend/src/components/DetectionSetup.tsx`, `backend/gemini_detector/gemini_sahi.py`, `backend/garnet/OCR_prompts/` | Pipeline job accepts `ocr_route`, Stage 2 writes the shared OCR bundle for each supported route, and Gemini route also writes raw patch/crop audit artifacts | DONE |
| S1-04 | Normalize detector outputs into separate evidence tables for small objects, arrows, equipment, and off-page connectors | A4-A6 | `backend/garnet/pid_extractor.py`, helper module to be created if needed | Detection run writes structured evidence JSON/CSV per category | TODO |
| S1-05 | Add provenance fields so each evidence item records source stage, model/input, and confidence | A2-A6 | `backend/garnet/pid_extractor.py`, `backend/schema/graph_v1.json` if needed | Evidence tables include provenance bundle | TODO |

## Sprint 2 - Geometry engine

**Objective**
- Make geometry extraction the real backbone of the pipeline, with clean transitions from mask to graph candidates.

**Definition of done**
- The geometry path is explicit and measurable: pipe mask, morphological seal, skeleton, node candidates, clustered nodes, traced edges.
- Crossings and junctions are no longer conflated by default.

| ID | Task | Master-plan refs | Repo targets | Verification | Status |
|----|------|------------------|--------------|--------------|--------|
| S2-01 | Extract B1 pipe mask generation into a named stage with its own output artifact and metrics | B1 | `backend/garnet/pid_extractor.py`, `backend/garnet/pipe_mask.py` | Run to `--stop-after 5` and inspect `stage5_pipe_mask_summary.json` and related artifacts | DONE |
| S2-02 | Add explicit B1.5 morphological sealing before skeletonization, with conservative defaults and audit counters | B1.5 | `backend/garnet/pid_extractor.py`, `backend/garnet/pipe_seal.py` | Audit JSON reports holes sealed, blobs removed, and changed-pixel count | DONE |
| S2-03 | Separate skeleton generation from skeleton-node detection and persist raw degree maps | B2-B3 | `backend/garnet/pid_extractor.py`, `backend/garnet/pipe_skeleton.py`, `backend/garnet/pipe_nodes.py` | Raw skeleton, endpoint map, and junction-candidate map are written separately | DONE |
| S2-04 | Add node clustering for graph-node candidates instead of relying on raw skeleton pixels | B6 | `backend/garnet/pid_extractor.py`, `backend/garnet/pipe_node_clusters.py` | Clustered node file includes centroid, member count, and type guess | DONE |
| S2-05 | Implement explicit crossing-vs-junction disambiguation with unresolved-candidate output | B4 | `backend/garnet/pid_extractor.py`, `backend/garnet/pipe_junctions.py` | Run produces confirmed junctions, non-junction crossings, and unresolved queue | DONE |
| S2-06 | Add topology-aware skeleton cleanup and traced edge polylines before graph assembly | B5-B7 | `backend/garnet/pid_extractor.py`, `backend/garnet/pipe_edges.py` | Edge extraction report shows traced polyline count and cleanup removals | DONE |

## Sprint 3 - Attachment and semantic association

**Objective**
- Attach meaning to geometry without collapsing back into nearest-neighbor shortcuts.

**Definition of done**
- Equipment, inline objects, text, and arrows attach through multi-signal logic.
- Directionality is explicit where evidence exists and absent where it does not.

| ID | Task | Master-plan refs | Repo targets | Verification | Status |
|----|------|------------------|--------------|--------------|--------|
| S3-01 | Replace simple equipment snapping with multi-signal attachment scoring | C1 | `backend/garnet/pid_extractor.py`, `backend/garnet/pipe_equipment_attachment.py` | Attachment report includes scores, chosen edge, and ambiguous candidates | DONE (provisional via Stage 4 detections; dedicated equipment detection still planned as `Stage 4.1`) |
| S3-02 | Formalize inline-object association rules and edge-splitting thresholds | C2 | `backend/garnet/edge_split.py`, `backend/garnet/pid_extractor.py`; evidence: `stage10d_split_edges.json`, `stage10d_split_nodes.json`, `stage10d_split_report.json`, `stage10d_split_summary.json`, `backend/tests/test_edge_split.py` | Inline association report shows confident splits vs unresolved items | DONE |
| S3-03 | Convert flow-arrow handling from visual overlay to edge-direction assignment and local propagation | C3 | `backend/garnet/edge_direction.py`, `backend/garnet/pid_extractor.py`; branch/commit: TBD; evidence: `stage10c_edge_direction.json`, `stage10c_arrow_assignments.json`, `stage10c_edge_direction_summary.json`, `backend/tests/test_edge_direction.py` | Directed-edge metrics show assigned, propagated, and unresolved arrow-edge matches | DONE |
| S3-04 | Associate text to equipment, edges, inline objects, and off-page connectors using more than distance alone | C4 | `backend/garnet/text_ocr.py`, `backend/garnet/pid_extractor.py`, `backend/garnet/line_number_fusion.py`, `backend/garnet/pipe_text_attachment.py` | Text-link report shows target type, confidence, and unresolved text queue | DONE: line-number→edge + instrument-tag→edge (pre-existing); equipment-tag→node (this sprint) |
| S3-05 | Represent off-page connectors as explicit graph nodes with labels and page-reference fields | C5 | `backend/garnet/pid_extractor.py`, `backend/schema/graph_v1.json`, export helper to be created if needed | Export contains connector nodes with attachment metadata | DONE |

## Sprint 4 - Graph, QA, and export

**Objective**
- Make the graph the authoritative validation surface and keep exports practical.

**Definition of done**
- Graph schema supports provenance, confidence, review state, and direction.
- QA catches obvious topology anomalies.
- Export geometry is simplified enough for downstream use.

| ID | Task | Master-plan refs | Repo targets | Verification | Status |
|----|------|------------------|--------------|--------------|--------|
| S4-01 | Unify node and edge schema across in-memory graph, JSON export, and export adapter | D1-D2 | `backend/garnet/pid_extractor.py`, export helper to be created if needed, `backend/schema/graph_v1.json` | Export validation confirms required fields exist for nodes and edges | DONE (stage12b_graph_v1.json + graph_export_adapter.py)
| S4-02 | Add graph-native QA primitives for connected components, degree anomalies, articulation points, and orphan terminals | D3, E1 | `backend/garnet/pid_extractor.py`, `backend/garnet/pipe_graph_qa.py` | QA report JSON lists anomaly counts and affected node/edge ids | DONE |
| S4-03 | Generate an anomaly report and retry queue instead of only overlays | E1-E2 | `backend/garnet/pid_extractor.py`, `backend/garnet/pipe_graph_qa.py` | Output includes machine-readable anomaly and retry files | DONE |
| S4-04 | Add polyline simplification before export with configurable tolerance and compression metrics | D4 | `backend/garnet/polyline_simplify.py`, `backend/garnet/pid_extractor.py`; branch/commit: TBD; evidence: `stage10b_pipe_edges_simplified.json`, `stage10b_polyline_simplification_summary.json`, `backend/tests/test_polyline_simplify.py` | Compare export payload size before/after simplification | DONE |
| S4-05 | Expose graph QA and export outputs through the backend service where needed for later review tooling | D1-E1 | `backend/api.py`, `backend/schema/graph_v1.json` | API returns graph/QA artifacts for a sample run | DONE |

## Sprint 5 - Recovery loop and review boundary

**Objective**
- Limit human review to unresolved ambiguity and make recovery targeted rather than global.

**Definition of done**
- Recovery tasks are explicit and rerunnable.
- Manual review queue is small, categorized, and backed by artifact files or API payloads.

| ID | Task | Master-plan refs | Repo targets | Verification | Status |
|----|------|------------------|--------------|--------------|--------|
| S5-01 | Add targeted reprocessing hooks for OCR, attachment, crossing, arrow, and morphology retries | E2 | `backend/garnet/pid_extractor.py`, helper modules as needed | Retry queue items can trigger limited re-runs without replaying the whole sheet | ✅ DONE — `backend/garnet/recovery_loop.py` implements bounded non-destructive recovery (Approach 2). Stage 14 wired into pipeline. `stage5_recovery_decisions.json` written per run. Exposed via `GET /api/pipeline/jobs/{id}` as `recovery_decisions`. |
| S5-02 | Define unresolved queues for ambiguous crossings, text conflicts, uncertain direction, and connector mismatches | E3 | `backend/garnet/pid_extractor.py`, `backend/schema/graph_v1.json` | Manual-review queue JSON contains category, geometry, evidence refs, and priority | ✅ DONE — `pipe_graph_qa.py` enriched all 4 review_queue categories with `geometry` and `evidence_refs`. See commit `3c2919d`. |
| S5-03 | Add backend endpoints or export artifacts for review-ready unresolved cases | E3, V3 prep | `backend/api.py`, `backend/garnet/pid_extractor.py` | Sample run exposes unresolved queue through API or stable output artifact | ✅ DONE — `GET /api/pipeline/jobs/{id}` returns `graph_v1`, `review_queue` (stage13), and `recovery_decisions` (stage14) alongside job status/manifest. No new endpoints needed; existing payload covers the requirement. |
| S6-01 | Add `off_page_connector` field to `graph_v1.json` edge schema (reference_type, reference_value, direction, exit_terminal) | V3 prep | `backend/schema/graph_v1.json` | Field present and documented | ✅ DONE — field added to edge schema in `graph_v1.json`. |
| S6-02 | Implement `_build_off_page_connector_map()` in `graph_export_adapter.py` — join page connection attachments to stage12c labels via object_id/det_id | V3 prep | `backend/garnet/graph_export_adapter.py` | Function handles anchors top/bottom→destination, left/right→source; skips edges without labels | ✅ DONE — `_build_off_page_connector_map()` added, joins via graph topology (attach_edge.source = 'connection::{det_id}'), not via attachment.edge_id. Passes stage12_graph + connection_attachments_payload + page_connector_labels_payload. |
| S6-03 | Wire `off_page_connector` into `build_graph_v1_payload()` — pass stage12_graph + connection_attachments_payload + page_connector_labels_payload, set edge['off_page_connector'] on matched attach_edge | V3 prep | `backend/garnet/graph_export_adapter.py` | Off_page_connector appears on the attach_edge whose source is the PC node | ✅ DONE — `off_page_by_edge` map built and applied in edge loop. New test `test_off_page_connector_set_on_attach_edge` covers correct join (graph topology, not attachment.edge_id). |
| S7-01 | Implement merge engine — resolve off-page connector pairs across sheets by (reference_type, reference_value) merge key; output cross-sheet virtual edges and merge_issues | V3 | `backend/garnet/pipe_sheet_merge.py`, `backend/tests/test_pipe_sheet_merge.py` | Pairing logic: >2 sheets=AMBIGUOUS, 1 sheet=DANGLING, same sheet=INTRA_SHEET_DUP, direction conflict=DIRECTION_CONFLICT | ✅ DONE — `resolve_merge_pairs()` in `pipe_sheet_merge.py`. 12 unit tests covering all pairing cases and conflict types. |
| S7-02 | Wire merge engine into API + CLI tool | V3 | `backend/api.py`, `backend/tools/merge_sheets.py` | POST /api/pipeline/merge accepts job_ids, returns graph_v2; CLI tool for local use | ✅ DONE — `POST /api/pipeline/merge` endpoint. Standalone `tools/merge_sheets.py` CLI. |





## Sprint 6 / Phase 2 — Continuity-Aware Pipeline (2026-05-07)

**Objective**
- Add Stage 10→12 gap feedback loop: detect near-edge gaps in traced topology, feed them into graph assembly validation, resolve via the recovery engine.
- Render the connection + pipe-segment overlay (S16) so reviewers can visually verify page connection anchoring.

**Definition of done**
- `validated_edges` > 0 in `stage10_continuity_result.json` (not 0)
- `total_anomalies` present in `stage13_graph_qa_summary.json`
- `stage16_connection_pipeline_overlay.png` produced per run
- Smoke test passes on all 4 test images

| ID | Task | Master-plan refs | Repo targets | Verification | Status |
|----|------|------------------|--------------|--------------|--------|
| S8-01 | Stage 10 post-trace continuity check: `run_post_trace_continuity_check()` classifies each edge as validated/provisional/orphan/gap_candidate; runs after `_trace_edges()` completes | B5 | `backend/garnet/pipe_edges.py`, `backend/garnet/pipe_continuity_helpers.py` | `stage10_continuity_result.json` has `validated_edges>0`, `provisional_edges>0` | ✅ DONE — commit `84d7db3` |
| S8-02 | Fix terminal_role inference: edges have `source`/`target` node IDs but no `terminal_role` field; infer `junction_terminal` from `junction_*` prefix so junction→junction edges are validated | B5 | `backend/garnet/pipe_continuity_helpers.py` | junction→junction edge: status=validated; endpoint→endpoint: status=provisional | ✅ DONE — commit `5dd8cb3` |
| S8-03 | `merge_continuity_into_graph()`: enrich Stage 12 edges with continuity metadata before graph assembly | C1 | `backend/garnet/continuity_aware_connections.py` | Enriched edges passed to `build_pipe_edge_connectivity()` | ✅ DONE |
| S8-04 | `validate_connections_against_gaps()`: compare Stage 12 connections vs Stage 10 gap summary; produce `stage12_connection_validation.json` with `missed_gaps` list | C1 | `backend/garnet/continuity_aware_connections.py` | `stage12_connection_validation_summary.json` shows `missed_by_stage12` count | ✅ DONE |
| S8-05 | Stage 14 continuity checker: 10-rule violation detector with overlay | D3 | `backend/garnet/pipe_continuity_checker.py`, `backend/garnet/run_continuity_checker_stage.py` | `stage14_continuity_result.json` + `stage14_violations.json` + `stage14_continuity_violations_overlay.png` | ✅ DONE |
| S8-06 | `near_edge_gap` recovery handler: score gap by confidence; ≤5px auto-close, 5–15px human review, >20px skip | E2 | `backend/garnet/recovery_loop.py` | `stage5_recovery_decisions.json` has near_edge_gap items with accept/review actions | ✅ DONE |
| S8-07 | Stage 14→15 feedback loop: feed Stage 12 connection validation → recovery engine → gap closure decisions | E2 | `backend/garnet/pid_extractor.py`, `backend/garnet/recovery_loop.py` | RecoveryEngine processes two sources: stage13_review_queue + stage12_connection_validation.json | ✅ DONE |
| S8-08 | Stage 16 connection + pipe-segment overlay: render accepted page connections, connected pipe segments, inline element connectors on original P&ID background | V3 prep | `backend/garnet/render_connection_pipeline_overlay.py`, `backend/garnet/pid_extractor.py` | `stage16_connection_pipeline_overlay.png` saved per run (~3MB) | ✅ DONE |
| S8-09 | Fix `total_anomalies` missing from `stage13_graph_qa_summary.json` | QA | `backend/garnet/pipe_graph_qa.py` | Field present: total_anomalies=articulation+isolated+crossings+terminals | ✅ DONE — commit `5dd8cb3` |

### Smoke test results — Test-00008.jpg (2026-05-07)

```
Pipeline: 16/16 stages ✅ ~50s
validated_edges: 652 / 783 (83%)
total_anomalies: 274
review_queue: 42 (31 unresolved crossings, 5 terminal edges, 4 articulations, 2 isolated)
near_edge_gaps: 665 (gap_threshold=20px — tuning needed)
Stage 14 violations: 1,233 (errors:254, warnings:979)
Recovery: ACCEPT=491, HUMAN_REVIEW=197
Page connections (accepted): 7
```

### Open: gap threshold tuning (S8-TBD)

`GAP_THRESHOLD_PX=20.0` is too loose for dense P&ID drawings — generates 665 false-positive near-edge candidates (mostly parallel pipes running close together). Threshold tuning needed.


## Cross-sprint rules
- Do not add new topology behavior without adding or updating stage artifacts and metrics.
- Do not close a sprint without updating the scorecard from Sprint 0.
- Prefer small PRs grouped by task ID, not one large sprint branch.
- Every sprint should keep the baseline sample command runnable.

## Recommended verification commands
- Compile check:
  - `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py`
- Baseline staged run:
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --coco coco_annotations.json --ocr ocr_results.json --arrow-coco coco_arrows.json --out output/baseline_s0`
- API smoke check when API-related tasks land:
  - `cd backend && uvicorn api:app --reload --port 8001`

## Immediate next actions
1. Start with Sprint 0 and treat it as mandatory stabilization, not optional cleanup.
2. Do `S0-01`, `S0-02`, and `S0-03` before any new master-plan feature work.
3. Once baseline metrics are stable, execute Sprint 1 and Sprint 2 in order because Phase C-D work depends on those artifacts.

## Rebuild slices (hard reset track)

**Note**
- As of 2026-03-07, the active implementation path is a hard reset of `backend/garnet/pid_extractor.py`.
- New work proceeds in small vertical slices from raw image input only.
- The current detailed plan for the first slice lives in [`docs/plans/2026-03-07-slice-1-stage1-pipeline-and-frontend.md`](/Users/maetee/Code/GARNET/docs/plans/2026-03-07-slice-1-stage1-pipeline-and-frontend.md).

| Slice | Outcome | Evidence | Status |
|------|---------|----------|--------|
| Slice 1 | Stage 1-only pipeline from raw image input, backend job API, frontend Pipeline mode with stage progress and artifact review | `backend/output/slice1_stage1`, `backend/output/pipeline_jobs`, `backend/tests/test_pid_extractor_cli.py`, `backend/tests/test_pipeline_api.py` | DONE |
| Slice 2 | SAHI-style tiled EasyOCR discovery from image only, visible as a second reviewable stage in API and frontend | `stage2_ocr_regions.json`, `stage2_ocr_overlay.png`, `stage2_ocr_summary.json`, `stage2_ocr_exception_candidates.json` | DONE |
| Slice 3 | Selectable OCR routes: user chooses one OCR route per run, currently `easyocr`, `gemini`, `paddleocr`, or `ocrmac`, with Gemini using patched full-page prompts and route-local crop fallback and OCRMac using a tiled macOS Vision route | `stage2_ocr_regions.json`, `stage2_ocr_overlay.png`, `stage2_ocr_summary.json`, `stage2_ocr_exception_candidates.json`, optional Gemini raw patch/crop artifacts | DONE |
| Slice 4 | Fixed-baseline Stage 4 object detection using Ultralytics + SAHI with `backend/yolo_weights/yolo26n_PPCL_640_20260227.pt`, plus dedicated line-number fusion and instrument semantic fusion from Stage 4 detections and Stage 2/crop OCR with provenance buckets | `stage4_objects.json`, `stage4_objects_overlay.png`, `stage4_objects_summary.json`, `stage4_line_numbers.json`, `stage4_line_number_summary.json`, `stage4_line_number_overlay.png`, `stage4_instrument_tags.json`, `stage4_instrument_tag_summary.json`, `stage4_instrument_tag_overlay.png` | DONE |
| Slice 5 | Conservative Stage 5 pipe-mask generation from Stage 1 binaries with Stage 2 OCR suppression and Stage 4 object suppression | `stage5_pipe_mask.png`, `stage5_pipe_mask_overlay.png`, `stage5_pipe_mask_summary.json` | DONE |
| Slice 6 | Conservative Stage 6 morphological sealing over the Stage 5 mask with audit counters and reviewable overlays | `stage6_pipe_mask_sealed.png`, `stage6_pipe_mask_sealed_overlay.png`, `stage6_pipe_mask_sealed_summary.json` | DONE |
| Slice 7 | Stage 7 skeleton generation from the sealed mask with reviewable skeleton artifacts | `stage7_pipe_skeleton.png`, `stage7_pipe_skeleton_overlay.png`, `stage7_pipe_skeleton_summary.json` | DONE |
| Slice 8 | Stage 8 raw skeleton node detection with separate endpoint and junction candidate maps | `stage8_endpoints.png`, `stage8_junctions.png`, `stage8_nodes_overlay.png`, `stage8_node_summary.json` | DONE |
| Slice 9 | Stage 9 node clustering for endpoint and junction candidates | `stage9_endpoint_clusters.png`, `stage9_junction_clusters.png`, `stage9_node_clusters.json`, `stage9_node_cluster_summary.json` | DONE |
| Slice 10 | Stage 10 traced-edge extraction from the Stage 7 skeleton and Stage 9 clustered nodes | `stage10_pipe_edges.json`, `stage10_pipe_edges_overlay.png`, `stage10_pipe_edge_summary.json` | DONE |
| Slice 11 | Stage 11 junction review with confirmed vs unresolved junction outputs | `stage11_confirmed_junctions.png`, `stage11_unresolved_junctions.png`, `stage11_junctions.json`, `stage11_junction_review_summary.json` | DONE |
| Slice 12 | Stage 12 provisional graph assembly from clustered nodes, traced edges, reviewed junctions, provisional equipment attachments, fused line-number text attachments, and instrument semantic attachments with adaptive/toleranced thresholding | `stage12_equipment_attachments.json`, `stage12_equipment_attachment_summary.json`, `stage12_text_attachments.json`, `stage12_text_attachment_summary.json`, `stage12_text_attachment_overlay.png`, `stage12_instrument_tag_attachments.json`, `stage12_instrument_tag_attachment_summary.json`, `stage12_graph.json`, `stage12_graph_summary.json` | DONE |
| Slice 13 | Stage 13 graph-native QA and machine-readable review queue | `stage13_graph_anomalies.json`, `stage13_review_queue.json`, `stage13_graph_qa_summary.json` | DONE |
