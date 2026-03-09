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
| Staged orchestrator | [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py) currently implements `stage1_input_normalization`, `stage2_ocr_discovery`, `stage4_object_detection`, `stage5_pipe_mask`, `stage6_morphological_sealing`, `stage7_skeleton_generation`, `stage8_skeleton_node_detection`, `stage9_node_clustering`, `stage10_edge_tracing`, `stage11_junction_review`, `stage12_graph_assembly`, and `stage13_graph_qa` | Active | Stage numbering is intentionally sparse and the rebuild now reaches provisional graph + QA, but semantic attachment/export refinement are still incomplete |
| OCR ingestion | [`backend/garnet/text_ocr.py`](/Users/maetee/Code/GARNET/backend/garnet/text_ocr.py), [`backend/garnet/easyocr_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/easyocr_sahi.py), planned Gemini route helper under `backend/garnet/`, Gemini adapter in [`backend/gemini_detector/gemini_sahi.py`](/Users/maetee/Code/GARNET/backend/gemini_detector/gemini_sahi.py) | Active | The rebuild needs selectable OCR routing so a run uses either EasyOCR or Gemini, not a mandatory chained Stage 2 -> Stage 3 path |
| Object detection | `/api/detect` flow in [`backend/api.py`](/Users/maetee/Code/GARNET/backend/api.py), [`backend/garnet/object_detection_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/object_detection_sahi.py), and Stage 4 in [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py) | Active | Detection is now part of the staged rebuild, but it still uses a fixed baseline configuration and does not yet emit richer evidence tables |
| Pipe geometry | [`backend/garnet/pipe_mask.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_mask.py), [`backend/garnet/pipe_seal.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_seal.py), [`backend/garnet/pipe_skeleton.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_skeleton.py), [`backend/garnet/pipe_nodes.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_nodes.py), [`backend/garnet/pipe_node_clusters.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_node_clusters.py), [`backend/garnet/pipe_edges.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_edges.py), [`backend/garnet/pipe_junctions.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_junctions.py) | Active | Geometry path now exists end-to-end, but crossing semantics and cleanup are still heuristic and need later refinement |
| Graph assembly | [`backend/garnet/pipe_graph.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_graph.py), [`backend/garnet/pipe_graph_qa.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_graph_qa.py) | Active | Graph JSON and QA artifacts now exist, but export refinement, attachment, directionality, and API/UI exposure are still future work |
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
| S3-01 | Replace simple equipment snapping with multi-signal attachment scoring | C1 | `backend/garnet/pid_extractor.py` | Attachment report includes scores, chosen edge, and ambiguous candidates | TODO |
| S3-02 | Formalize inline-object association rules and edge-splitting thresholds | C2 | `backend/garnet/pid_extractor.py`, helper module to be created if needed | Inline association report shows confident splits vs unresolved items | TODO |
| S3-03 | Convert flow-arrow handling from visual overlay to edge-direction assignment and local propagation | C3 | `backend/garnet/pid_extractor.py`, `backend/schema/graph_v1.json` | Directed-edge metrics show assigned, propagated, and unresolved arrow-edge matches | TODO |
| S3-04 | Associate text to equipment, edges, inline objects, and off-page connectors using more than distance alone | C4 | `backend/garnet/text_ocr.py`, `backend/garnet/pid_extractor.py` | Text-link report shows target type, confidence, and unresolved text queue | TODO |
| S3-05 | Represent off-page connectors as explicit graph nodes with labels and page-reference fields | C5 | `backend/garnet/pid_extractor.py`, `backend/schema/graph_v1.json`, export helper to be created if needed | Export contains connector nodes with attachment metadata | TODO |

## Sprint 4 - Graph, QA, and export

**Objective**
- Make the graph the authoritative validation surface and keep exports practical.

**Definition of done**
- Graph schema supports provenance, confidence, review state, and direction.
- QA catches obvious topology anomalies.
- Export geometry is simplified enough for downstream use.

| ID | Task | Master-plan refs | Repo targets | Verification | Status |
|----|------|------------------|--------------|--------------|--------|
| S4-01 | Unify node and edge schema across in-memory graph, JSON export, and export adapter | D1-D2 | `backend/garnet/pid_extractor.py`, export helper to be created if needed, `backend/schema/graph_v1.json` | Export validation confirms required fields exist for nodes and edges | TODO |
| S4-02 | Add graph-native QA primitives for connected components, degree anomalies, articulation points, and orphan terminals | D3, E1 | `backend/garnet/pid_extractor.py`, `backend/garnet/pipe_graph_qa.py` | QA report JSON lists anomaly counts and affected node/edge ids | DONE |
| S4-03 | Generate an anomaly report and retry queue instead of only overlays | E1-E2 | `backend/garnet/pid_extractor.py`, `backend/garnet/pipe_graph_qa.py` | Output includes machine-readable anomaly and retry files | DONE |
| S4-04 | Add polyline simplification before export with configurable tolerance and compression metrics | D4 | graph/export helper modules to be created if needed | Compare export payload size before/after simplification | TODO |
| S4-05 | Expose graph QA and export outputs through the backend service where needed for later review tooling | D1-E1 | `backend/api.py`, `backend/schema/graph_v1.json` | API returns graph/QA artifacts for a sample run | TODO |

## Sprint 5 - Recovery loop and review boundary

**Objective**
- Limit human review to unresolved ambiguity and make recovery targeted rather than global.

**Definition of done**
- Recovery tasks are explicit and rerunnable.
- Manual review queue is small, categorized, and backed by artifact files or API payloads.

| ID | Task | Master-plan refs | Repo targets | Verification | Status |
|----|------|------------------|--------------|--------------|--------|
| S5-01 | Add targeted reprocessing hooks for OCR, attachment, crossing, arrow, and morphology retries | E2 | `backend/garnet/pid_extractor.py`, helper modules as needed | Retry queue items can trigger limited re-runs without replaying the whole sheet | TODO |
| S5-02 | Define unresolved queues for ambiguous crossings, text conflicts, uncertain direction, and connector mismatches | E3 | `backend/garnet/pid_extractor.py`, `backend/schema/graph_v1.json` | Manual-review queue JSON contains category, geometry, evidence refs, and priority | TODO |
| S5-03 | Add backend endpoints or export artifacts for review-ready unresolved cases | E3, V3 prep | `backend/api.py`, `backend/garnet/pid_extractor.py` | Sample run exposes unresolved queue through API or stable output artifact | TODO |
| S5-04 | Define the multi-sheet merge contract for off-page connectors without implementing the full merge engine yet | V3 prep | `backend/schema/graph_v1.json`, export helper to be created if needed | Schema/update note documents connector keys required for later merge work | TODO |

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
| Slice 3 | Selectable OCR routes: user chooses one OCR route per run, currently `easyocr`, `gemini`, or `paddleocr`, with Gemini using patched full-page prompts and route-local crop fallback | `stage2_ocr_regions.json`, `stage2_ocr_overlay.png`, `stage2_ocr_summary.json`, `stage2_ocr_exception_candidates.json`, optional Gemini raw patch/crop artifacts | DONE |
| Slice 4 | Fixed-baseline Stage 4 object detection using Ultralytics + SAHI with `backend/yolo_weights/yolo26n_PPCL_640_20260227.pt` | `stage4_objects.json`, `stage4_objects_overlay.png`, `stage4_objects_summary.json` | DONE |
| Slice 5 | Conservative Stage 5 pipe-mask generation from Stage 1 binaries with Stage 2 OCR suppression and Stage 4 object suppression | `stage5_pipe_mask.png`, `stage5_pipe_mask_overlay.png`, `stage5_pipe_mask_summary.json` | DONE |
| Slice 6 | Conservative Stage 6 morphological sealing over the Stage 5 mask with audit counters and reviewable overlays | `stage6_pipe_mask_sealed.png`, `stage6_pipe_mask_sealed_overlay.png`, `stage6_pipe_mask_sealed_summary.json` | DONE |
| Slice 7 | Stage 7 skeleton generation from the sealed mask with reviewable skeleton artifacts | `stage7_pipe_skeleton.png`, `stage7_pipe_skeleton_overlay.png`, `stage7_pipe_skeleton_summary.json` | DONE |
| Slice 8 | Stage 8 raw skeleton node detection with separate endpoint and junction candidate maps | `stage8_endpoints.png`, `stage8_junctions.png`, `stage8_nodes_overlay.png`, `stage8_node_summary.json` | DONE |
| Slice 9 | Stage 9 node clustering for endpoint and junction candidates | `stage9_endpoint_clusters.png`, `stage9_junction_clusters.png`, `stage9_node_clusters.json`, `stage9_node_cluster_summary.json` | DONE |
| Slice 10 | Stage 10 traced-edge extraction from the Stage 7 skeleton and Stage 9 clustered nodes | `stage10_pipe_edges.json`, `stage10_pipe_edges_overlay.png`, `stage10_pipe_edge_summary.json` | DONE |
| Slice 11 | Stage 11 junction review with confirmed vs unresolved junction outputs | `stage11_confirmed_junctions.png`, `stage11_unresolved_junctions.png`, `stage11_junctions.json`, `stage11_junction_review_summary.json` | DONE |
| Slice 12 | Stage 12 provisional graph assembly from clustered nodes, traced edges, and reviewed junctions | `stage12_graph.json`, `stage12_graph_summary.json` | DONE |
| Slice 13 | Stage 13 graph-native QA and machine-readable review queue | `stage13_graph_anomalies.json`, `stage13_review_queue.json`, `stage13_graph_qa_summary.json` | DONE |
