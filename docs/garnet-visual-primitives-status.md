# Visual Primitives Pipeline — Status Summary

**Date:** 2026-05-23
**Branch:** `feat/geometric-line-extraction`
**Commit:** `0a17868`

## What We Built

### Agent 1: Global Equipment Detector
- File: `backend/garnet/visual_primitives/agent1_equipment_detector.py`
- VLM-based equipment detection on full P&ID sheets
- Uses Gemini 2.5 Pro via OpenRouter
- Protocol: `<|ref|>CLASS<|/ref|><|box|>[[coords]]<|/box|>`
- 7-class taxonomy + "other" with confidence scoring
- Canvas module with adaptive downsampling for large sheets
- 9 equipment detected on Test-00001, clean protocol usage

### Agent 2: Pipeline Tracers (Three Implementations)

#### 2a. Pure VLM Tracer
- File: `backend/garnet/visual_primitives/agent2_pipeline_tracer.py`
- Step-by-step VLM tracing: crop around cursor → VLM decides next step → repeat
- Red crosshair + green visited-path overlay
- Costs: ~14 min per sheet, 100K+ tokens, 25 segments
- Status: functional but slow — baseline reference

#### 2b. Hybrid CV + VLM Tracer (default)
- File: `backend/garnet/visual_primitives/agent2_hybrid.py` (`--tracer cv`)
- CV pipe follower: `cv_pipe_follower.py` — walks binary pipe mask at ~1000 px/ms
- CV directions: UP/DOWN/LEFT/RIGHT only, threshold=2 for thin-line detection
- VLM called only for: port detection (start point), terminal classification (end point)
- P&ID rules baked in:
  - Lines run horizontal/vertical only
  - Main line continues straight through junctions
  - 90-degree corners detected via window scanning
  - Gap bridging for inline objects (valves, reducers)
- Results: 15,465px traced, 87s, 9 segments, 15 VLM calls
- 9% of pure-VLM runtime, 7% of token cost

#### 2c. VLM Step-by-Step Tracer
- File: `agent2_hybrid.py` (`--tracer vlm`)
- VLMCursor: `vlm_cursor.py` — grey dot + line marking on temp image copy
- Protocol tokens: `<|go|>DIR DIST`, `<|turn|>DIR`, `<|hit|>CLASS`, `<|jump|>DIR DIST`, `<|term|>CLASS`
- Crop + mark → VLM decides direction → cursor moves → repeat
- VLM sees actual drawing (not binary mask) — identifies instrument tags, junctions, vessels naturally
- Results: 5,234px traced, 393s, 9 segments, 74 VLM calls
- Key strength: instrument tags and equipment classified directly by VLM (no YOLO dependency)

### Supporting Modules

| File | Role |
|------|------|
| `prompts.py` | All VLM system/user prompts (Agent 1, Agent 2, port finder, classifier, VLM tracer) |
| `schemas.py` | Pydantic models: EquipmentEntry, EquipmentRegistry, EquipmentClass, Confidence |
| `canvas.py` | Image loading, downsizing, coordinate normalization [0,999] |
| `patch_utils.py` | Crop extraction with grid overlay |
| `response_parser.py` | VLM response parsing: `<box>` extraction, IoU merge, coordinate conversion |
| `vlm_trace_parser.py` | Parse `<go>`, `<turn>`, `<hit>`, `<jump>`, `<term>` tokens |
| `cursor.py` | VLM trace cursor for Agent 2 (pre-cursor to vlm_cursor) |

### Port Detection
- VLM port finder in `agent2_pipeline_tracer.py`: crops connection symbol, VLM returns EDGE + FRACTION
- Corrected 3 of 9 heuristic errors on Test-00001
- Falls back to heuristic (bbox proportions) when VLM fails

### Tests
- 47 unit tests in `backend/tests/test_visual_primitives/`
- Covering: canvas, schemas, parser, cursor, trace parser, agent2

## Ground Truth vs. Results

### Test-00001 Page Connection Paths

| PC | Ground Truth | CV Result | VLM Result |
|----|-------------|-----------|------------|
| obj_000190 | UP→RIGHT→END(jct) | page_connection 3225px ✓ path correct, overshoots | junction 168px (short) |
| obj_000191 | RIGHT→DOWN→jump→END(tag) | dead_end 735px ✗ | vessel 1356px ✗ wrong terminal |
| obj_000192 | UP→jump→UP→END(tag) | dead_end 1970px ✗ | **instrument tag 583px ✓** |
| obj_000193 | LEFT→UP→LEFT→reducer→END(jct) | dead_end 1960px ~ | page_connection 0px ✗ port |
| obj_000194 | RIGHT→UP→RIGHT→UP→END(V-2501) | dead_end 1010px ~ | junction 249px ✗ |
| obj_000195 | LEFT→UP→jump→reducer→valve→UP→LEFT→END(V-2501) | dead_end 665px ✗ | page_connection 121px ✗ |
| obj_000196 | UP→jump→UP→END(tag) | dead_end 35px ✗ port | max_steps 1658px ~ |
| obj_000197 | RIGHT→DOWN→check_valve→valve→END(jct) | connection 2860px ~ | junction 1099px ~ |
| obj_000198 | LEFT→reducer→LEFT→END(V-2501) | dead_end 3005px ~ | page_connection 0px ✗ port |

## Key Learnings

1. **CV is fast but blind.** Mask-walking misses instrument tags, text labels, and can't distinguish pipe from annotation noise (3500+ components on raw mask).

2. **VLM sees the drawing.** VLM step-tracer correctly identifies instrument tags as terminals and distinguishes junctions from corners — things CV can't do.

3. **Port detection is critical.** Wrong initial direction kills the trace. VLM port finder is reliable (~8/9 correct) but slow (9 VLM calls before tracing starts).

4. **Visited-path marking prevents backtracking.** Line-drawing visited marks (5px thick) eliminated the backtracking issue that plagued earlier CV runs.

5. **Junction "END" rule needs more thought.** P&ID convention says page connection traces end at junctions, but programmatically distinguishing "turn at tee" from "end at tee" is nontrivial.

## Way Forward

### Short-term (improve VLM step tracer)
1. Crop size tuning: 350px may be too small. Try 500px for longer jumps.
2. Max steps increase: 25 steps limits traces to ~2000px. Try 40.
3. Port crop issue: obj_000193/198 get 0px because VLM sees "page_connection" at the port. Need to teach VLM to ignore the origin symbol and trace outward.
4. gap/jump testing: VLM should return `<jump>` for inline objects — needs prompt refinement.

### Medium-term (hybrid approach)
5. Best-of-both: use CV for straight-line walking (fast), VLM for turns and classification (accurate).
6. Multi-sheet testing: run on Test-00003, Test-00008 to validate generalization.
7. Performance targets: <2 min/sheet for CV mode, <5 min/sheet for VLM mode.

### Long-term (productization)
8. Batch runner: single command to process all PPCL test sheets.
9. SQLite output: store traces, terminals, equipment in queryable format.
10. Overlay export: annotated P&ID PNG with pipe colors, equipment labels, connection IDs.
11. Graph output: pipe connectivity graph (which equipment connects to what).
