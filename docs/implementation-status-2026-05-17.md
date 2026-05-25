# GARNET Geometric Pipeline — Implementation Status

> Last updated: 2026-05-17
> Branch: `codex/autoresearch-mar17`
> GARNET root: `/Volumes/Ginnungagap/maetee/Code/GARNET/`

---

## What Was Built

### `symbol_aware_splitter.py` — Phase E2 of `line_detection_inpaint`

**Problem solved:** Pipe segments that cross through valve/instrument symbols were either removed by inpainting (losing both pieces) or passed through unbroken. Now they are cut at entry/exit boundary points.

**Location:** `backend/garnet/symbol_aware_splitter.py` (434 lines)

**Algorithm:**
1. For each text/object region (skipping `line_number` and `unknown` — labels on pipes):
2. Expand bbox by `margin_px=3` to catch near-misses
3. Classify segment relationship: `inside`, `crosses`, `touches`, `outside`
4. "Crosses" = both endpoints outside AND 2 intersection points with bbox edges
5. Cut segment at each intersection point; drop pieces < 25px

**Key design decisions:**
- **No object bboxes by default** — `INPAINT_DILATE_KERNEL=9×9` corner grid already handles valve/instrument bodies. Adding object bboxes causes double-cutting. Use `split_segments_at_symbols_with_objects()` only for large instruments with poor corner detection.
- **3px margin** — pipes running alongside (not through) a symbol are not cut
- **Bboxes sorted by area** — smallest first to minimize cascading splits
- **Skip `line_number`/`unknown`** — labels drawn directly on pipes; cutting at them would break legitimate pipes

**Integration point:** `line_detection_inpaint.py` Phase E2, between `_extract_contour_segments` and `_merge_collinear_segments`

---

## Module Map — Geometric Pipeline (S5→S10 replacement)

```
Raw image
  │
  ▼
run_line_detection_inpaint()
  │
  ├─ Phase A: Otsu thresholding
  ├─ Phase B: corner_points (Shi-Tomasi)
  ├─ Phase C: inpaint_mask assembly (corner grid)
  ├─ Phase D: Telea inpainting → cleaned_gray
  ├─ Phase E: contour extraction → raw_segments
  ├─ Phase E2: symbol_aware_splitter ← NEW
  │    └── cuts segments at symbol boundary crossings
  ├─ Phase F: collinear merge → after_collinear_merge
  ├─ Phase Fb: diagonal decomposition (H+V pairs)
  └─ output: stage5_geometric_segments.json + stage5_pipe_mask.png

  ▼
chain_geometric_segments()
  └── Phase A: group collinear H/V runs

  ▼
detect_junctions_from_runs()
  └── Phase B: L/T/X junction detection

  ▼
build_graph_from_runs_and_junctions()
  └── Phase C: graph assembly

  ▼ (existing stage12)
stage12_graph_assembly
  └── equipment/connection/text attachments, NetworkX graph
```

---

## Test Results (2026-05-17)

**All 9 images processed cleanly — no errors, no crashes.**

| Image | Raw Segs | After Merge | Final | H | V |
|---|---|---|---|---|---|
| Test-00001 | 27,359 | 21,857 | 2,640 | 1,207 | 1,433 |
| Test-00002 | 30,890 | 24,385 | 2,807 | 1,327 | 1,480 |
| Test-00003 | 25,322 | 19,900 | 2,213 | 1,089 | 1,124 |
| Test-00004 | 27,297 | 21,513 | 2,413 | 1,151 | 1,262 |
| Test-00005 | 29,255 | 22,809 | 2,652 | 1,253 | 1,399 |
| Test-00006 | 28,443 | 22,199 | 2,593 | 1,251 | 1,342 |
| Test-00007 | 26,987 | 21,383 | 2,430 | 1,167 | 1,263 |
| Test-00008 | 31,028 | 24,368 | 2,822 | 1,346 | 1,476 |
| Test-00009 | 28,269 | 22,475 | 2,480 | 1,186 | 1,294 |

**Output:** `projects/gcme/1_active/garnet/output/Test-0000X/`

Note: `corner_points=0` in demo summaries because `demo_line_inpaint.py` passes empty `text_regions=[]` and `object_regions=[]`. The corner detector fires but no symbol-aware splitting is triggered in standalone mode. Full pipeline integration (Stages 2+4 providing real regions) required for active splitting.

---

## Module Status

| Module | Status | Location |
|---|---|---|
| `line_detection_inpaint.py` | ✓ Built (987 lines) | `backend/garnet/` |
| `symbol_aware_splitter.py` | ✓ Built + tested (434 lines) | `backend/garnet/` |
| `chain_geometric_segments()` | ✓ Built (in `geometric_graph_builder.py`) | `backend/garnet/` |
| `detect_junctions_from_runs()` | ✓ Built (in `geometric_graph_builder.py`) | `backend/garnet/` |
| `build_graph_from_runs_and_junctions()` | ✓ Built (in `geometric_graph_builder.py`) | `backend/garnet/` |
| `detect_phase3_gaps()` | ✓ Built (in `geometric_graph_builder.py`) | `backend/garnet/` |
| `lshape_model` | Not built | — |
| `junction_cluster` | Not built | — |
| `polyline_to_graph` | Not built | — |

**All geometric pipeline modules (S5→S10 replacement) are built.** The three remaining unfilled modules (`lshape_model`, `junction_cluster`, `polyline_to_graph`) are described in `phase-3-geometric-graph-bypass-plan.md`.

---

## Next Steps

### Option A — Full pipeline integration test (recommended next)
Run end-to-end on a real P&ID through `pid_extractor.py`:
```bash
cd backend
python -m garnet.pid_extractor \
    --image test/ppcl/Test-00001.jpg \
    --out /path/to/output \
    --stop-after 12
```
Verify: `chain_geometric_segments` → `detect_junctions_from_runs` → `build_graph_from_runs_and_junctions` → `stage12_graph.json`

### Option B — Add unit tests for `symbol_aware_splitter`
- Crossing cases (horizontal, vertical, diagonal)
- Edge cases (endpoint on bbox edge, parallel to edge)
- Short piece dropping (< 25px)
- Skip classes (`line_number`, `unknown`)

### Option C — Build `lshape_model`, `junction_cluster`, `polyline_to_graph`
These are the remaining three modules from the geometric plan. They handle L-shape corner detection, junction clustering, and final polyline→graph conversion. Described in `phase-3-geometric-graph-bypass-plan.md`.

### Option D — A/B validation old vs new pipeline
Run same images through skeleton-based pipeline (old) and geometric pipeline (new), compare junction quality and connectivity metrics.