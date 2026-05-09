# Line Detection — Implementation Report & Integration Guide

> **Status**: Module written and profiled. Performance is insufficient for production use.
> The core algorithm is correct but the O(n²) merge phases are the blocking issue.
> Use this document as the specification for a separate implementation pass.

---

## What Was Built

**File**: `backend/garnet/line_detection_inpaint.py` (622 lines)

The module implements the Ali et al. (2026) geometric line-extraction pipeline — replacing GARNET's rectangular text-suppression mask approach with Telea inpainting of symbol regions.

### Pipeline Phases

```
A. adaptive_threshold_mask   → binary pipe candidate mask (Gaussian adaptive)
B. detect_corner_points      → Shi-Tomasi angle-change feature points (max 2000)
C. assemble_inpaint_mask     → grid-hash bboxes from corners + text + object boxes
D. inpaint_masked_region     → Telea inpainting removes symbols, preserves pipes
E. extract_contour_segments  → connected-component + Douglas-Peucker polyline
F. merge_collinear_segments  → greedy O(n²) collinearity merge + H/V split
G. merge_nearby_endpoints    → greedy O(n²) L-joint merge + orphan prune
```

### Public API

```python
from garnet.line_detection_inpaint import run_line_detection_inpaint, render_line_overlay

result = run_line_detection_inpaint(
    stage1_gray=gray,           # grayscale from Stage 1
    text_regions=[],             # Stage 2 OCR bboxes
    object_regions=[],           # Stage 4 detection bboxes
    image_id="sheet_01",
)
# Returns: segments, horizontal_segments, vertical_segments,
#          inpaint_mask, cleaned_gray, cleaned_binary, corner_points, summary
```

---

## Test Results

**Image**: `test/!test01.png` (1646 × 2183 px)
**Python**: GARNET `.venv` (cv2 4.10.0)

### Per-Phase Timing

| Phase | Function | Time | Notes |
|---|---|---|---|
| A | `_adaptive_threshold_mask` | 0.005s | Fast |
| B | `_detect_corner_points` | 0.018s | Fast; capped at 2000 corners |
| C1 | `_points_to_bboxes_fast` | 0.002s | Grid-hash O(n); 48 boxes |
| C2 | `_assemble_inpaint_mask` | 0.011s | Dilate kernel (21,21) |
| D | `_inpaint_masked_region` | 0.032s | Telea radius=5 |
| E1 | `_cleaned_to_binary` | 0.004s | Fast |
| E2 | `_extract_contour_segments` | **0.507s** | 8285 raw segments extracted |
| **F1** | **`_merge_collinear_segments`** | **34.332s** | **BOTTLENECK — O(n²) on 8285 items** |
| F2 | `_split_horizontal_vertical` | 0.001s | Fast |
| **G1** | **`_merge_nearby_endpoints`** | **20.976s** | **BOTTLENECK — O(n²) on 7340 items** |
| G2 | `_prune_orphan_segments` | 0.000s | Fast |

**Total: ~56 seconds** (dominated by two O(n²) merge phases)

### Segment Counts Through Pipeline

| Stage | Count |
|---|---|
| Raw contour segments | 8,285 |
| After collinear merge | 7,340 |
| After endpoint merge | 2,906 |
| Final (min_len=12px) | **1,704** (H=872, V=832) |

---

## What's Broken / Needs Fixing

### 1. O(n²) Collinear Merge (Phase F1) — 34s

**Current**: Greedy nested-loop pairwise comparison of all segments.

```python
# Current code — O(n²)
while changed:
    for i in range(len(merged)):
        for j in range(i+1, len(merged)):
            if _segments_are_collinear(merged[i], merged[j]):
                merged[i] = _merge_segment_pair(...)
                merged.pop(j)
                changed = True
```

**Problem**: With 8,285 raw segments, this is ~34 million comparisons. Each comparison involves angle computation, AABB overlap check, and IoU calculation.

**Fix**: Use **angle-bucket pre-grouping + interval-tree merging**:

1. Bucket segments by angle rounded to nearest 5° (36 buckets for 0–180°)
2. Within each bucket, sort by the dominant-axis min coordinate
3. Merge overlapping intervals in a single linear sweep (union-find)
4. Only run the expensive O(n²) check within same bucket (collinearity requires angle match first)

```python
from collections import defaultdict

def merge_collinear_segments_fast(segments: list[Segment]) -> list[Segment]:
    BUCKET_SIZE_DEG = 5.0
    buckets: dict[int, list[Segment]] = defaultdict(list)
    for seg in segments:
        ang = round(_segment_angle_deg(seg) / BUCKET_SIZE_DEG) * int(BUCKET_SIZE_DEG)
        buckets[ang].append(seg)

    result = []
    for ang, bucket in buckets.items():
        # Sort by dominant axis
        if ang <= 45 or ang >= 135:
            bucket.sort(key=lambda s: s["x1"])   # horizontal-dominant
        else:
            bucket.sort(key=lambda s: s["y1"])   # vertical-dominant

        # Linear sweep: merge overlapping segments in sorted order
        i = 0
        while i < len(bucket):
            current = bucket[i]
            j = i + 1
            while j < len(bucket) and _segments_are_collinear(current, bucket[j]):
                current = _merge_segment_pair(current, bucket[j])
                j += 1
            result.append(current)
            i = j
    return result
```

**Expected time**: ~1–3s (from 34s).

---

### 2. O(n²) Endpoint Merge (Phase G1) — 21s

**Current**: Same greedy nested-loop approach on ~7,340 post-merge segments.

**Fix**: Use a **2D spatial hash grid** with cell size = `ENDPOINT_MERGE_PX` (15px).

1. Build a hash table: cell_key = `(x // cell_px, y // cell_px)`
2. For each segment, insert its 2 endpoints into the grid
3. For each endpoint, check the 3×3 neighborhood of cells for nearby endpoints
4. Only do the O(n²) distance check on endpoints within the same cell neighborhood

```python
from collections import defaultdict

def merge_nearby_endpoints_fast(segments: list[Segment], *, merge_px=15.0) -> list[Segment]:
    cell_px = merge_px
    grid: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    # key = (col, row), value = list of (x, y, segment_index)

    for idx, seg in enumerate(segments):
        for px, py in [(seg["x1"], seg["y1"]), (seg["x2"], seg["y2"])]:
            key = (int(px // cell_px), int(py // cell_px))
            grid[key].append((px, py, idx))

    merged = list(segments)
    changed = True
    while changed:
        changed = False
        visited = set()
        for key, cell in grid.items():
            for i, (x1, y1, si) in enumerate(cell):
                if (key, i) in visited:
                    continue
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        neighbor_key = (key[0] + dx, key[1] + dy)
                        neighbor_cell = grid.get(neighbor_key, [])
                        for j, (x2, y2, sj) in enumerate(neighbor_cell):
                            if si == sj or (neighbor_key, j) in visited:
                                continue
                            if math.hypot(x1 - x2, y1 - y2) <= merge_px:
                                # Merge segments si and sj
                                candidate = _merge_endpoint_pair(merged[si], merged[sj])
                                if candidate:
                                    merged[si] = candidate
                                    merged.pop(sj)
                                    # Rebuild grid (expensive but only runs a few times)
                                    changed = True
                                    break
                        if changed:
                            break
                    if changed:
                        break
                visited.add((key, i))
    return merged
```

**Expected time**: ~0.5–2s (from 21s).

---

### 3. Corner Detection Quality (Phase B)

Shi-Tomasi finds up to 2000 corners on the pipe mask. The corners are dominated by pipe intersection geometry (which is correct) but also picks up noise from text edges and symbol outlines (which is noise for inpainting purposes).

**Current params**:
```python
CORNER_MAX_CORNERS = 2000
CORNER_QUALITY_LEVEL = 0.02
CORNER_MIN_DISTANCE = 8
```

**Observation**: 2000 corners → 48 bounding boxes after grid-hash grouping. This is reasonable but the mask has 67,451 white pixels — meaning corner density is low relative to symbol area.

**Alternative approach**: Instead of (or in addition to) Shi-Tomasi, consider:
- `cv2.LSD` (Line Segment Detector) for pipe geometry
- Morphological top-hat to isolate thin pipe strokes
- Keep Shi-Tomasi only for junction/intersection regions

---

### 4. Inpaint Mask Assembly (Phase C)

The grid-hash approach (`_points_to_bboxes_fast`) runs in 0.002s — correctly fast. However, it relies on `cv2.groupRectangles` which has a quirk:

**Requirement**: `groupRectangles` requires integer `(x, y, w, h)` tuples. Float tuples cause `TypeError: Sequence item with index 0 has wrong type`. This was patched — verify any future edits maintain integer casts.

---

## Integration into GARNET Pipeline

### Where It Fits (Stage 5)

```python
# In pid_extractor.py — Stage 5 slot

def stage5_line_detection(state: PipelineState) -> PipelineState:
    cfg = state.pipeline_config

    if cfg.get("use_inpaint_line_detection", False):
        from garnet.line_detection_inpaint import run_line_detection_inpaint
        result = run_line_detection_inpaint(
            stage1_gray=state.images["normalized_gray"],
            text_regions=state.ocr_text_regions,    # from Stage 2
            object_regions=state.detected_objects,  # from Stage 4
            image_id=state.image_id,
        )
        state.line_segments = result["segments"]
        state.horizontal_segments = result["horizontal_segments"]
        state.vertical_segments = result["vertical_segments"]
        # Optionally cache inpainted images for debugging
        state.debug["inpaint_mask"] = result["inpaint_mask"]
        state.debug["cleaned_gray"] = result["cleaned_gray"]
    else:
        # Existing pipe_mask pipeline
        state = stage5_pipe_mask(state)

    return state
```

### Output Contract

`run_line_detection_inpaint` returns:
- `segments`: Full list of `{"x1", "y1", "x2", "y2", "length", ...}` dicts
- `horizontal_segments`: Subset with angle within 45° of horizontal
- `vertical_segments`: Subset with angle within 45° of vertical
- `inpaint_mask`: `H×W uint8` mask used for Telea
- `cleaned_gray`: `H×W uint8` inpainted grayscale
- `cleaned_binary`: `H×W uint8` thresholded cleaned image
- `corner_points`: `Nx2` array of Shi-Tomasi corner coordinates
- `summary`: Stats dict for logging

### Downstream Consumers

| Stage | Consumer | Expected Input |
|---|---|---|
| Stage 6 | Graph construction | `horizontal_segments` + `vertical_segments` |
| Stage 7 | Pipe continuity | Same |
| Stage 8 | Junction detection | Same |
| Stage 9 | Arrow directionality | `segments` + pipe connectivity |
| Stage 10 | Continuity check | Same |

---

## Parameters to Tune

| Parameter | Current | Range | Effect |
|---|---|---|---|
| `CORNER_MAX_CORNERS` | 2000 | 500–5000 | More corners = more symbol coverage but slower |
| `CORNER_QUALITY_LEVEL` | 0.02 | 0.001–0.1 | Lower = more corners, higher = fewer but stronger |
| `CORNER_GRID_CELL_PX` | 40 | 20–80 | Smaller = finer bbox grouping, larger = coarser |
| `INPAINT_DILATE_KERNEL` | (21,21) | (5,5)–(41,41) | Larger = more aggressive inpainting |
| `INPAINT_RADIUS` | 5 | 3–15 | Telea radius; affects gap-filling quality |
| `MIN_COMPONENT_AREA` | 15 | 10–50 | Filter tiny noise components |
| `MAX_COMPONENT_AREA` | 500,000 | — | Upper bound to skip large background regions |
| `COLLINEAR_ANGLE_TOLERANCE_DEG` | 15.0 | 5–20 | Angle difference to still consider collinear |
| `COLLINEAR_IoU_THRESHOLD` | 0.15 | 0.1–0.3 | Overlap ratio to trigger merge |
| `HV_ANGLE_DEG` | 45.0 | 30–60 | Threshold between H and V classification |
| `ENDPOINT_MERGE_PX` | 15.0 | 8–30 | Distance to trigger L-joint merge |
| `MIN_SEGMENT_LENGTH_PX` | 12.0 | 5–25 | Orphan/noise filter |

---

## Files Produced

| File | Purpose |
|---|---|
| `backend/garnet/line_detection_inpaint.py` | Core module — all phases A–G |
| `backend/scripts/demo_line_inpaint.py` | Standalone demo with 7 output artifacts |
| `backend/scripts/debug_timing.py` | Per-phase profiler (use to validate optimizations) |
| `backend/tests/test_line_detection_inpaint.py` | Unit tests for geometry functions |
| `docs/line_detection_gap_analysis.md` | Previous gap analysis (pipeline context) |

---

## References

- Ali et al. (2026) — `ssrn-6083108.pdf`; geometric line extraction via adaptive threshold + corner detection + contour merge
- Azure P&ID Solution (`Azure-Samples/pid-solution/`) — thinning + Hough fallback; Stage 6–10 design docs
- GARNET existing: `pipe_mask.py`, `pipe_skeleton.py`, `pipe_crossings.py`, `pipe_edges.py`, `pipe_continuity_helpers.py`
