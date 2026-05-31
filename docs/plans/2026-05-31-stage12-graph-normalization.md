# Stage 12 Graph Normalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Normalize the Stage 12 trace-edge graph so branch starts, tee terminals, and intersecting trace endpoints become meaningful shared topology nodes.

**Architecture:** Keep Stage 5b walking and Stage 11 associations unchanged. Add a normalization pass inside `backend/garnet/trace_graph_builder.py` that transforms Stage 11 trace edges before graph assembly: split edges at geometric junctions, merge coincident nodes, collapse duplicate reverse paths, and preserve review evidence. Stage 12 QA should then operate on normalized graph output and show reduced connected components and tee-degree mismatches.

**Tech Stack:** Python, plain dict JSON payloads, `unittest`, existing Stage 11/12 artifacts, OpenCV overlays already used by the pipeline.

---

### Task 1: Add Synthetic Tests For Branch-Start Merge

**Files:**
- Modify: `backend/tests/test_trace_graph_builder.py` or create it if missing
- Modify: `backend/garnet/trace_graph_builder.py`

**Step 1: Write failing test**

Create a synthetic Stage 11 payload with:
- one main trace from `A -> tee_junction J`
- one branch trace whose source point lies on the main trace segment
- branch terminal to equipment

Expected:
- graph has one shared junction node at the branch source point
- main trace is split into two edges at that junction
- branch trace source uses the same junction node, not a separate `branch_start` node
- the junction degree is at least 3

Example test shape:

```python
def test_branch_start_on_main_trace_merges_into_junction_and_splits_main_edge():
    payload = {
        "image_id": "synthetic.png",
        "trace_source": "stage11_trace_associations",
        "trace_edges": [
            {
                "trace_id": "obj_main",
                "trace_kind": "port",
                "source_obj_id": "obj_main",
                "source_obj_type": "page_connection",
                "port": {"x": 0, "y": 100, "direction": "RIGHT"},
                "terminal_type": "tee_junction",
                "terminal_obj_id": "node_1",
                "terminal_xy": [200, 100],
                "segments": [{"x1": 0, "y1": 100, "x2": 200, "y2": 100, "direction": "RIGHT", "length_px": 200}],
                "polyline": [{"x": 0, "y": 100}, {"x": 200, "y": 100}],
                "attachments": {"line_numbers": [{"id": "line_1"}]},
                "status": "ok",
            },
            {
                "trace_id": "branch_000001",
                "trace_kind": "branch",
                "source_obj_id": "branch_000001",
                "source_obj_type": "branch_candidate",
                "port": {"x": 100, "y": 100, "direction": "DOWN"},
                "terminal_type": "equipment",
                "terminal_obj_id": "equip_1",
                "terminal_xy": [100, 200],
                "segments": [{"x1": 100, "y1": 100, "x2": 100, "y2": 200, "direction": "DOWN", "length_px": 100}],
                "polyline": [{"x": 100, "y": 100}, {"x": 100, "y": 200}],
                "attachments": {"line_numbers": [{"id": "line_1"}]},
                "status": "traced",
            },
        ],
    }
    result = build_trace_graph_from_stage11(payload, image_id="synthetic.png")
    graph = result["graph_payload"]
    junction_nodes = [node for node in graph["nodes"] if node["type"] == "tee_junction"]
    assert len(junction_nodes) == 1
    junction_id = junction_nodes[0]["id"]
    degree = sum(1 for edge in graph["edges"] if edge["source"] == junction_id or edge["target"] == junction_id)
    assert degree >= 3
```

**Step 2: Run test and verify it fails**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest backend.tests.test_trace_graph_builder -v
```

Expected: fails because branch source remains a `branch_start` node and the main edge is not split.

**Step 3: Commit test only if using separate commits**

Do not commit yet if executing in one small implementation batch.

---

### Task 2: Build Geometry Utilities For Axis-Aligned Edge Splitting

**Files:**
- Modify: `backend/garnet/trace_graph_builder.py`
- Test: `backend/tests/test_trace_graph_builder.py`

**Step 1: Add helper tests**

Add tests for:
- point lies on horizontal segment within tolerance
- point lies on vertical segment within tolerance
- split polyline at interior point creates two non-empty polylines
- split ignores endpoint-only duplicate splits

**Step 2: Implement helpers**

Add private helpers:

```python
def _point_near_axis_segment(point, start, end, tolerance_px): ...
def _project_point_to_axis_segment(point, start, end): ...
def _split_polyline_at_points(polyline, split_points, tolerance_px): ...
def _dedupe_split_points(points, tolerance_px): ...
```

Rules:
- Only split axis-aligned segments.
- Preserve original edge direction order.
- Do not create edges shorter than `min_split_edge_length_px`, default `8`.
- Retain trace metadata on child edges with suffixes like `trace::obj_main::part_001`.

**Step 3: Run helper tests**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest backend.tests.test_trace_graph_builder -v
```

Expected: helper tests pass, branch merge test may still fail until normalization is wired.

---

### Task 3: Add Stage 12 Normalization Pass Before Node Registry Assembly

**Files:**
- Modify: `backend/garnet/trace_graph_builder.py`
- Test: `backend/tests/test_trace_graph_builder.py`

**Step 1: Add normalization entrypoint**

Add:

```python
def normalize_stage11_trace_edges(trace_edges, *, split_tolerance_px=10.0, merge_tolerance_px=12.0):
    ...
```

Inputs are Stage 11 trace-edge dicts. Output should be normalized trace-edge dicts plus normalization metadata.

**Step 2: Detect branch source split points**

For every `trace_kind == "branch"` edge:
- read source point from `port`
- find non-branch trace segments that contain that point within tolerance
- add split point to the containing trace
- mark the branch source as a `tee_junction` source override, not `branch_start`

**Step 3: Detect tee terminal split points**

For every edge with `terminal_type == "tee_junction"`:
- use `terminal_xy`
- split any other trace segment that contains that point
- assign stable junction id from `terminal_obj_id` when available

**Step 4: Split source traces**

For each trace edge with split points:
- split polyline/segments into child edges
- child edges inherit attachments, hits, line numbers, source metadata, and terminal metadata where applicable
- child edge endpoint node types are based on split point roles: original source, intermediate tee_junction, original terminal

**Step 5: Wire into `build_trace_graph_from_stage11`**

At the start of `build_trace_graph_from_stage11`, normalize `payload["trace_edges"]` into an internal edge list. Preserve original payload in metadata.

**Step 6: Run tests**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest backend.tests.test_trace_graph_builder -v
```

Expected: branch merge test passes.

---

### Task 4: Collapse Duplicate And Reverse Physical Paths

**Files:**
- Modify: `backend/garnet/trace_graph_builder.py`
- Test: `backend/tests/test_trace_graph_builder.py`

**Step 1: Add failing test**

Create two traces with the same endpoints and nearly identical polylines, one reversed. Expected:
- only one promoted physical edge remains
- metadata records the duplicate trace id in `merged_trace_ids`
- review queue gets an info item, not a high QA issue

**Step 2: Implement duplicate collapse**

Add helper:

```python
def _collapse_duplicate_trace_edges(edges, endpoint_tolerance_px=8.0): ...
```

Rules:
- If endpoints match same or reversed within tolerance and line numbers do not conflict, merge.
- Preserve all trace ids in `merged_trace_ids`.
- Merge attachments by group, de-duplicating by id.
- If line numbers conflict, do not collapse; add review item `possible_duplicate_conflicting_line_number`.

**Step 3: Run tests**

Run the same unittest command.

---

### Task 5: Improve Node Type Semantics After Normalization

**Files:**
- Modify: `backend/garnet/trace_graph_builder.py`
- Test: `backend/tests/test_trace_graph_builder.py`

**Step 1: Add tests**

Cases:
- branch source on main line becomes `tee_junction`, not `branch_start`
- terminal `tee_junction` with no object id merges by position
- equipment port nodes remain distinct from equipment terminal nodes
- page/utility connections keep stable ids

**Step 2: Add source/terminal override support**

Normalized edge dicts may include:

```json
{
  "source_node_override": {"type": "tee_junction", "stable_id": "junction::...", "position": {...}},
  "terminal_node_override": {...}
}
```

Update graph assembly to use overrides before `_source_node_type()` and `_terminal_node_type()`.

**Step 3: Run tests**

Run the same unittest command.

---

### Task 6: Add Normalization Summary And Overlay Metadata

**Files:**
- Modify: `backend/garnet/trace_graph_builder.py`
- Modify: `backend/garnet/pid_extractor.py`

**Step 1: Extend builder return payload**

Add:

```json
{
  "normalization_summary": {
    "input_trace_edge_count": 0,
    "output_trace_edge_count": 0,
    "split_trace_count": 0,
    "split_point_count": 0,
    "merged_duplicate_count": 0,
    "branch_start_merge_count": 0
  },
  "normalization_payload": {...}
}
```

**Step 2: Save artifacts in Stage 12**

In `stage12_geometric_graph_assembly`, save:
- `stage12_graph_normalization.json`
- `stage12_graph_normalization_summary.json`

**Step 3: Overlay support**

Add split points and merged junction markers to `stage12_graph_overlay.png` if useful, or create separate `stage12_graph_normalization_overlay.png` if the overlay gets noisy.

**Step 4: Run compile check**

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py
```

---

### Task 7: Run Current Test Image Regression

**Files:**
- Generated only under `backend/output_debug/Test-0000X/`

**Step 1: Run Stage 11/12 only for all test images**

Use the existing direct method runner pattern:

```bash
cd /Users/maetee/Code/GARNET/backend
for t in Test-00001 Test-00002 Test-00003 Test-00004 Test-00005 Test-00006 Test-00007 Test-00008 Test-00009; do
  /Users/maetee/Code/GARNET/.venv/bin/python - "/Users/maetee/Code/GARNET/backend/output_debug/${t}" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path
from garnet.pid_extractor import PIDPipeline, PipelineConfig
out_dir = Path(sys.argv[1]).resolve()
manifest = json.loads((out_dir / "stage_manifest.json").read_text(encoding="utf-8"))
pipe = PIDPipeline(
    image_path=str(manifest["image_path"]),
    output_dir=str(out_dir),
    cfg=PipelineConfig(use_geometric_line_detection=True),
)
pipe.stage11_trace_associations()
pipe.stage12_geometric_graph_assembly()
pipe.stage12c_page_connector_labeling()
pipe.stage12b_graph_export()
PY
done
```

**Step 2: Compare before/after metrics**

Expected direction:
- `connected_component_count` should decrease.
- `tee_degree_mismatch` should decrease.
- `duplicate_physical_path` should decrease or move to review metadata.
- `edge_count` may increase because main traces are split at tees.
- `node_count` may decrease or stay similar depending on merge count.

**Step 3: Inspect overlays**

Open/review:
- `stage12_graph_overlay.png`
- `stage12_graph_qa_overlay.png`
- `stage12_graph_normalization_overlay.png` if created

---

### Task 8: Update Stage 12 QA Expectations

**Files:**
- Modify: `backend/garnet/trace_graph_qa.py`
- Test: `backend/tests/test_trace_graph_builder.py` or new `backend/tests/test_trace_graph_qa.py`

**Step 1: Adjust tee degree semantics**

After normalization, `tee_junction` degree less than 3 should remain high severity. This becomes a stronger signal.

**Step 2: Add branch-start residual QA**

If any `branch_start` nodes remain after normalization, emit medium/high QA:

```python
category="unmerged_branch_start"
message="Branch start was not merged into a physical trace junction."
```

**Step 3: Run QA tests and current image regression**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py
```

---

### Task 9: Final Verification And Commit

**Files:**
- Verify all modified files

**Step 1: Run narrow tests**

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest backend.tests.test_trace_graph_builder -v
```

**Step 2: Run broader backend checks**

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v
```

**Step 3: Run diff hygiene**

```bash
cd /Users/maetee/Code/GARNET
git diff --check
git status --short
```

**Step 4: Commit**

```bash
cd /Users/maetee/Code/GARNET
git add backend/garnet/trace_graph_builder.py backend/garnet/trace_graph_qa.py backend/garnet/pid_extractor.py backend/tests/test_trace_graph_builder.py docs/plans/2026-05-31-stage12-graph-normalization.md
git commit -m "feat: normalize stage12 trace graph topology"
```

---

## Acceptance Criteria

- Stage 5b artifacts are unchanged by this work.
- Stage 11 association schema remains backward-compatible.
- Stage 12 graph has normalized junction nodes where branch starts meet main traces.
- Stage 12 graph splits main traces at branch/tee points.
- Stage 12 QA reports fewer tee-degree mismatches on Test 01-09.
- Connected component count decreases on at least the dense multi-branch test images.
- New normalization artifacts explain every split/merge decision.
- Existing graph export still writes `stage12b_graph_v1.json`.

## Non-Goals

- Do not change walking logic.
- Do not change OCR or object detection.
- Do not solve line-number association in this slice.
- Do not infer process direction or flow semantics beyond preserving existing arrows and line attachments.
