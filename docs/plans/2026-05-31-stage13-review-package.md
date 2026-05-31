# Stage 13 Review Package Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a process-facing Stage 13 review package from Stage 12 graph QA, Stage 12 review queue, and Stage 11 line-number HITL artifacts.

**Architecture:** Stage 13 is non-destructive: it does not mutate `stage12_graph.json`. It consolidates Stage 12 graph issues and Stage 12 review items into ranked, grouped HITL review records with enough geometry/evidence for a future review UI. It emits `stage13_review_items.json`, `stage13_review_summary.json`, and `stage13_review_overlay.png`.

**Tech Stack:** Python, existing JSON artifacts, OpenCV overlay rendering, `unittest`.

---

### Task 1: Add Stage 13 Review Builder Module

**Files:**
- Create: `/Users/maetee/Code/GARNET/backend/garnet/stage13_review_package.py`
- Create: `/Users/maetee/Code/GARNET/backend/tests/test_stage13_review_package.py`

**Step 1: Write failing test for QA issue conversion**

Create a test that calls `build_stage13_review_package(...)` with:

```python
graph_payload = {
    "image_id": "synthetic.png",
    "nodes": [{"id": "junction::1", "type": "tee_junction", "position": {"x": 10, "y": 20}}],
    "edges": [],
    "review_queue": [],
}
stage12_qa_payload = {
    "image_id": "synthetic.png",
    "issues": [
        {
            "id": "qa::tee_degree_mismatch::junction::1",
            "category": "tee_degree_mismatch",
            "severity": "high",
            "node_id": "junction::1",
            "geometry": {"x": 10, "y": 20},
            "message": "Tee junction node has degree below 3.",
        }
    ],
}
```

Expected:
- one review item
- `review_item_type == "topology"`
- `category == "tee_degree_mismatch"`
- `priority == 10`
- `status == "open"`
- geometry preserved

**Step 2: Run test and verify failure**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage13_review_package.py' -v
```

Expected: import failure because module does not exist.

**Step 3: Implement minimal builder**

Add:

```python
def build_stage13_review_package(*, image_id, graph_payload, stage12_qa_payload, stage12_review_queue_payload, stage11_line_number_review_payload=None):
    ...
```

Output shape:

```python
{
    "review_items_payload": {
        "image_id": image_id,
        "source": "stage13_review_package",
        "review_items": [...],
    },
    "summary": {
        "image_id": image_id,
        "review_item_count": N,
        "category_counts": {...},
        "severity_counts": {...},
        "priority_counts": {...},
    },
}
```

**Step 4: Verify test passes**

Run the same unittest command.

---

### Task 2: Normalize Stage 12 Review Queue Into Stage 13 Items

**Files:**
- Modify: `/Users/maetee/Code/GARNET/backend/garnet/stage13_review_package.py`
- Test: `/Users/maetee/Code/GARNET/backend/tests/test_stage13_review_package.py`

**Step 1: Write failing test**

Use Stage 12 review queue input:

```python
stage12_review_queue_payload = {
    "review_queue": [
        {
            "id": "review::line_number_conflict::component_00001",
            "issue_type": "line_number_conflict",
            "severity": "review",
            "message": "Connected trace component has multiple reviewed line numbers.",
            "candidate_line_number_ids": ["line_1", "line_2"],
            "component_edge_ids": ["trace::a", "trace::b"],
        }
    ]
}
```

Expected:
- category `line_number_conflict`
- review item type `line_number`
- priority higher than `line_number_inferred`
- evidence includes candidate line IDs and edge IDs

**Step 2: Implement category mapping**

Add mapping helpers:

```python
CATEGORY_TYPE = {
    "line_number_conflict": "line_number",
    "line_number_missing_after_propagation": "line_number",
    "unmerged_tee_terminal": "topology",
    "tee_degree_mismatch": "topology",
    "dead_end_not_expected": "topology",
    "duplicate_physical_path": "topology",
    "dead_end_trace": "trace_terminal",
    "duplicate_trace_collapsed": "info",
}
```

Priority rules:
- `tee_degree_mismatch`: 10
- `line_number_conflict`: 9
- `dead_end_not_expected`: 8
- `duplicate_physical_path`: 8
- `unmerged_tee_terminal`: 6
- `line_number_missing_after_propagation`: 6
- `dead_end_trace`: 5
- info items: 2

**Step 3: Deduplicate items by source id**

If the same source item appears in both QA and review queue, keep the higher priority item and merge evidence.

**Step 4: Verify tests pass**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage13_review_package.py' -v
```

---

### Task 3: Add Stage 13 Overlay Renderer

**Files:**
- Modify: `/Users/maetee/Code/GARNET/backend/garnet/stage13_review_package.py`
- Test: `/Users/maetee/Code/GARNET/backend/tests/test_stage13_review_package.py`

**Step 1: Write smoke test**

Create a small blank image and a payload with one review item with geometry. Call:

```python
render_stage13_review_overlay(image_bgr, review_items_payload)
```

Expected:
- returns image with same shape
- output differs from blank image

**Step 2: Implement overlay**

Use OpenCV if available. Draw:
- High priority red circle/label.
- Medium priority orange circle/label.
- Low/info cyan circle/label.
- Text label as `{priority}:{category}`.

**Step 3: Verify test passes**

Run Stage 13 tests.

---

### Task 4: Wire Stage 13 Into Pipeline

**Files:**
- Modify: `/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py`
- Test: existing Stage 13 tests plus py_compile

**Step 1: Update imports**

Import:

```python
from garnet.stage13_review_package import build_stage13_review_package, render_stage13_review_overlay
```

**Step 2: Replace geometric route `stage13_graph_qa` behavior**

Current `stage13_graph_qa` uses old `run_pipe_graph_qa_stage(...)` and writes `stage15_*`. Replace it for the geometric trace graph route with:

```python
graph_payload = self._load_json_artifact("stage12_graph")
qa_payload = self._load_json_artifact("stage12_graph_qa")
review_queue_payload = self._load_json_artifact("stage12_review_queue")
line_review_payload = self._load_json_artifact_or_default("stage11_line_number_review", {})
result = build_stage13_review_package(...)
self._save_json("stage13_review_items", result["review_items_payload"])
self._save_json("stage13_review_summary", result["summary"])
self._save_img("stage13_review_overlay", render_stage13_review_overlay(...))
```

Keep old pipe-graph QA behavior only if Stage 12 trace graph artifacts are missing.

**Step 3: Verify syntax**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py
```

---

### Task 5: Run Stage 13 On Test 01-09

**Files:**
- Generated under `/Users/maetee/Code/GARNET/backend/output_debug/Test-000??`

**Step 1: Run Stage 13 only from existing Stage 12 artifacts**

Use a small script to instantiate `PIDPipeline` for each `backend/output_debug/Test-000??` and call `stage13_graph_qa()`.

**Step 2: Verify artifacts exist**

For every test folder, verify:
- `stage13_review_items.json`
- `stage13_review_summary.json`
- `stage13_review_overlay.png`

**Step 3: Summarize review counts**

Print a table with:
- total review item count
- high priority count
- category counts
- line-number conflict count
- unmerged tee terminal count
- dead-end count

**Step 4: Final verification**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage13_review_package.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage11_line_number_hitl.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_trace_graph_builder.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_trace_graph_qa.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py
cd /Users/maetee/Code/GARNET && git diff --check
```
