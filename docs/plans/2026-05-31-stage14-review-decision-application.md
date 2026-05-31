# Stage 14 Review Decision Application Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Stage 14 to apply Stage 13 review decisions and emit a corrected, process-facing graph artifact.

**Architecture:** Stage 14 is non-destructive to `stage12_graph.json`. It reads Stage 12 graph, Stage 13 review items, and an optional decisions artifact, then writes a corrected graph plus correction audit records. The first implementation is an identity pass: if no real decisions exist, Stage 14 marks review items as accepted-by-assumption and outputs the graph unchanged, creating the stable handoff point for later HITL correction logic.

**Tech Stack:** Python, existing JSON pipeline artifacts, OpenCV overlay rendering, `unittest`.

---

### Task 1: Add Stage 14 Correction Module With Identity Pass

**Files:**
- Create: `/Users/maetee/Code/GARNET/backend/garnet/stage14_review_decisions.py`
- Create: `/Users/maetee/Code/GARNET/backend/tests/test_stage14_review_decisions.py`

**Step 1: Write the failing test**

Create `test_apply_stage14_review_decisions_identity_pass`:

```python
def test_apply_stage14_review_decisions_identity_pass(self) -> None:
    graph_payload = {
        "image_id": "synthetic.png",
        "nodes": [{"id": "n1", "type": "tee_junction", "position": {"x": 10, "y": 20}}],
        "edges": [{"id": "e1", "source": "n1", "target": "n2", "polyline": []}],
    }
    review_items_payload = {
        "image_id": "synthetic.png",
        "review_items": [
            {
                "id": "stage13::qa::tee_degree_mismatch::n1",
                "category": "tee_degree_mismatch",
                "priority": 10,
                "status": "open",
                "geometry": {"x": 10, "y": 20},
            }
        ],
    }

    result = apply_stage14_review_decisions(
        image_id="synthetic.png",
        graph_payload=graph_payload,
        review_items_payload=review_items_payload,
        decisions_payload={"decisions": []},
    )

    self.assertEqual(result["corrected_graph_payload"]["nodes"], graph_payload["nodes"])
    self.assertEqual(result["corrected_graph_payload"]["edges"], graph_payload["edges"])
    self.assertEqual(result["summary"]["correction_count"], 0)
    self.assertEqual(result["summary"]["assumed_resolved_count"], 1)
    self.assertEqual(result["review_resolution_payload"]["resolutions"][0]["resolution_state"], "accepted_by_assumption")
```

**Step 2: Run test and verify failure**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage14_review_decisions.py' -v
```

Expected: import failure because `garnet.stage14_review_decisions` does not exist.

**Step 3: Implement minimal module**

Add:

```python
def apply_stage14_review_decisions(*, image_id, graph_payload, review_items_payload, decisions_payload):
    corrected_graph_payload = copy.deepcopy(graph_payload)
    resolutions = []
    for item in review_items_payload.get("review_items", []):
        resolutions.append({
            "review_item_id": item.get("id"),
            "category": item.get("category"),
            "resolution_state": "accepted_by_assumption",
            "decision_source": "stage14_identity_pass",
            "graph_changed": False,
        })
    return {
        "corrected_graph_payload": corrected_graph_payload,
        "review_resolution_payload": {"image_id": image_id, "resolutions": resolutions},
        "correction_audit_payload": {"image_id": image_id, "corrections": []},
        "summary": {
            "image_id": image_id,
            "input_review_item_count": len(review_items_payload.get("review_items", [])),
            "decision_count": len(decisions_payload.get("decisions", [])),
            "correction_count": 0,
            "assumed_resolved_count": len(resolutions),
        },
    }
```

**Step 4: Run test and verify pass**

Run the same unittest command.

---

### Task 2: Add Decision Schema and Explicit No-Op Decisions

**Files:**
- Modify: `/Users/maetee/Code/GARNET/backend/garnet/stage14_review_decisions.py`
- Modify: `/Users/maetee/Code/GARNET/backend/tests/test_stage14_review_decisions.py`

**Step 1: Write failing test for explicit accepted decision**

Input decision:

```python
{
    "review_item_id": "stage13::qa::tee_degree_mismatch::n1",
    "decision": "accept_as_is",
    "reviewer": "human_assumed",
    "note": "Known valid junction geometry.",
}
```

Expected:
- resolution state is `accepted_as_is`
- `decision_source` is `human_assumed`
- no graph change
- summary `explicit_resolution_count == 1`

**Step 2: Implement decision indexing**

Build `decisions_by_review_item_id` from `decisions_payload["decisions"]`.

Supported initial decisions:
- `accept_as_is`: no graph mutation
- `false_positive`: no graph mutation yet, but marks item as dismissed
- `defer`: leaves resolution open

Unknown decisions should become `unsupported_decision` resolution entries and should not mutate the graph.

**Step 3: Run tests**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage14_review_decisions.py' -v
```

---

### Task 3: Add First Real Correction: Line Number Override

**Files:**
- Modify: `/Users/maetee/Code/GARNET/backend/garnet/stage14_review_decisions.py`
- Modify: `/Users/maetee/Code/GARNET/backend/tests/test_stage14_review_decisions.py`

**Step 1: Write failing test**

Decision:

```python
{
    "review_item_id": "stage13::review::line_number_conflict::component_00001",
    "decision": "set_line_number",
    "line_number_id": "line_123",
    "edge_ids": ["e1", "e2"],
    "reviewer": "human_assumed",
}
```

Graph has edges `e1`, `e2` with conflicting `effective_line_number_ids`.

Expected:
- corrected graph edges `e1`, `e2` have `effective_line_number_ids == ["line_123"]`
- each corrected edge gets `line_number_review_state == "human_reviewed"`
- correction audit has one entry with affected edges
- summary correction count is `1`

**Step 2: Implement edge lookup**

Create helper:

```python
def _edges_by_id(graph_payload):
    return {str(edge.get("id")): edge for edge in graph_payload.get("edges", [])}
```

**Step 3: Implement `set_line_number` decision**

For each `edge_id` in the decision:
- if edge exists, set `effective_line_number_ids`
- add `reviewed_line_number_id`
- add `line_number_review_state`
- record audit
- missing edges go to `warnings`

**Step 4: Run tests**

Run Stage 14 tests.

---

### Task 4: Add Correction Overlay Renderer

**Files:**
- Modify: `/Users/maetee/Code/GARNET/backend/garnet/stage14_review_decisions.py`
- Modify: `/Users/maetee/Code/GARNET/backend/tests/test_stage14_review_decisions.py`

**Step 1: Write overlay smoke test**

Use a blank image and a correction audit payload with one item containing geometry.

Expected:
- output image shape equals input
- output image sum is greater than zero

**Step 2: Implement renderer**

Add:

```python
def render_stage14_correction_overlay(image_bgr, corrected_graph_payload, correction_audit_payload, review_resolution_payload):
    ...
```

Draw:
- green marker for accepted/no-change reviewed items
- yellow marker for corrected items
- red marker for unsupported/deferred items
- label as `S14:{decision}` or `S14:assumed`

**Step 3: Run tests**

Run Stage 14 tests.

---

### Task 5: Wire Stage 14 Into Pipeline

**Files:**
- Modify: `/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py`

**Step 1: Add imports**

```python
from garnet.stage14_review_decisions import apply_stage14_review_decisions, render_stage14_correction_overlay
```

**Step 2: Replace or rename current Stage 14 method carefully**

Current `stage14_continuity_check` exists for legacy pipe graph continuity. Do not delete it.

Add a new method:

```python
def stage14_apply_review_decisions(self) -> None:
    graph_payload = self._load_json_artifact("stage12_graph")
    review_items_payload = self._load_json_artifact("stage13_review_items")
    decisions_payload = self._load_json_artifact_or_default("stage13_review_decisions", {"decisions": []})
    result = apply_stage14_review_decisions(...)
    self._save_json("stage14_corrected_graph", result["corrected_graph_payload"])
    self._save_json("stage14_review_resolutions", result["review_resolution_payload"])
    self._save_json("stage14_correction_audit", result["correction_audit_payload"])
    self._save_json("stage14_correction_summary", result["summary"])
    self._save_img("stage14_correction_overlay", render_stage14_correction_overlay(...))
```

**Step 3: Update geometric route stage definitions**

For `use_geometric_line_detection=True`, Stage 14 should call `stage14_apply_review_decisions`.

Keep legacy `stage14_continuity_check` for the older non-geometric route.

**Step 4: Run syntax check**

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py
```

---

### Task 6: Run Stage 14 on Test 01-09

**Files:**
- Generated only under `/Users/maetee/Code/GARNET/backend/output_debug/Test-0000X/`

**Step 1: Run Stage 14 only against existing artifacts**

Use a short script from `backend/` to instantiate `PIDPipeline` for each `output_debug/Test-00001` to `Test-00009`, then call the Stage 14 method.

Expected artifacts per test:
- `stage14_corrected_graph.json`
- `stage14_review_resolutions.json`
- `stage14_correction_audit.json`
- `stage14_correction_summary.json`
- `stage14_correction_overlay.png`

**Step 2: Summarize counts**

Report per image:
- input review item count
- explicit decision count
- assumed resolved count
- correction count
- warning count

**Step 3: Verify corrected graph exists and is valid JSON**

Run a small script loading all nine `stage14_corrected_graph.json` files.

---

### Task 7: Regression Verification

**Files:**
- No new files unless tests fail.

**Step 1: Run targeted tests**

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage14_review_decisions.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage13_review_package.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_trace_graph*.py' -v
```

**Step 2: Run backend syntax check**

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py
```

**Step 3: Run diff whitespace check**

```bash
cd /Users/maetee/Code/GARNET
git diff --check
```

---

## Output Contract

Stage 14 must emit:

```text
stage14_corrected_graph.json
stage14_review_resolutions.json
stage14_correction_audit.json
stage14_correction_summary.json
stage14_correction_overlay.png
```

`stage14_corrected_graph.json` becomes the graph source for later export and process analysis stages.

---

## Deferred Corrections

Do not implement these in the first pass unless explicitly requested:

- topology merge/split decisions
- manual tee relocation
- false branch deletion
- equipment-port remapping
- trace polyline editing
- graph export formatting

Those should be added after the identity pass and line-number override are stable.
