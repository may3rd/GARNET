# Stage 11 Line Number HITL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Treat Stage 11 line-number associations as human-reviewed truth, then use Stage 12 topology to propagate reviewed line numbers while preserving review evidence.

**Architecture:** Stage 11 will mark accepted line-number associations with review metadata (`review_state=accepted`, `review_source=human_assumed`) and emit a review artifact that can later be edited by a human UI. Stage 12 will consume only reviewed/direct line-number evidence for topology propagation and will keep direct, inferred, missing, and conflict states separate on graph edges. Stage 12 QA will use effective line-number IDs but still expose unresolved/conflict cases for HITL.

**Tech Stack:** Python, existing JSON stage artifacts, `unittest`, current Stage 11/12 pipeline outputs.

---

### Task 1: Add Stage 11 Review Metadata For Line Numbers

**Files:**
- Modify: `/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py`
- Test: `/Users/maetee/Code/GARNET/backend/tests/test_stage11_line_number_hitl.py`

**Step 1: Write failing test**

Create a test for a small accepted line-number association dict. The test should call a new helper function and verify that accepted line-number associations get:

```python
{
    "review_state": "accepted",
    "review_source": "human_assumed",
    "review_required": False,
}
```

Rejected line numbers should get:

```python
{
    "review_state": "needs_review",
    "review_source": "system",
    "review_required": True,
}
```

**Step 2: Run test and verify it fails**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage11_line_number_hitl.py' -v
```

Expected: FAIL because helper does not exist.

**Step 3: Implement helper**

Add a small module-level helper in `pid_extractor.py`:

```python
def _mark_line_number_review_state(association: dict[str, Any], *, accepted: bool) -> dict[str, Any]:
    result = dict(association)
    if accepted:
        result.update({
            "review_state": "accepted",
            "review_source": "human_assumed",
            "review_required": False,
        })
    else:
        result.update({
            "review_state": "needs_review",
            "review_source": "system",
            "review_required": True,
        })
    return result
```

**Step 4: Apply helper in `stage11_trace_associations`**

After `_trace_assoc_attach_bbox_items(...)` returns accepted/rejected line numbers:

```python
accepted = [_mark_line_number_review_state(item, accepted=True) for item in accepted]
rejected = [_mark_line_number_review_state(item, accepted=False) for item in rejected]
```

Important: update the attached edge entries too, not only the `associations` payload. Easiest safe method: mark before `_trace_assoc_add` by adding optional review metadata support inside `_trace_assoc_attach_bbox_items` for `group == "line_numbers"`.

**Step 5: Verify**

Run the new test and syntax check:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage11_line_number_hitl.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py
```

---

### Task 2: Emit Stage 11 HITL Review Artifact

**Files:**
- Modify: `/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py`
- Test: `/Users/maetee/Code/GARNET/backend/tests/test_stage11_line_number_hitl.py`

**Step 1: Write failing test**

Test a helper that builds a Stage 11 line-number review payload:

```python
payload = build_stage11_line_number_review_payload(
    image_id="synthetic.png",
    accepted=[{"id": "ln1", "trace_id": "trace_a", "review_state": "accepted"}],
    rejected=[{"id": "ln2", "reason": "distance_over_threshold", "review_state": "needs_review"}],
    traces_without_line_number=["trace_b"],
)
```

Expected output shape:

```python
{
    "image_id": "synthetic.png",
    "review_assumption": "accepted_line_numbers_are_human_reviewed",
    "accepted": [...],
    "needs_review": [...],
    "traces_without_line_number": ["trace_b"],
}
```

**Step 2: Implement helper**

Add module-level helper:

```python
def build_stage11_line_number_review_payload(...):
    return {...}
```

**Step 3: Save artifacts in `stage11_trace_associations`**

Save:

```python
self._save_json("stage11_line_number_review", review_payload)
self._save_json("stage11_line_number_review_summary", review_summary)
```

Summary fields:

```python
{
    "image_id": image_id,
    "accepted_count": len(accepted),
    "needs_review_count": len(rejected),
    "trace_without_line_number_count": len(traces_without_line_number),
    "review_assumption": "accepted_line_numbers_are_human_reviewed",
}
```

**Step 4: Verify**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage11_line_number_hitl.py' -v
```

---

### Task 3: Propagate Reviewed Line Numbers In Stage 12 Graph

**Files:**
- Modify: `/Users/maetee/Code/GARNET/backend/garnet/trace_graph_builder.py`
- Test: `/Users/maetee/Code/GARNET/backend/tests/test_trace_graph_builder.py`

**Step 1: Write failing tests**

Add tests for:

1. A connected component with one direct reviewed line number propagates to all edges.
2. A connected component with two distinct reviewed line numbers marks all affected edges as `conflict`.
3. A component with no reviewed line number remains `missing`.

Expected edge fields:

```python
"direct_line_number_ids": [...]
"inferred_line_number_ids": [...]
"effective_line_number_ids": [...]
"line_number_assignment_state": "direct" | "inferred" | "missing" | "conflict"
```

**Step 2: Run tests and verify failure**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_trace_graph_builder.py' -v
```

**Step 3: Implement propagation helper**

Add helper in `trace_graph_builder.py` after graph edges are built:

```python
def _apply_line_number_component_propagation(nodes, edges):
    ...
```

Rules:
- Build connected components from graph edges.
- Direct IDs come from existing `line_number_ids` only when the source association is reviewed/accepted.
- If a component has exactly one reviewed line number, add it to every edge as `effective_line_number_ids`.
- If an edge did not have it directly, put it in `inferred_line_number_ids` and state `inferred`.
- If component has multiple reviewed line numbers, state `conflict` and keep candidates in `effective_line_number_ids`.
- If none, state `missing`.

**Step 4: Add review items**

Add Stage 12 review items:
- `line_number_inferred` severity `info`
- `line_number_missing_after_propagation` severity `review`
- `line_number_conflict` severity `review`

Do not remove the original direct `missing_line_number` item until the new summary is stable.

**Step 5: Verify tests**

Run builder tests.

---

### Task 4: Update Stage 12 QA To Use Effective Line Numbers

**Files:**
- Modify: `/Users/maetee/Code/GARNET/backend/garnet/trace_graph_qa.py`
- Test: `/Users/maetee/Code/GARNET/backend/tests/test_trace_graph_qa.py`

**Step 1: Write failing test**

Create a graph component where edges have no direct `line_number_ids` but have `effective_line_number_ids`. QA should not emit `missing_line_number_component`.

**Step 2: Implement helper**

In QA, replace direct line-number lookup:

```python
line_ids = {str(line_id) for edge in edges for line_id in (edge.get("line_number_ids") or [])}
```

with helper:

```python
def _edge_effective_line_number_ids(edge):
    return edge.get("effective_line_number_ids") or edge.get("line_number_ids") or []
```

**Step 3: Verify**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_trace_graph_qa.py' -v
```

---

### Task 5: Run Real Stage 11/12 On Test Images And Compare

**Files:**
- Generated only under `/Users/maetee/Code/GARNET/backend/output_debug/Test-000??`

**Step 1: Rerun Stage 11 and 12 on Test 01-09**

Use existing stage artifacts and rerun Stage 11/12 only from each output folder.

**Step 2: Compare summaries**

Compare:
- `stage11_trace_association_summary.json`
- `stage11_line_number_review_summary.json`
- `stage12_graph_summary.json`
- `stage12_graph_qa_summary.json`

Expected:
- Stage 11 accepted line-number count unchanged.
- New Stage 11 review artifacts exist.
- Stage 12 `missing_line_number_component` decreases when connected components have one reviewed line number.
- Conflict cases are explicit and not silently resolved.

**Step 3: Final verification**

Run:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage11_line_number_hitl.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_trace_graph_builder.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_trace_graph_qa.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py
cd /Users/maetee/Code/GARNET && git diff --check
```
