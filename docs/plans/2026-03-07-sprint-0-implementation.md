# Sprint 0 Baseline Stabilization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stabilize the current P&ID pipeline enough to produce a repeatable baseline run, a stage manifest, and a safer staged CLI.

**Architecture:** Keep the current staged runner in place and add small, test-backed changes around it. Capture execution metadata in a manifest, document baseline evidence in a separate progress log, and narrow code changes to the CLI/stage runner instead of refactoring the full pipeline.

**Tech Stack:** Python 3.14, existing repo venv, FastAPI backend package, OpenCV/NumPy/torch/NetworkX pipeline code, lightweight `unittest` or script-based regression coverage.

---

### Task 1: Document the baseline and progress channel

**Files:**
- Create: `docs/plans/2026-03-07-sprint-0-design.md`
- Create: `docs/plans/2026-03-07-sprint-0-implementation.md`
- Create: `SPRINT_0_PROGRESS.md`

**Step 1: Record the approved design**
- Summarize the approved minimal stabilization approach and the baseline findings.

**Step 2: Record the execution plan**
- Break Sprint 0 into `S0-01` through `S0-03` with explicit verification.

**Step 3: Start the progress log**
- Add the first entries for baseline reproduction and environment findings.

### Task 2: Add failing coverage for stage runner behavior

**Files:**
- Modify: `backend/garnet/pid_extractor.py`
- Test: `backend/tests/test_pid_extractor_cli.py`

**Step 1: Write failing tests**
- Add coverage for:
  - a manifest file being written for staged runs
  - `--stop-after` stopping after the requested stage
  - the stage runner surfacing stage failure details in the manifest

**Step 2: Run the tests to verify failure**
- Run targeted tests with the repo venv and capture the failing output.

**Step 3: Implement the minimal stage-runner changes**
- Add a small stage execution wrapper and manifest writer.

**Step 4: Run the tests to verify they pass**
- Re-run the targeted tests until green.

### Task 3: Remove duplicated Stage 6 and align CLI semantics

**Files:**
- Modify: `backend/garnet/pid_extractor.py`
- Test: `backend/tests/test_pid_extractor_cli.py`

**Step 1: Write failing regression coverage**
- Assert the CLI stage count and stop-after behavior match the actual exposed stages.

**Step 2: Remove the duplicate implementation**
- Keep a single `stage6_line_graph` path and one consistent call chain.

**Step 3: Verify compile and targeted tests**
- Run compile and targeted tests after the cleanup.

### Task 4: Make the baseline command practical in this workspace

**Files:**
- Modify: `backend/garnet/pid_extractor.py`
- Modify: `SPRINT_0_PROGRESS.md`
- Modify: `IMPLEMENTATION_TRACKER.md`

**Step 1: Decide the minimal handling for missing DeepLSD weights**
- Prefer explicit manifest reporting and a reproducible partial baseline over a hard crash with no structured evidence.

**Step 2: Implement the narrowest viable behavior**
- Record missing optional asset state clearly and keep output artifacts useful.

**Step 3: Run the baseline command**
- Use the repo venv and write output to `backend/output/baseline_s0`.

**Step 4: Update progress and tracker status**
- Record evidence paths, verification commands, and remaining blockers.

### Task 5: Verify Sprint 0 partial delivery

**Files:**
- Modify: `SPRINT_0_PROGRESS.md`
- Modify: `IMPLEMENTATION_TRACKER.md`

**Step 1: Run compile verification**
- `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py`

**Step 2: Run targeted regression tests**
- `cd backend && ../.venv/bin/python -m unittest ...`

**Step 3: Run the baseline pipeline command**
- `cd backend && ../.venv/bin/python -m garnet.pid_extractor --image sample.png --coco coco_annotations.json --ocr ocr_results.json --arrow-coco coco_arrows.json --out output/baseline_s0`

**Step 4: Record actual status honestly**
- Mark `S0-01` through `S0-03` as `DONE`, `DOING`, or `BLOCKED` with evidence.
