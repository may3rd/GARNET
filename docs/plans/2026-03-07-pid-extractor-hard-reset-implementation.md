# P&ID Extractor Hard Reset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `backend/garnet/pid_extractor.py` with a master-plan-only orchestrator that removes the legacy pipeline and runs the new 13-stage flow end to end.

**Architecture:** Keep the file as a single orchestrator for now, but make its structure mirror `MASTER_PLAN.md` exactly. Implement lightweight, test-backed stage contracts and artifact outputs first, then let later sprints deepen the stage internals.

**Tech Stack:** Python 3.14, NumPy, OpenCV, scikit-image, NetworkX, existing backend JSON/image inputs, `unittest`.

---

### Task 1: Lock the new stage contract in tests

**Files:**
- Modify: `backend/tests/test_pid_extractor_cli.py`

**Step 1: Write the failing test**
- Assert the runner exposes the 13 master-plan stage names in order.

**Step 2: Run test to verify it fails**
- Run: `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v`

**Step 3: Write minimal implementation**
- Add stage definitions and runner support for the new names.

**Step 4: Run test to verify it passes**
- Re-run the same command and confirm green.

### Task 2: Replace the legacy orchestrator

**Files:**
- Modify: `backend/garnet/pid_extractor.py`

**Step 1: Write the failing test**
- Add or adjust tests so they expect no legacy stage names and a valid master-plan manifest.

**Step 2: Run test to verify it fails**
- Re-run the targeted test suite.

**Step 3: Write minimal implementation**
- Replace the current file with:
  - shared data structures
  - stage manifest helpers
  - 13 stage methods aligned to the master plan
  - CLI with `--stop-after 1..13`

**Step 4: Run test to verify it passes**
- Re-run the targeted suite.

### Task 3: Make the sample run execute end to end

**Files:**
- Modify: `backend/garnet/pid_extractor.py`

**Step 1: Write the failing test**
- Assert the sample pipeline can run through the stage runner contract without legacy DeepLSD dependence.

**Step 2: Run test to verify it fails**
- Re-run targeted tests or the sample command as the red check.

**Step 3: Write minimal implementation**
- Implement lightweight stage internals for normalization, OCR/detection evidence loading, mask/seal/skeleton, node clustering, graph assembly, and QA outputs.

**Step 4: Run test to verify it passes**
- `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --coco coco_annotations.json --ocr ocr_results.json --arrow-coco coco_arrows.json --out output/master_plan_reset`

### Task 4: Verify and document the reset

**Files:**
- Modify: `IMPLEMENTATION_TRACKER.md`
- Modify: `SPRINT_0_PROGRESS.md` only if needed for continuity notes

**Step 1: Run compile verification**
- `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py`

**Step 2: Run regression tests**
- `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v`

**Step 3: Run the sample pipeline**
- `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --coco coco_annotations.json --ocr ocr_results.json --arrow-coco coco_arrows.json --out output/master_plan_reset`

**Step 4: Record actual status**
- Update tracker/docs with the new stage model and any follow-up gaps.
