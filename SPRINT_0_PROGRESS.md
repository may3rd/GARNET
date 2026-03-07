# Sprint 0 Progress Log

## Purpose
- Working log for Sprint 0 implementation progress.
- Keep this separate from [`IMPLEMENTATION_TRACKER.md`](/Users/maetee/Code/GARNET/IMPLEMENTATION_TRACKER.md), which remains the backlog and status board.

## Entry format
- Timestamp
- Task ID
- Action
- Evidence
- Verification
- Next step / blocker

## Entries

### 2026-03-07 20:03 ICT
- Task ID: `S0-01`
- Action: Reproduced the current baseline command path and verified the real Python runtime available in this workspace.
- Evidence:
  - `which python3` -> `/opt/homebrew/bin/python3`
  - repo venv interpreter exists at [`/Users/maetee/Code/GARNET/.venv/bin/python`](/Users/maetee/Code/GARNET/.venv/bin/python)
- Verification:
  - `cd backend && python3 -m py_compile api.py garnet/*.py garnet/utils/*.py` -> exit 0
  - `cd backend && python -m garnet.pid_extractor ...` -> failed immediately because `python` command is not present
- Next step / blocker:
  - Use the repo venv interpreter for Sprint 0 verification commands.

### 2026-03-07 20:05 ICT
- Task ID: `S0-01`
- Action: Ran the baseline pipeline with the repo venv to identify the first real runtime blocker.
- Evidence:
  - Partial outputs written under [`/Users/maetee/Code/GARNET/backend/output/baseline_s0`](/Users/maetee/Code/GARNET/backend/output/baseline_s0)
  - Completed stages:
    - `stage1_summary.json`
    - `stage2_*`
    - `stage3_*`
    - `stage4_*`
    - `stage5_skeleton_inverted.png`
- Verification:
  - `cd backend && ../.venv/bin/python -m garnet.pid_extractor --image sample.png --coco coco_annotations.json --ocr ocr_results.json --arrow-coco coco_arrows.json --out output/baseline_s0`
  - Result: failed in Stage 5 with `FileNotFoundError` for `DeepLSD/weights/deeplsd_md.tar`
- Next step / blocker:
  - Decide and implement the minimal Sprint 0 handling for missing DeepLSD weights so the baseline run is structured and traceable.

### 2026-03-07 20:06 ICT
- Task ID: `S0-03`
- Action: Confirmed structural cleanup target in the pipeline orchestrator.
- Evidence:
  - [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py) contains two `stage6_line_graph` definitions.
- Verification:
  - `rg -n '^\\s*def stage6_line_graph' backend/garnet/pid_extractor.py`
- Next step / blocker:
  - Add failing coverage before removing the duplicate definition.

### 2026-03-07 20:10 ICT
- Task ID: `S0-02`
- Action: Added stage-manifest support and a single staged `run()` path to the pipeline runner.
- Evidence:
  - [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py) now writes [`backend/output/baseline_s0/stage_manifest.json`](/Users/maetee/Code/GARNET/backend/output/baseline_s0/stage_manifest.json)
- Verification:
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v` -> 3 tests passed
- Next step / blocker:
  - Re-run the baseline command and confirm the manifest captures real stage outputs.

### 2026-03-07 20:10 ICT
- Task ID: `S0-03`
- Action: Removed the duplicate `stage6_line_graph` definition and moved CLI execution to a single `run(stop_after=...)` flow.
- Evidence:
  - `rg -n '^\\s*def stage6_line_graph' backend/garnet/pid_extractor.py` now reports one definition
- Verification:
  - targeted `unittest` coverage passed
  - `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py` -> exit 0
- Next step / blocker:
  - Re-run the baseline and confirm the practical Stage 5 fallback works in the real sample flow.

### 2026-03-07 20:10 ICT
- Task ID: `S0-01`
- Action: Re-ran the baseline pipeline with the repo venv after the Sprint 0 runner changes.
- Evidence:
  - [`backend/output/baseline_s0/stage_manifest.json`](/Users/maetee/Code/GARNET/backend/output/baseline_s0/stage_manifest.json)
  - [`backend/output/baseline_s0/stage5_stats.json`](/Users/maetee/Code/GARNET/backend/output/baseline_s0/stage5_stats.json)
- Verification:
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --coco coco_annotations.json --ocr ocr_results.json --arrow-coco coco_arrows.json --out output/baseline_s0` -> exit 0
- Next step / blocker:
  - DeepLSD weights are still absent, so the baseline completes with `deeplsd_status=missing_checkpoint` instead of full line extraction. This is now a recorded dependency rather than a crash.

### 2026-03-07 20:11 ICT
- Task ID: `S0-02`
- Action: Verified that rerunning the same baseline output directory still produces useful manifest artifact lists.
- Evidence:
  - [`backend/output/baseline_s0/stage_manifest.json`](/Users/maetee/Code/GARNET/backend/output/baseline_s0/stage_manifest.json) now lists per-stage artifact files for Stages 1 through 5
- Verification:
  - fresh rerun of the baseline command after artifact-tracking adjustment -> exit 0
- Next step / blocker:
  - Sprint 0 still has `S0-04` regression harness expansion and `S0-05` scorecard automation left open.
