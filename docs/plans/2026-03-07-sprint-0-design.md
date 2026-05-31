# Sprint 0 Baseline Stabilization Design

## Context
- Scope: Sprint 0 from [`IMPLEMENTATION_TRACKER.md`](/Users/maetee/Code/GARNET/IMPLEMENTATION_TRACKER.md)
- Target module: [`backend/garnet/`](/Users/maetee/Code/GARNET/backend/garnet)
- Goal: make the current staged pipeline reproducible enough to measure progress before Phase A/B feature work.

## Approved approach
- Use the existing staged orchestrator in [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py) as the canonical runner.
- Add thin stabilization around it instead of refactoring the full architecture now.
- Keep Sprint 0 limited to:
  - canonical baseline command and output folder
  - machine-readable stage manifest
  - cleanup of duplicated Stage 6 logic and `--stop-after` semantics
  - separate Sprint 0 progress log

## Findings from baseline reproduction
- The workspace does not expose `python`; runtime commands need `python3` or the repo venv interpreter.
- The repo venv at [`.venv/bin/python`](/Users/maetee/Code/GARNET/.venv/bin/python) has the required Python packages, including `torch`.
- The current full baseline run reaches Stage 4 and then fails at Stage 5 because the old DeepLSD-backed line extraction step expects `DeepLSD/weights/deeplsd_md.tar`, which is missing in this workspace.
- [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py) contains duplicated `stage6_line_graph` definitions, which is a real maintenance risk and a Sprint 0 target.

## Design decisions
- Canonical runtime for Sprint 0 verification will use `../.venv/bin/python` from `backend/`.
- Stage execution will write a manifest that records:
  - stage index and stage name
  - start/end timestamps
  - duration
  - status (`started`, `completed`, `failed`, `skipped`)
  - known artifact filenames
  - error message when a stage fails
- Stage 5 should degrade clearly when optional DeepLSD weights are missing:
  - do not silently hide the problem
  - record the failure/skipped dependency in the manifest
  - keep the baseline reproducible enough to compare pre-graph artifacts
- `--stop-after` should map cleanly to the actual numbered stages exposed by the CLI.

## Out of scope
- Re-architecting the pipeline to fully match the master plan
- Implementing the full regression harness and scorecard automation
- Changing model behavior or geometry semantics

## Acceptance criteria
- A documented baseline run can be reproduced with the repo venv.
- A stage manifest is written into the baseline output folder.
- There is only one active `stage6_line_graph` implementation.
- `--stop-after` behavior is deterministic and documented by tests or harness checks.
