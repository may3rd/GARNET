# P&ID Extractor Hard Reset Design

## Context
- File in scope: [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
- Source of truth: [`MASTER_PLAN.md`](/Users/maetee/Code/GARNET/MASTER_PLAN.md)
- User decision: hard reset the orchestrator and keep only the master-plan approach.

## Problem
- The current file mixes old graph-building logic, DeepLSD-specific routing, symbol-port repair heuristics, and stage names that do not match the master plan.
- The file is too large and internally inconsistent to evolve safely.
- Carrying the old and new ideas together would make every later sprint slower and more error-prone.

## Chosen approach
- Replace the file with a new orchestrator whose public execution model is the 13 master-plan stages only.
- Keep only shared models, generic persistence helpers, CLI parsing, and stage-manifest support that still fit the new architecture.
- Remove legacy pipeline-specific logic entirely instead of wrapping or aliasing it.

## New execution model
- Stage 1: input normalization and scale pyramid
- Stage 2: full-page OCR discovery
- Stage 3: crop OCR refinement
- Stage 4: small-object, flow-arrow, and equipment detection
- Stage 5: pipe mask generation
- Stage 6: morphological sealing
- Stage 7: skeleton generation
- Stage 8: skeleton node detection and clustering
- Stage 9: crossing/junction disambiguation
- Stage 10: skeleton cleanup and edge tracing
- Stage 11: attachment and association
- Stage 12: graph assembly and simplification
- Stage 13: QA and recovery loop

## Design rules
- Geometry-first structure must be visible in code order and artifact names.
- Each stage writes inspectable outputs or metrics JSON.
- No stage depends on optional DeepLSD weights.
- Semantic association happens after geometry artifacts exist.
- Graph QA is explicit and machine-readable.

## First reset scope
- Build a clean orchestrator and data flow.
- Implement minimal but valid behavior for all 13 stages so the pipeline runs end to end on the sample assets.
- Preserve stage manifest support and the CLI entrypoint.
- Update tests to assert the new stage contract.

## Out of scope for this reset
- Full production-grade attachment scoring
- Final DEXPI parity
- Recovery-loop sophistication beyond queue generation
- DeepLSD-based line extraction

## Acceptance criteria
- `pid_extractor.py` contains only the master-plan stage structure.
- Old stage names like `stage4_linework`, `stage5_graph`, and `stage6_line_graph` are gone.
- The sample run completes with the new 13-stage manifest.
- Regression tests validate the new stage runner contract.

## Current live implementation note
- The master-plan target keeps crossing/junction disambiguation logically ahead of final edge tracing.
- The current live rebuild keeps public stage numbering stable, so the explicit crossing-resolution prepass is implemented inside the existing Stage 10 flow instead of introducing a new public stage number.
- Current Stage 10 therefore writes both:
  - `stage10_crossing_resolution.json`
  - `stage10_pipe_edges.json`
- This is an implementation constraint for the active rebuild, not a contradiction of the roadmap.
