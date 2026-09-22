# Agent instructions (scope: `garnet/` and below)

Status as of 2026-09-22. This is the **active** P&ID extraction tree. `backend/garnet/` is the
older pipeline the FastAPI service serves; it still runs, but new extraction work goes here.

## Run it

From the **repo root**, with the root venv:

```bash
.venv/bin/python -m garnet.stage5b.run --stem 14780-8120-25-25-0002A --indir garnet/tests/input --out output/14780-8120-25-25-0002A
.venv/bin/python -m garnet.stage5b.run_associations --stem 14780-8120-25-25-0002A
```

`bash garnet/run_test.sh` clears `output/` and re-runs all nine fixture
sheets. Checks: `.venv/bin/python -m py_compile garnet/*.py garnet/stage5b/*.py`, then
`.venv/bin/python -m unittest discover -s garnet/tests -p "test_*.py"`.

## What is here

| File | Status |
| --- | --- |
| `stage5b/` | The live pipeline: mask → trace → 5c merge → review gate → Stage 6. See its `__init__.py` docstring for the module map and the divergence record. |
| `detect_objects.py` | YOLO + SAHI symbols → objects JSON. Standalone, no pipeline. |
| `detect_equipment_bboxes.py` | Vision LLM (ollama-cloud, creds in `~/.hermes/.env`) → Contract B equipment + nozzles. YOLO detects no equipment; this is the only source. |
| `trace_pipes.py` | Separate self-contained tracer — borrows the walking algorithm only, shares no code with `stage5b/`. Fixes made in `stage5b/` do **not** reach it. |
| `draw_boxes.py` | Box-only overlays for checking a fixture JSON against its raster. |
| `run_stage5b.py` | **Backend route, not this tree.** It imports `garnet.pid_extractor` / `garnet.ai_import`, i.e. `backend/garnet/`, so it must run from `backend/` and it drives the backend tracer copy, which is behind. |
| `tests/input/` | Nine sheet fixtures (`14780-8120-25-25-*`) — raster + objects + equipment + line numbers. |
| `tests/` | `test_detect_objects.py`, `test_stage5b_dashed_start.py`, `test_stage5b_occupancy.py`. |

There is **no line-number extraction script.** Equipment has one; line numbers come from the
`garnet-line-numbers` skill / `garnet-line-number-extractor` agent, driven by
`backend/garnet/OCR_prompts/pid_line_number_extraction_prompt.md`.

`garnet/` has no `__init__.py` — it is a namespace package. Only `garnet/stage5b/` is a real
package.

## Fixture conventions

`stage5b/host.py:load_fixtures` picks inputs up **by filename** from `--indir`:

- `<stem>.jpg`
- `<stem>_objects.json`
- `<stem>_equipment_bboxes.json` **or** `<stem>_equipment.json`
- `<stem>_line_number_boxes.json` **or** `<stem>_line_numbers.json`

The first spelling wins when both exist (sheet 0007 has both equipment files); the fixture dir
uses a mix. Only `<stem>_objects.json` is required — a missing equipment or line-number file is
not an error, the sheet just runs without that evidence.

The backend pipeline instead takes explicit `--ai-equipment` / `--ai-line-numbers` paths.
`.claude/skills/garnet-digitize/reference/contracts.md` documents both routes side by side.

`Equipment_type` must land in `EQUIPMENT_LABELS` after the synonym map in
`_equipment_as_objects`, or the item is **silently dropped**: the sheet traces nothing from it
and still exits 0. Check the equipment count in the run log. The LLM JSON's pixel frame is the
sheet's own — extract from the exact raster you trace.

## Rules

- **Do not resync `garnet/stage5b/` to `backend/garnet/path_tracer/`.** The copy has diverged
  deliberately and by now substantially (~1000 diff lines in each of `cv_pipe_tracer.py` and
  `stage5b_pipeline.py`). Port fixes *out* of `garnet/`, never *in*. `stage5b/__init__.py` lists
  what is only here; `git hash-object` both copies before claiming parity.
- Stage 2 OCR is opt-in (`run --ocr`), off by default.
- `coverage_audit.py` after a tracing change: walk counts and terminal histograms can be
  identical while whole pipe branches go unvisited.
- Never fabricate a line-number attachment. Out-of-range labels become `needs_review`; traces
  without a label go to `traces_without_line_number`.
- Outputs go to `output/` at the repo root. Keep them out of commits, along with weights and
  credentials.
