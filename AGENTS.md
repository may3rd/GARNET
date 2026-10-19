# Agent instructions (scope: this directory and subdirectories)

# Lessons

- Run patch scripts from the repository root and stop the shell if patching fails before verification.

- When adding a review gate, update event-history assertions for later reopen and reconfirm paths.

- Use single-quoted Python patch markers for JSX attributes so triple-quote delimiters cannot terminate them.

- When orthogonal snapping a lone trace endpoint, constrain it to the existing segment axis; do not insert a bend.

- Run root-level patch scripts from the repository root; the frontend directory has no `.venv/`.
- Check live route registrations before trusting an API's older module description; `garnet/api.py` now serves trace jobs as well as detection.
- Do not assign a UI click failure to a confirmation dialog without reproducing the click path; verify the rendered event and API request first.
- The **active** P&ID extraction work is the top-level `garnet/` package, not `backend/garnet/`. Run it from the **repo root** with `.venv/bin/python`. `backend/garnet/` is the older pipeline behind the FastAPI service; it still runs, but new extraction work goes in `garnet/`.
- From `backend/`, use the root virtualenv as `../.venv/bin/python` or its absolute path, never `.venv/bin/python`.
- Inspect the on-disk target before choosing an exact patch marker; injected instructions may include content absent from the file.
- Run non-package tests with unittest discovery or file patterns; do not address `tests.*` modules when `tests/` has no `__init__.py`.

## Scope and layout

- This AGENTS.md applies to the repository root and everything below it.
- Keep root guidance global. Put current-extraction specifics in `garnet/AGENTS.md`, legacy-pipeline specifics in `backend/garnet/AGENTS.md`, Gemini adapter specifics in `backend/gemini_detector/AGENTS.md`.
- There are now **two P&ID extraction trees**. Decide which one a task belongs to before editing:
  - `garnet/` — current work. Fixture-driven, backend-free, run from the repo root.
  - `backend/garnet/` — the pipeline the API serves (`--ai-equipment` / `--ai-line-numbers` flags, stage manifests, HITL review state).

## Modules / subprojects

| Module           | Type             | Path                       | What it owns                                                                     | How to run                                                                                | Tests / checks                                              | AGENTS                              |
| ---------------- | ---------------- | -------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------- |
| P&ID extraction  | python package   | `garnet/`                  | **Current** detection + LLM extraction + pipe tracing + Stage 6 associations      | From **repo root**: `.venv/bin/python -m garnet.stage5b.run --stem <S> --indir garnet/tests/input --out output/<S>` | `.venv/bin/python -m unittest discover -s garnet/tests -p "test_*.py"` | `garnet/AGENTS.md`                  |
| Backend API      | fastapi          | `backend/`                 | HTTP API, file upload flow, result serving, runtime config                       | From `backend/`: `uvicorn api:app --reload --port 8090`                                   | `python -m py_compile api.py garnet/*.py garnet/utils/*.py` | `backend/garnet/AGENTS.md`          |
| Legacy pipeline  | python package   | `backend/garnet/`          | Stage-by-stage P&ID rebuild behind the API, shared OCR/image utilities           | From `backend/`: `../.venv/bin/python -m garnet.pid_extractor`                            | See module AGENTS                                           | `backend/garnet/AGENTS.md`          |
| Gemini detector  | python adapter   | `backend/gemini_detector/` | Gemini/OpenRouter SAHI detector for text-like classes                            | Called from backend code                                                                  | `python -m py_compile gemini_detector/*.py`                 | `backend/gemini_detector/AGENTS.md` |
| Frontend         | react + vite     | `frontend/`                | Review UI, canvas editing, exports, backend API client                           | From `frontend/`: `bun run dev`                                                           | `bun run build`, `bun run lint`                             | none                                |
| Design assets    | docs/assets      | `design/`                  | Design references and non-runtime materials                                      | Open only when task requires design context                                               | n/a                                                         | none                                |

## The `garnet/` extraction flow

Three inputs per sheet, from three different sources. They are **not** interchangeable:

| Input                      | Source                                                                                           | Fixture filename                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| Symbols / instruments      | YOLO + SAHI — `garnet/detect_objects.py` (weight in `backend/yolo_weights/`)                      | `<stem>_objects.json`                                          |
| Equipment boxes + nozzles  | **Vision LLM** — `garnet/detect_equipment_bboxes.py` (ollama-cloud, creds in `~/.hermes/.env`)     | `<stem>_equipment_bboxes.json` or `<stem>_equipment.json`      |
| Line-number text           | **Vision LLM via the Claude skill/agent** (`garnet-line-numbers` / `garnet-line-number-extractor`) | `<stem>_line_number_boxes.json` or `<stem>_line_numbers.json`  |

- YOLO detects neither equipment nor text. Both of those are LLM work now — that is the reason equipment and line numbers arrive as separate JSON files rather than out of detection.
- **Equipment has an in-repo script; line numbers do not.** There is no `garnet/detect_line_numbers.py`. Line numbers come from the skill/agent driven by `backend/garnet/OCR_prompts/pid_line_number_extraction_prompt.md`; the equipment prompt is `backend/garnet/OCR_prompts/pid_equipment_bbox_prompt.md`.
- **Consumption differs between the two trees.** `garnet/stage5b/host.py:load_fixtures` picks the JSON up **by filename convention** from `--indir` (both spellings above are accepted). The backend pipeline instead takes explicit `--ai-equipment` / `--ai-line-numbers` paths. For `garnet/` work, drop the files in the fixture dir under one of the accepted names; `.claude/skills/garnet-digitize/reference/contracts.md` lists them.
- `Equipment_type` must land in `EQUIPMENT_LABELS` after the synonym map in `_equipment_as_objects`, or the item is **silently dropped** and the sheet traces nothing from it while still exiting 0. Check the equipment count in the run log.
- Frame must match the raster: the LLM JSON's pixel frame is the sheet's own. Extract from the exact image you trace.

Stage 5c (`garnet/stage5b/merge_branches.py`) is not a separate CLI — `stage5b_pipeline.py` calls it so a branch line ends at a real terminate point rather than the next junction dot. Stage 2 OCR is opt-in (`run --ocr`).

Other entrypoints: `garnet/trace_pipes.py` (self-contained tracer — shares no code with `stage5b/`, so fixes there do not reach it), `garnet/run_stage5b.py` (**backend route**: imports `garnet.pid_extractor`, must run from `backend/`), `garnet/stage5b/coverage_audit.py` (run after a tracing change — walk counts can look identical while whole branches go unvisited), `garnet/draw_boxes.py` (box-only overlays), `garnet/stage5b/review_gate.py` (cleans up the traced network), `garnet/stage5b/run_associations.py` (Stage 6), `garnet/stage5b/inspect_line_numbers.py` (visual review of Stage 6 assignment). `bash garnet/run_test.sh` clears `output/` and re-runs every fixture sheet.

- **Two things in `garnet/` make network calls to a vision model**: `detect_equipment_bboxes.py` (always) and `stage5b/review_gate.py` (on the R1 residual only — `--no-llm` for deterministic rules, `--dry-run` to inspect without calling). Everything else is local CV.

- **Do not resync `garnet/stage5b/` to `backend/garnet/path_tracer/`.** The copy has deliberately diverged and carries fixes that exist only here; `garnet/stage5b/__init__.py` lists them. Port fixes *out* of `garnet/` if the backend needs them, never *in*.

## Cross-domain workflows

- Frontend <-> backend: the Vite app proxies `/api` and `/runs` to `VITE_API_URL`, defaulting to `http://localhost:8090`. Keep backend route changes synchronized with frontend API usage.
- API <-> legacy pipeline: `backend/api.py` is the service entrypoint, but its extraction logic lives in `backend/garnet/`. Change request/response shapes in the API layer only after checking downstream pipeline output and frontend expectations. `garnet/` is not wired to the API.
- Legacy stage sequence (sparse, still the backend runner's order): `1, 2, 4, 5, 5b, 6, 7, 7c, 7b, 8, 9, 10, 11`. Stage 3 is external HITL input.
- Generated artifacts: `output/` (root, `garnet/` runs) and the backend artifact folders stay out of commits. Do not make the frontend depend on developer-local filesystem paths.
- Pipeline HITL review: the frontend has separate review gates for object boxes and traced paths. Keep class suggestions tied to the active review type, preserve free-text class entry for add-box workflows, and make trace/branch overlays selectable without blocking normal canvas pan/zoom.

## Verification (preferred commands)

- Default rule: run checks from the owning module, keep them narrow first, and widen only after the touched area is stable.
- `garnet/`: from the repo root, `.venv/bin/python -m py_compile garnet/*.py garnet/stage5b/*.py`, then the unit tests, then `bash garnet/run_test.sh` for a full fixture sweep when tracing behaviour changed.
- Backend: from `backend/`, `python -m py_compile api.py garnet/*.py garnet/utils/*.py`; targeted runtime checks only when the weights/deps are present.
- Frontend: from `frontend/`, `bun run lint` and `bun run build`.
- Root changes: verify the specific files you added or edited, and confirm scoped AGENTS files do not conflict.

## Docs usage

- Do not open or edit `design/` unless the task needs UI/design context.
- Prefer `README.md`, `docs/`, and the nearest scoped `AGENTS.md` before reading deeper project material.
- `MASTER_PLAN.md` and `punch_list.md` were removed; do not cite them.

## Global conventions

- Route work to the nearest scoped AGENTS file before making module-specific changes.
- Prefer `bun` over `npm` for frontend install, dev, build, and lint commands unless the user explicitly asks otherwise.
- Keep generated outputs, caches, model weights, and secrets out of commits unless the user explicitly asks for them. LLM credentials live in `~/.hermes/.env` / `.env`, never in code.
- Preserve existing module boundaries. If a task crosses backend, frontend, and extraction code, describe the touch points clearly and verify each touched module separately.
- When a new submodule develops its own conventions or risk profile, add a nested `AGENTS.md` there instead of bloating the root file.
- For frontend canvas changes, verify both visual alignment and pointer interaction. Trace/path overlays should use image-coordinate alignment, wide hit targets, and pass-through empty areas so object editing and pan/zoom still work.
