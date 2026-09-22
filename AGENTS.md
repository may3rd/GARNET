# Agent instructions (scope: this directory and subdirectories)

# Lessons

- From `backend/`, use the root virtualenv as `../.venv/bin/python` or its absolute path, never `.venv/bin/python`.
- Inspect the on-disk target before choosing an exact patch marker; injected instructions may include content absent from the file.
- Run non-package tests with unittest discovery or file patterns; do not address `tests.*` modules when `tests/` has no `__init__.py`.

## Scope and layout

- This AGENTS.md applies to the repository root and everything below it.
- Keep root guidance global. Put Python pipeline specifics in `backend/garnet/AGENTS.md` and Gemini adapter specifics in `backend/gemini_detector/AGENTS.md`.
- Use `MASTER_PLAN.md` as the architecture roadmap for P&ID digitizing work. Do not restate or contradict it in ad hoc task notes.

## Modules / subprojects

| Module          | Type             | Path                       | What it owns                                                                    | How to run                                              | Tests / checks                                              | Docs                | AGENTS                              |
| --------------- | ---------------- | -------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------- | ------------------- | ----------------------------------- |
| Backend API     | fastapi          | `backend/`                 | HTTP API, file upload flow, result serving, runtime config                      | From `backend/`: `uvicorn api:app --reload --port 8090` | `python -m py_compile api.py garnet/*.py garnet/utils/*.py` | `README.md`         | `backend/garnet/AGENTS.md`          |
| P&ID pipeline   | python package   | `backend/garnet/`          | Stage-by-stage P&ID rebuild, shared OCR/image utilities, pipeline orchestration | From `backend/`: `python -m garnet.pid_extractor`       | See module AGENTS                                           | `MASTER_PLAN.md`    | `backend/garnet/AGENTS.md`          |
| Gemini detector | python adapter   | `backend/gemini_detector/` | Gemini/OpenRouter SAHI detector for text-like classes                           | Called from backend code                                | `python -m py_compile gemini_detector/*.py`                 | module file         | `backend/gemini_detector/AGENTS.md` |
| Frontend        | react + vite     | `frontend/`                | Review UI, canvas editing, exports, backend API client                          | From `frontend/`: `bun run dev`                         | `bun run build`, `bun run lint`                             | `README.md`         | none                                |
| DeepLSD         | vendored library | `DeepLSD/`                 | Line-detection experiments and model support code                               | Follow local README / requirements                      | Module-specific                                             | `DeepLSD/README.md` | none                                |
| Design assets   | docs/assets      | `design/`                  | Design references and non-runtime materials                                     | Open only when task requires design context             | n/a                                                         | `design/README.md`  | none                                |

## Cross-domain workflows

- Frontend <-> backend: the Vite app proxies `/api` and `/runs` to `VITE_API_URL`, defaulting to `http://localhost:8090`. Keep backend route changes synchronized with frontend API usage.
- API <-> pipeline: `backend/api.py` is the service entrypoint, but the extraction logic lives in `backend/garnet/`. Change request/response shapes in the API layer only after checking the downstream pipeline output and frontend expectations.
- Pipeline roadmap: for P&ID digitizing features, preserve the stage model in `MASTER_PLAN.md` and the scoped rules in `backend/garnet/AGENTS.md`. The live runner is the sparse sequence documented in `MASTER_PLAN.md` (`1, 2, 4, 5, 5b, 6, 7, 7c, 7b, 8, 9, 10, 11`); Stage 3 remains external HITL input.
- Generated artifacts: keep predictions, runs, temp files, and debug outputs in backend-owned artifact folders. Do not make the frontend depend on developer-local filesystem paths.
- Pipeline HITL review: the frontend has separate review gates for object boxes and traced paths. Keep class suggestions tied to the active review type, preserve free-text class entry for add-box workflows, and make trace/branch overlays selectable without blocking normal canvas pan/zoom.

## Verification (preferred commands)

- Default rule: run checks from the owning module, keep them narrow first, and widen only after the touched area is stable.
- Backend: from `backend/`, use `python -m py_compile api.py garnet/*.py garnet/utils/*.py`; run targeted runtime checks only when the relevant weights/dependencies are present.
- Frontend: from `frontend/`, use `bun run lint` and `bun run build`.
- Root changes: verify the specific files you added or edited, and confirm scoped AGENTS files do not conflict.

## Docs usage

- Do not open or edit `design/` unless the task needs UI/design context.
- Prefer `README.md`, `MASTER_PLAN.md`, and the nearest scoped `AGENTS.md` before reading deeper project material.

## Global conventions

- Route work to the nearest scoped AGENTS file before making module-specific changes.
- Prefer `bun` over `npm` for frontend install, dev, build, and lint commands unless the user explicitly asks otherwise.
- Keep generated outputs, caches, model weights, and secrets out of commits unless the user explicitly asks for them.
- Preserve existing module boundaries. If a task crosses backend, frontend, and pipeline code, describe the touch points clearly and verify each touched module separately.
- When a new submodule develops its own conventions or risk profile, add a nested `AGENTS.md` there instead of bloating the root file.
- For frontend canvas changes, verify both visual alignment and pointer interaction. Trace/path overlays should use image-coordinate alignment, wide hit targets, and pass-through empty areas so object editing and pan/zoom still work.
