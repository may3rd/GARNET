# Agent instructions (scope: this directory and subdirectories)

## Scope and layout
- This AGENTS.md applies to the repository root and everything below it.
- Keep root guidance global. Put Python pipeline specifics in `backend/garnet/AGENTS.md` and Gemini adapter specifics in `backend/gemini_detector/AGENTS.md`.
- Use `MASTER_PLAN.md` as the architecture roadmap for P&ID digitizing work. Do not restate or contradict it in ad hoc task notes.

## Modules / subprojects

| Module | Type | Path | What it owns | How to run | Tests / checks | Docs | AGENTS |
|--------|------|------|--------------|------------|----------------|------|--------|
| Backend API | fastapi | `backend/` | HTTP API, file upload flow, result serving, runtime config | From `backend/`: `uvicorn api:app --reload --port 8001` | `python -m py_compile api.py garnet/*.py garnet/utils/*.py` | `README.md` | `backend/garnet/AGENTS.md` |
| P&ID pipeline | python package | `backend/garnet/` | Stage-by-stage P&ID rebuild, shared OCR/image utilities, pipeline orchestration | From `backend/`: `python -m garnet.pid_extractor` | See module AGENTS | `MASTER_PLAN.md` | `backend/garnet/AGENTS.md` |
| Gemini detector | python adapter | `backend/gemini_detector/` | Gemini/OpenRouter SAHI detector for text-like classes | Called from backend code | `python -m py_compile gemini_detector/*.py` | module file | `backend/gemini_detector/AGENTS.md` |
| Frontend | react + vite | `frontend/` | Review UI, canvas editing, exports, backend API client | From `frontend/`: `npm run dev` | `npm run build`, `npm run lint` | `README.md` | none |
| DeepLSD | vendored library | `DeepLSD/` | Line-detection experiments and model support code | Follow local README / requirements | Module-specific | `DeepLSD/README.md` | none |
| Design assets | docs/assets | `design/` | Design references and non-runtime materials | Open only when task requires design context | n/a | `design/README.md` | none |

## Cross-domain workflows
- Frontend <-> backend: the Vite app proxies `/api` and `/runs` to `VITE_API_URL`, defaulting to `http://localhost:8001`. Keep backend route changes synchronized with frontend API usage.
- API <-> pipeline: `backend/api.py` is the service entrypoint, but the extraction logic lives in `backend/garnet/`. Change request/response shapes in the API layer only after checking the downstream pipeline output and frontend expectations.
- Pipeline roadmap: for P&ID digitizing features, preserve the stage model in `MASTER_PLAN.md` and the scoped rules in `backend/garnet/AGENTS.md`. The live rebuild is currently Stage 1-only.
- Generated artifacts: keep predictions, runs, temp files, and debug outputs in backend-owned artifact folders. Do not make the frontend depend on developer-local filesystem paths.

## Verification (preferred commands)
- Default rule: run checks from the owning module, keep them narrow first, and widen only after the touched area is stable.
- Backend: from `backend/`, use `python -m py_compile api.py garnet/*.py garnet/utils/*.py`; run targeted runtime checks only when the relevant weights/dependencies are present.
- Frontend: from `frontend/`, use `npm run lint` and `npm run build`.
- Root changes: verify the specific files you added or edited, and confirm scoped AGENTS files do not conflict.

## Docs usage
- Do not open or edit `design/` unless the task needs UI/design context.
- Prefer `README.md`, `MASTER_PLAN.md`, and the nearest scoped `AGENTS.md` before reading deeper project material.

## Global conventions
- Route work to the nearest scoped AGENTS file before making module-specific changes.
- Keep generated outputs, caches, model weights, and secrets out of commits unless the user explicitly asks for them.
- Preserve existing module boundaries. If a task crosses backend, frontend, and pipeline code, describe the touch points clearly and verify each touched module separately.
- When a new submodule develops its own conventions or risk profile, add a nested `AGENTS.md` there instead of bloating the root file.

## Do not
- Do not put backend pipeline implementation rules in the root file when they belong in `backend/garnet/AGENTS.md`.
- Do not assume frontend, backend, and pipeline changes can be validated with one command.
- Do not treat generated artifacts or local weights as source files.

## Links to module instructions
- `backend/garnet/AGENTS.md`
- `backend/gemini_detector/AGENTS.md`
