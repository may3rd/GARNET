# Path Tracer Status

The active `garnet.path_tracer` package is CV-only. The main pipeline uses `cv_pipe_tracer.py` for Stage 5b pipe walking and uses CV edge scanning for page-connection port detection.

## Active Modules

| File | Role |
|------|------|
| `cv_pipe_tracer.py` | Main Stage 5b geometric pipe walker used by `PIDPipeline` |

## Removed Legacy Code

VLM port detection, hybrid tracing, VLM cursor helpers, VLM prompts, and legacy trace schemas were removed. The production path no longer imports `agent2_hybrid` or exposes `vlm` / `vlm+cv` port detection modes.

## Verification

Use these checks after path-tracer edits:

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests/test_path_tracer -p 'test*.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m unittest discover -s tests -p 'test_stage5b_branch_terminal.py' -v
/Users/maetee/Code/GARNET/.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py garnet/path_tracer/*.py
```
