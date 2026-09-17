"""Backend-free Stage 5b pipe tracer.

Vendored from backend/garnet/path_tracer/ so the trace + branch-trace overlays can be produced
from garnet/ alone, with no dependency on the backend package:

    cv_pipe_tracer.py     the CV walker (borrowed algorithm) — self-contained upstream
    stage5b_pipeline.py   Stage5bPipelineMixin, the port/branch orchestration
    port_projection.py    the one helper it used from backend/garnet/ai_import.py
    host.py               Stage5bHost — replaces PIDPipeline as the mixin's host
    run.py                CLI that wires fixtures -> mask -> host -> tracer

`stage5b_pipeline.py` and `cv_pipe_tracer.py` are byte-identical to their backend originals
except for import paths, so behaviour matches the pipeline.
"""

from __future__ import annotations

__all__ = ["Stage5bConfig", "Stage5bHost", "build_pipe_mask", "load_fixtures",
           "load_equipment_ports", "merge_line_numbers"]


def __getattr__(name: str):
    # Lazy so `import garnet.stage5b` does not require numpy/cv2 just to read the docstring.
    if name in ("Stage5bConfig", "Stage5bHost", "build_pipe_mask", "load_fixtures",
                "load_equipment_ports", "merge_line_numbers"):
        from . import host
        return getattr(host, name)
    raise AttributeError(name)
