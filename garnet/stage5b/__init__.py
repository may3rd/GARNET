"""Backend-free Stage 5b pipe tracer + review gate.

Vendored from backend/garnet/path_tracer/ so the trace + branch-trace overlays can be produced
from garnet/ alone, with no dependency on the backend package:

    cv_pipe_tracer.py     the CV walker (borrowed algorithm) — self-contained upstream
    stage5b_pipeline.py   Stage5bPipelineMixin, the port/branch orchestration
    port_projection.py    the one helper it used from backend/garnet/ai_import.py
    host.py               Stage5bHost — replaces PIDPipeline as the mixin's host
    run.py                CLI that wires fixtures -> mask -> host -> tracer

Review gate over the traced network — mostly deterministic:

    review_gate.py        R2/R3 in Python, R1 in Python except its residual -> vision model
    review_rules.md       the rules, thresholds and their calibration
    review_prompt.md      the narrow question the model is asked (residual only)

Stage 6 — attach semantic evidence to the traced paths (line numbers, valves, tags, arrows):

    trace_associations.py  vendored byte-identical from backend/garnet/
    flow_direction.py      vendored byte-identical (its only dependency)
    run_associations.py    CLI: stage5b artifacts -> stage6_* artifacts + overlay

    python -m garnet.stage5b.run_associations --stem Test-00001

**Line-number association is the point of Stage 6.** A label attaches to the path it is written
ALONG, not merely the nearest one: when the globally nearest segment runs perpendicular to the
label's long axis (a crossing pipe clipping a corner), an orientation-matching segment within
threshold wins instead. Measured on Test-00001 that preference decides 24 of 25 attachments, and
in three cases it deliberately picks a path 50-83px away over one 2-32px away — correctly, since
the near path runs perpendicular to the label. A naive "nearest path wins" check flags those
three as errors; they are not. Labels that find nothing in range become `needs_review` and
traces with no label are reported in `traces_without_line_number` — never fabricated.

**This copy has DIVERGED from the backend original — deliberately.** The tracer and the mixin
were vendored byte-identical except for import paths, but both now carry fixes that exist only
here:

  * `cv_pipe_tracer.py` — a pressure relief valve is a terminal rather than a forced 90-degree
    turn (`psv_is_terminal`); a short mask break is bridged when equipment lies just beyond it
    (`gap_bridge_max_px`); position-only terminals clear `terminal_obj_id` instead of carrying a
    stale id from earlier in the walk.
  * `stage5b_pipeline.py` — a branch merging into an existing path records the joined path in
    `joined_trace_id` instead of mislabelling `terminal_obj_id`.

Do NOT assume the two trees still agree. Run

    git hash-object garnet/stage5b/cv_pipe_tracer.py \
        backend/garnet/path_tracer/cv_pipe_tracer.py

before claiming parity: matching hashes mean identical, differing hashes mean the backend copy
is BEHIND and needs the same fixes if the pipeline is still in use.
"""

from __future__ import annotations

__all__ = ["Stage5bConfig", "Stage5bHost", "build_pipe_mask", "load_fixtures",
           "load_equipment_ports", "merge_line_numbers",
           "load_walks", "r1_profile", "r1_verdict", "r2_verdict", "r3_verdict",
           "overlap_analysis"]

_LAZY = {
    "Stage5bConfig": "host", "Stage5bHost": "host", "build_pipe_mask": "host",
    "load_fixtures": "host", "load_equipment_ports": "host",
    "merge_line_numbers": "host",
    "load_walks": "review_gate", "r1_profile": "review_gate", "r1_verdict": "review_gate",
    "r2_verdict": "review_gate", "r3_verdict": "review_gate",
    "overlap_analysis": "review_gate",
}


def __getattr__(name: str):
    # Lazy so `import garnet.stage5b` does not require numpy/cv2 just to read the docstring.
    mod = _LAZY.get(name)
    if mod:
        from importlib import import_module
        return getattr(import_module(f".{mod}", __package__), name)
    raise AttributeError(name)
