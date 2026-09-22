"""Backend-free Stage 5b pipe tracer, review gate and Stage 6 associations.

Vendored from backend/garnet/path_tracer/ so the trace + branch-trace overlays can be produced
from garnet/ alone, with no dependency on the backend package.

TRACING (Stage 5b -> 5c)

    cv_pipe_tracer.py     the CV walker (borrowed algorithm) — self-contained upstream
    stage5b_pipeline.py   Stage5bPipelineMixin, the port/branch orchestration
    merge_branches.py     Step 5c — merge branch runs that join end-to-end into one line
    port_projection.py    the one helper it used from backend/garnet/ai_import.py
    host.py               Stage5bHost — replaces PIDPipeline as the mixin's host
    run.py                CLI that wires fixtures -> (optional OCR) -> mask -> host -> tracer

5c is not a separate CLI: `stage5b_pipeline.py` calls `merge_branch_runs` on the branch
results, so a branch line ends at a real terminate point instead of at the next junction dot.

    stage2_ocr.py         Stage 2 OCR text discovery (macOS Vision), opt-in via `run --ocr`
    ocrmac_sahi.py        the vendored OCR engine stage2_ocr drives

Stage 2 exists here because `build_pipe_mask` erases text so only pipework survives; without
OCR its only text boxes come from the fixture object list, and undetected text stays in the
mask as ink the tracer can walk. It is OFF by default — pass `--ocr`.

REVIEW GATE over the traced network — mostly deterministic:

    review_gate.py        R2/R3 in Python, R1 in Python except its residual -> vision model
    review_rules.md       the rules, thresholds and their calibration
    review_prompt.md      the narrow question the model is asked (residual only)
    review_gate_jev.design.md, review_gate_jev.v2.design.md   design notes, not code

INSPECTION — neither is part of a run; both are run by hand after one:

    coverage_audit.py     pipe mask ink no walk covered, split pipe-like vs frame/blob.
                          Walk counts and terminal histograms can be identical while whole
                          branches go unvisited; this is the check that catches that.
    inspect_line_numbers.py  one review tile per assigned label, showing the winner AND the
                          geometrically closer path that lost on orientation

STAGE 6 — attach semantic evidence to the traced paths (line numbers, valves, tags, arrows):

    trace_associations.py  vendored byte-identical from backend/garnet/
    flow_direction.py      vendored byte-identical (its only dependency)
    run_associations.py    CLI: stage5b artifacts -> stage6_* artifacts + overlay

    python -m garnet.stage5b.run_associations --stem 14780-8120-25-25-0002A

**Line-number association is the point of Stage 6.** A label attaches to the path it is written
ALONG, not merely the nearest one: when the globally nearest segment runs perpendicular to the
label's long axis (a crossing pipe clipping a corner), an orientation-matching segment within
threshold wins instead. Measured on 14780-8120-25-25-0002A that preference decides 24 of 25 attachments, and
in three cases it deliberately picks a path 50-83px away over one 2-32px away — correctly, since
the near path runs perpendicular to the label. A naive "nearest path wins" check flags those
three as errors; they are not. Labels that find nothing in range become `needs_review` and
traces with no label are reported in `traces_without_line_number` — never fabricated.

**This copy has DIVERGED from the backend original — deliberately, and by now substantially.**
Both `cv_pipe_tracer.py` and `stage5b_pipeline.py` differ from the backend copies by roughly a
thousand diff lines each. Fixes that exist only here:

  * `cv_pipe_tracer.py` — a pressure relief valve is a terminal rather than a forced 90-degree
    turn (`psv_is_terminal`); a short mask break is bridged when equipment lies just beyond it
    (`gap_bridge_max_px`), screened against dash trains so a bridge does not jump onto a dashed
    line; a whole dashed-line screen keeps walks on solid pipe (no `dash`/`dashed` handling
    exists in the backend copy at all); a walk that ARRIVES at a junction dot stops there, and
    the junction rule crosses only when the line continues, measured from the dot; position-only
    terminals clear `terminal_obj_id` instead of carrying a stale id from earlier in the walk.
  * `stage5b_pipeline.py` — a branch merging into an existing path records the joined path in
    `joined_trace_id` instead of mislabelling `terminal_obj_id`; branches are seeded from orphan
    junction dots and branches already owned by a trace are suppressed; duplicate walks are
    suppressed and degenerate loops trimmed; Step 5c merging (`merge_branches`).

Do NOT assume the two trees still agree. Run

    git hash-object garnet/stage5b/cv_pipe_tracer.py \
        backend/garnet/path_tracer/cv_pipe_tracer.py

before claiming parity: matching hashes mean identical, differing hashes mean the backend copy
is BEHIND and needs the same fixes if the pipeline is still in use. As of this writing they
differ. Port fixes OUT of garnet/ if the backend needs them, never in.
"""

from __future__ import annotations

__all__ = ["Stage5bConfig", "Stage5bHost", "build_pipe_mask", "load_fixtures",
           "load_equipment_ports", "merge_line_numbers", "merge_ocr_regions",
           "Stage2OcrConfig", "run_stage2_ocr", "merge_branch_runs",
           "load_walks", "r1_profile", "r1_verdict", "r2_verdict", "r3_verdict",
           "overlap_analysis"]

_LAZY = {
    "Stage5bConfig": "host", "Stage5bHost": "host", "build_pipe_mask": "host",
    "load_fixtures": "host", "load_equipment_ports": "host",
    "merge_line_numbers": "host", "merge_ocr_regions": "host",
    "Stage2OcrConfig": "stage2_ocr", "run_stage2_ocr": "stage2_ocr",
    "merge_branch_runs": "merge_branches",
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
