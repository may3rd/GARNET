# Stage 5b pipe-trace review rules

Rules for the review gate over `stage5b_trace_results.json` and
`stage5b_branch_trace_results.json`.

**Implemented mostly in Python.** R2 and R3 are decided deterministically, always. R1 is
decided deterministically when clear-cut; only its residual reaches a vision model. On
Test-00001 that is 7 model calls out of 67 walks (was 17 when the model saw everything), with
60 of 67 walks decided by Python alone. See `review_gate.py` for the thresholds and
`review_prompt.md` for the narrow question the model is asked.

Vocabulary used below:

- **trace** — a walk launched from an equipment nozzle or a connection-object port.
- **branch** — a walk launched from a tee/junction discovered mid-trace.
- **terminal** — where a walk stopped (`terminal_type`).

## R1 — a pipe line is a solid line

A traced path must follow a **solid** drawn line. Real pipes are drawn solid; dashed or
dotted strokes on a P&ID are signal lines, impulse lines, off-page continuation, or
dimension/leader lines, and must not be reported as pipe.

Measured three ways from pixels only:

- **coverage** — fraction of path pixels with drawing ink within a 2px window.
- **on_line** — fraction of path pixels whose nearest ink is within 1px across the line of
  travel. Diagnostic only, never a gate (see below).
- **unexplained gaps** — where ink is missing, is that gap inside a known object/symbol bbox?
  A gap inside a valve, instrument or equipment box is a symbol sitting on the pipe
  (legitimate). A gap in open paper means the walk lost its line.

Decision:

| condition | verdict |
|---|---|
| coverage < 0.20 | `reject` — almost no ink under the path |
| unexplained ≥ 100px | `reject` — a long run over paper no symbol explains |
| unexplained == 0 **and** coverage ≥ 0.60 | `accept` |
| otherwise | `review` — the model's only job |

**Why `on_line` is not a gate.** A walk correctly following real pipe through inline valves
scores a *low* `on_line`: ink inside a valve body sits off the centreline, so the median
offset is 0 while the centred fraction is only 0.60. Symbols both depress the score and
explain the gaps. Gating `accept` on `on_line ≥ 0.95` sent 24 walks to the model; gating on
explained-gaps instead sent 7, with identical correctness on the labelled set.

**Why gap size does not separate the cases.** `branch_000013` is a real pipe with a 36px
unexplained gap; `branch_000023` is a bad walk whose unexplained gaps are only 4px — it runs
along a `3/4"` text label. Size cannot distinguish a small symbol from a text label; only
looking at what is drawn there can. That is precisely the residual handed to the model.

**Calibration note (do not widen the ink window past ~3px).** Measured on a known-good and a
known-bad walk:

| ink window | good walk | bad walk | separation |
|---|---|---|---|
| 0px | 0.04 | 0.05 | none |
| 1px | 0.40 | 0.11 | 0.29 |
| **2px** | **0.46** | **0.15** | **0.31** |
| 4px | 0.56 | **1.00** | inverted |
| 6px | 0.75 | **1.00** | inverted |

At 0px both score ~0.04: the walker snaps to a centreline ~1px off the drawn line and JPEG
antialiasing leaves a light fringe. At 4px the *bad* walk scores a perfect 1.00, because it
passes within 8px of unrelated ink — the window starts admitting neighbouring linework and
coverage stops discriminating entirely. 2px is the maximum useful tolerance.

**Gap attribution must be majority-coverage, not any-hit.** A gap counts as explained only if
≥50% of its pixels lie inside known bboxes. Marking a gap explained when *any* pixel was known
let a single stray pixel excuse a 200px blank-paper run.

## R2 — only one traced line at the same location

Two walks must not retrace the same linework. Decided by pure geometry: rasterise every path
to a 3px-wide mask and intersect.

- `shared_ratio ≥ 0.95` → `reject` (nothing new)
- `shared_ratio ≥ 0.85` → `trim` from the first shared point, keeping the new prefix
- more than ~10px shared but below the thresholds → `accept` (junction overlap)

The earlier/longer path wins; it is the primary trace.

## R3 — a pipe line terminates on something real

Decided by pure lookup. Valid terminals:

| terminal | meaning |
|---|---|
| `equipment` | an equipment nozzle / bbox |
| `page_connection` | an off-page or utility connector |
| `tee_junction` | a tee or junction |
| `instrument_tag` | an inline instrument or flow instrument |

`flag` (valid walk, but not a network endpoint):

| terminal | meaning |
|---|---|
| `branch_connection` | stops on its parent line — useful linework, not an endpoint |
| any valid terminal whose `terminal_obj_id` bbox does **not** contain the terminal point | the label is wrong and must not be trusted — on Test-00001, 5 of 36 bbox-naming terminals were inconsistent, e.g. `branch_000008` claiming `obj_000202` at (935,1110) when that object sits at x=150–355 |

`reject`:

| terminal | meaning |
|---|---|
| `dead_end` | ran out of ink mid-sheet — either a real pipe end (rare) or a failed walk |
| `no_pipe` | never found pipe at the start point |
| `max_steps` | hit the step cap without reaching anything |
| `sheet_edge` | ran off the sheet, not through a border connector |
| `null` / missing | the walk was skipped or produced no terminal |

## Verdicts and what they mean

| verdict | action |
|---|---|
| `accept` | real pipe run; keep as-is |
| `flag` | keep the walk, but it is not an endpoint or carries an inconsistent label |
| `trim` | real pipe for part of its length; drop the tail after `trim_from` |
| `reject` | drop the walk |
| `review` | only R1's residual — the model decides `accept`/`reject`/`trim`/`unsure` |

The gate never mutates geometry — `reject`/`trim` only decide whether a walk survives into the
graph. Output goes to `reviewed_walks.json` under `kept` / `dropped`.

## Ground truth used for calibration

Twelve walks labelled by visual inspection of the rendered crops (12 walks: 6 bad, 6 good,
including a duplicate and two dead-ends). The split handles all 12 correctly — no bad walk is
accepted. Treat the thresholds as calibrated-but-provisional until a larger set is labelled.

