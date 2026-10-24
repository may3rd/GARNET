---
name: garnet-digitize
description: End-to-end GARNET P&ID digitization — YOLO symbol detection, LLM equipment/line-number extraction, pipe tracing and evidence association, for one sheet or a batch. Use for phrasings like "digitize this P&ID", "fully process this drawing", "build the line list for this sheet", "run the complete GARNET pipeline", or "digitize this batch of drawings". Orchestrates garnet-detect, garnet-equipment-bbox, garnet-line-numbers, and garnet-trace-graph — use those individually for a single stage instead.
---

# garnet-digitize

Orchestrates the full GARNET flow for one sheet: detection → LLM equipment/line-number
extraction → tracing → review gate → Stage 6 associations.

Full contract reference (all schemas, tunables, stage table, fixture staging):
`reference/contracts.md` (this skill owns it; other `garnet-*` skills link back here).

## Pick a route first

- **`garnet/` (repo root) — the current tree.** Fixture-driven, backend-free, actively
  developed. Reaches tracing + Stage 6 associations. **It does not assemble a graph or emit a
  line list.**
- **`backend/` — the legacy pipeline behind the API.** Only route that produces
  `stage7b_graph_v1.json`, `stage10_line_list.json`, `stage10_equipment_connectivity.json`.

Default to `garnet/`. Switch to `backend/` only when the deliverable is the graph or the line
list.

## Recipe — one sheet, `garnet/` route

Run everything from `/Users/maetee/Code/GARNET` with `.venv/bin/python`.

```bash
S=<stem>; IN=<indir>; OUT=output/$S      # IN holds <stem>.jpg

# 1. symbols (YOLO+SAHI, no LLM)
.venv/bin/python -m garnet.detect_objects --image $IN/$S.jpg --out $IN/${S}_objects.json

# 2a. equipment boxes + nozzles (vision LLM)
.venv/bin/python -m garnet.detect_equipment_bboxes --image $IN/$S.jpg \
  --out $IN/${S}_equipment_bboxes.json

# 2b. line-number text: no script — use the garnet-line-numbers skill, or dispatch the
#     garnet-line-number-extractor agent. Write the result to $IN/${S}_line_numbers.json

# 3. trace
.venv/bin/python -m garnet.stage5b.run --stem $S --indir $IN --out $OUT

# 4. review gate over the traced network (calls a vision model; --no-llm to skip)
.venv/bin/python -m garnet.stage5b.review_gate --stem $S --dir $OUT

# 5. attach line numbers, valves, tags, arrows
.venv/bin/python -m garnet.stage5b.run_associations --stem $S --indir $IN --dir $OUT
```

**Steps 2a and 2b are independent — run them in parallel.** Both must produce an overlay you
actually look at before step 3.

The three JSONs must land in `$IN` under the names `garnet/stage5b/host.py:load_fixtures`
accepts — that staging step is the one people skip, because steps 1–2 take `--image` while
steps 3–5 take `--stem`/`--indir`. Accepted spellings are listed in `reference/contracts.md`.

### Report from

- `$OUT/stage6_trace_associations.json` — every traced edge with its attachments.
- `$OUT/stage6_line_number_review.json` — line numbers accepted vs `needs_review`, plus
  `traces_without_line_number`.
- `$OUT/stage6_trace_association_overlay.png` — the visual deliverable.
- `$OUT/stage5b_trace_overlay.png`, `$OUT/stage5b_branch_trace_overlay.png`.

Verify the line-number assignments before reporting:
`.venv/bin/python -m garnet.stage5b.inspect_line_numbers --stem $S --dir $OUT` → open
`inspection.html`. A label attaches to the path it is written **along**, not the nearest —
green is the winner, blue is the closer path that lost on orientation. Blue winning visually
means a real error; green winning over a closer blue is usually correct.

## Recipe — one sheet, `backend/` route (graph + line list)

1. Detect through Stage 4 (see `garnet-detect`):
   ```bash
   cd /Users/maetee/Code/GARNET/backend
   /Users/maetee/Code/GARNET/.venv/bin/python -m garnet.pid_extractor \
     --image <sheet> --out <output-dir> --stop-after 4
   ```
2. Extract equipment and line numbers against the **same raster** (steps 2a/2b above).
3. Fold them in and finish:
   ```bash
   /Users/maetee/Code/GARNET/.venv/bin/python -m garnet.pid_extractor \
     --image <sheet> --out <output-dir> \
     --ai-equipment <equipment.json> --ai-line-numbers <line_numbers.json> \
     --stop-after 11
   ```
4. Report from `stage10_line_list.json`, `stage7b_graph_v1.json`,
   `stage10_equipment_connectivity.json`.

If tracing looks wrong at step 3, don't restart from step 1 — see `garnet-trace-graph` and
iterate with `run_stage5b_only.sh`.

## Batch / multi-sheet

Steps 1 and 2a are plain scripts — loop them per sheet. Step 2b is the only agent work: one
`garnet-line-number-extractor` call **per sheet**, dispatched in parallel, never one call for
all sheets (each agent handles exactly one sheet and views its own overlay). Then run steps
3–5 per sheet; `bash garnet/run_test.sh` does step 3 across every fixture sheet at once.

Multi-sheet merge via off-page connectors (`POST /api/pipeline/merge`) is backend-only and
outside what these skills cover.

## Common failures across the whole flow

- **Fixture not staged / wrong filename** — steps 3–5 silently run with zero equipment or zero
  line numbers and still exit 0. Check the counts in the run log before trusting the output.
- **Frame mismatch** between an LLM JSON and the raster — always extract from the exact image
  you trace.
- **Out-of-vocabulary `Equipment_type`** is silently skipped, not imported. Sheet 0004A once
  lost its only equipment this way ("Filter Vessel" matched neither `filter` nor `vessel`).
- **Backend equipment import replaces the whole equipment bucket** in `stage4_objects.json` —
  re-running with a revised equipment JSON discards the previous equipment objects.

## Agent-system route (pipeline + orchestrator)

Use for multi-sheet jobs, graph / line list / cross-sheet linking from `garnet/` without the
backend, and scored runs against golden sheets. `garnet.pipeline` runs every deterministic stage
and gate (G0–G7) and never calls a model; when it needs agent work it writes `dispatch.json` and
exits 3. Exit codes: 0 pass, 1 hard gate failed, 2 bad input, 3 waiting on agents.

```bash
# fixtures only, no agents (deterministic stages + gates)
.venv/bin/python -m garnet.pipeline --job <name> --sheets <stem>... --from-fixtures <indir> --no-agents
# full run: delegate to the garnet-orchestrator agent (dispatches the extractor/reader/reviewer agents, --resume loop)
```

Outputs land in `output/<job>/`: `run_report.json`, `dispatch.json`, `sheets/<stem>/`, `out/<stem>/graph.json`,
`graph_merged.json`, `review/review_decisions.json`, `eval/score.json` (with `--golden`).
Design doc: https://claude.ai/artifact/M6EYR2JnmD7mfrH6hGYCia
