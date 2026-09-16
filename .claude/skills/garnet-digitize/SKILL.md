---
name: garnet-digitize
description: End-to-end GARNET P&ID digitization — object detection, equipment/line-number extraction via AI, and pipe-tracing/graph assembly, for one sheet or a batch. Use for phrasings like "digitize this P&ID", "fully process this drawing", "build the full connectivity graph and line list for this sheet", "run the complete GARNET pipeline", or "digitize this batch of drawings". Orchestrates garnet-detect, garnet-equipment-bbox, garnet-line-numbers, and garnet-trace-graph — use those individually for a single stage instead.
---

# garnet-digitize

Orchestrates the full GARNET pipeline for one sheet: detection → AI equipment/line-number
extraction → tracing/graph assembly → report.

Full contract reference (all schemas, tunables, stage table): `reference/contracts.md` (this
skill owns it; other `garnet-*` skills link back here).

## Recipe — one sheet

1. **Detect.** Run Stage 4 only to get the object list (see `garnet-detect`):
   ```bash
   cd /Users/maetee/Code/GARNET/backend
   /Users/maetee/Code/GARNET/.venv/bin/python -m garnet.pid_extractor \
     --image <sheet> --out <output-dir> --stop-after 4
   ```
2. **Extract equipment and line numbers in parallel.** Dispatch both against the *same* raster
   used in step 1:
   - `garnet-equipment-extractor` agent (or the `garnet-equipment-bbox` skill inline) →
     equipment bbox + ports JSON (Contract B).
   - `garnet-line-number-extractor` agent (or the `garnet-line-numbers` skill inline) →
     line-number text JSON (Contract C).
   Both must produce a rendered, human-viewed overlay before returning — that's a hard
   requirement of the underlying prompts, not optional polish.
3. **Fold the AI JSON back in and finish the pipeline:**
   ```bash
   /Users/maetee/Code/GARNET/.venv/bin/python -m garnet.pid_extractor \
     --image <sheet> --out <output-dir> \
     --ai-equipment <equipment.json> --ai-line-numbers <line_numbers.json> \
     --stop-after 11
   ```
   (Only add `--ai-align` if the AI JSON's frame doesn't match — see the reference doc; prefer
   re-extracting from the same raster instead.)
4. **Report from:**
   - `<output-dir>/stage10_line_list.json` — line number → traced edges, canonical line id,
     length, flow direction.
   - `<output-dir>/stage7b_graph_v1.json` — full connectivity graph (equipment, ports, lines,
     instruments, relationships).
   - `<output-dir>/stage10_equipment_connectivity.json` — equipment-level connectivity.

If tracing looks wrong at step 3, don't restart from step 1 — see the "when tracing goes
wrong" table in `garnet-trace-graph` and iterate with `run_stage5b_only.sh`.

## Batch / multi-sheet

Run step 1 for every sheet first (cheap, no LLM). Then dispatch the extraction agents for
*all* sheets in parallel — one `garnet-equipment-extractor` and one `garnet-line-number-extractor`
call per sheet, not one call for all sheets (each agent operates on exactly one sheet and must
view its own overlay). Once every sheet's AI JSON is back, run step 3 per sheet. If the sheets
are connected by off-page connectors and need to be merged into one graph, use
`POST /api/pipeline/merge` (see root `CLAUDE.md`) after each sheet reaches Stage 11 — that is
outside what these skills cover.

## Common failures across the whole flow

- **Frame mismatch** between an AI JSON and the job image is the most common failure — always
  extract from the exact raster fed to `--image`.
- **Equipment import replaces the whole equipment bucket** in `stage4_objects.json` — re-running
  step 3 with a revised equipment JSON discards the previous equipment objects, not merges them.
- **Out-of-vocabulary `Equipment_type`** values are silently skipped, not imported — check the
  extraction agent's summary for a skipped count before assuming equipment is complete.
