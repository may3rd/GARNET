---
name: garnet-equipment-bbox
description: Extract major-equipment bounding boxes and nozzle/port locations from a GARNET P&ID sheet as Contract B JSON, using a vision LLM. Use for phrasings like "find the equipment boxes on this P&ID", "extract equipment tags and nozzles", "get bounding boxes for the vessels/pumps/exchangers on this sheet", or "prep equipment JSON for GARNET". YOLO does not detect equipment — this is the only source for it.
---

# garnet-equipment-bbox

Major equipment (vessels, columns, pumps, compressors, exchangers, tanks, reactors, filters…)
with bounding boxes and nozzle ports, from one P&ID sheet, as Contract B JSON.

**This is LLM work.** YOLO detects no equipment at all.

Full contract reference (Contract B schema, vocabulary gate, fixture staging):
`.claude/skills/garnet-digitize/reference/contracts.md`.

## Default route — one command

```bash
cd /Users/maetee/Code/GARNET
.venv/bin/python -m garnet.detect_equipment_bboxes --image <sheet>
```

`garnet/detect_equipment_bboxes.py` tiles the sheet, asks a vision model (ollama-cloud,
default `deepseek-v4.1-flash`) for boxes per tile, maps them back to sheet coordinates,
dedupes across tile seams, and writes:

- `output/<stem>_equipment_bboxes.json` — Contract B
- `output/<stem>_equipment_bboxes_overlay.png` — **view this** before using the JSON

Credentials: `$OLLAMA_API_KEY` or `~/.hermes/.env`. Flags: `--out`, `--model`, `--base-url`,
`--tile` (default 1024 — the provider downscales anything larger to ~1024, so bigger tiles
lose resolution, not gain it), `--overlap` (0.2), `--min-area` (400 px²), `--timeout` (300s
per tile), `--verbose`, `--no-overlay`.

## Fallback route — the extraction agent

When the script's output is poor on a sheet, or you need the richer fields the script does not
emit (`Service`, `Evidence`, `unresolved[]`, per-port `mark`/`size`/`line_number`), follow
`backend/garnet/OCR_prompts/pid_equipment_bbox_prompt.md` step by step yourself, or delegate to
the `garnet-equipment-extractor` agent (one sheet per call). Step 4 of that prompt — render the
overlay and actually look at it — is not optional.

Note: that agent still describes the hand-driven per-tile workflow the script now automates.
Prefer the script; use the agent when the script falls short.

## Constraints that bite either way

1. **`Equipment_type` vocabulary gate.** Only `EQUIPMENT_LABELS` values survive: `vessel,
   column, pump, compressor, blower, heat exchanger, tank, reactor, mixer, pot, knockout drum,
   filter, cooler, heater, injection pump`. Synonyms are mapped (static mixer→mixer, ko
   drum→knockout drum, shell and tube exchanger→heat exchanger, air cooler→cooler,
   drum/separator/accumulator→vessel; the `garnet/` loader additionally maps compound heads
   like "filter vessel"→filter). **Anything else is silently dropped** — a sheet can run with
   `equipment : 0`, trace nothing from equipment, and still exit 0. Check the count in the log.
2. **Ports need both a usable `side` (`top|bottom|left|right`) and a 2-element `point_px`**, or
   they are dropped. On the `garnet/` route, ports are read from a separate
   `ai_equipment_ports.json`, not from the bbox file.
3. **Frame must match the raster you trace.** Extract from the exact image you will feed
   downstream. (`--ai-align`, backend-only, fixes translation only — never scale — and needs
   `stage4_line_numbers.json` to already exist.)
4. **Backend import replaces the whole equipment bucket** in `stage4_objects.json` — every
   object whose `class_name` is in `EQUIPMENT_LABELS` is dropped before the new ones are added.

## Feeding it downstream

**`garnet/` route** — stage the file as a fixture, then trace:

```bash
cp output/<stem>_equipment_bboxes.json <indir>/<stem>_equipment_bboxes.json
.venv/bin/python -m garnet.stage5b.run --stem <stem> --indir <indir> --out output/<stem>
```

**backend route:**

```bash
cd backend && ../.venv/bin/python -m garnet.pid_extractor \
  --image <sheet> --out <dir> --ai-equipment <path.json> --stop-after 11
```

## Multi-sheet work

The script is cheap to loop — run it per sheet. Only fall back to parallel
`garnet-equipment-extractor` agent calls (one sheet each) when the script's output needs
replacing.
