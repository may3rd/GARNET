---
name: garnet-equipment-bbox
description: Extract major-equipment bounding boxes and nozzle/port locations from a GARNET P&ID sheet as Contract B JSON, for import into the pipeline with --ai-equipment. Use for phrasings like "find the equipment boxes on this P&ID", "extract equipment tags and nozzles", "get bounding boxes for the vessels/pumps/exchangers on this sheet", or "prep equipment JSON for GARNET". YOLO does not detect equipment — this is the only source for it.
---

# garnet-equipment-bbox

Extracts major equipment (vessels, columns, pumps, compressors, exchangers, tanks, reactors,
etc) with bounding boxes and nozzle ports from one P&ID sheet, as Contract B JSON for
`--ai-equipment`.

**Follow the prompt at `backend/garnet/OCR_prompts/pid_equipment_bbox_prompt.md` step by
step.** It defines the coordinate-frame procedure, the equipment inventory rules, the box
measurement rules, and the port-marking rules. Do not re-derive them here.

Full contract reference (Contract B, tunables, graph schema): `.claude/skills/garnet-digitize/reference/contracts.md`.

## GARNET-specific constraints on top of the prompt

1. **`Equipment_type` vocabulary gate.** Only values in `EQUIPMENT_LABELS`
   (`backend/garnet/pid_extractor.py`) import: `vessel, column, pump, compressor, blower, heat
   exchanger, tank, reactor, mixer, pot, knockout drum, filter, cooler, heater, injection
   pump`. `AI_CLASS_MAP` (`backend/garnet/ai_import.py`) maps a handful of synonyms (static
   mixer→mixer, ko drum→knockout drum, shell and tube exchanger→heat exchanger, air
   cooler→cooler, drum/separator/accumulator→vessel). Anything else is **silently skipped by
   the importer** — pick the nearest allowed label or put the item in `unresolved[]`.
2. **`Ports` is required per item**, not optional, even though older copies of this prompt
   omit it from the JSON template. A port needs both a usable `side`
   (`top|bottom|left|right`) and a 2-element `point_px`, or the importer drops it.
3. **Frame must match the job image**, or the import errors. Extract from the exact raster you
   will hand to `--image` (or a lossless re-render of the identical page/DPI). If you must use
   `--ai-align`, it only fixes translation (crop offset), never scale, and needs the job's own
   `stage4_line_numbers.json` to already exist — see the reference doc.
4. **Applying equipment replaces the whole equipment bucket** in `stage4_objects.json` — every
   existing object whose `class_name` is in `EQUIPMENT_LABELS` is dropped first, then this
   file's objects are added. Don't run this against a job whose existing equipment review you
   want to keep without checking first.
5. **Mandatory overlay verification** (Step 4 of the prompt is not optional here): render the
   boxes and ports onto the sheet and actually view the image before writing the output file.

## Output

Write the JSON next to your working files (e.g. `backend/output/claude-output/<sheet>_equipment_bboxes.json`)
and the verification overlay PNG alongside it. Feed it to the pipeline with:

```bash
python -m garnet.pid_extractor --image <sheet> --out <dir> --ai-equipment <path.json> --stop-after 11
```

## Multi-sheet work

For more than one sheet, delegate each sheet to the `garnet-equipment-extractor` agent in
parallel rather than doing them serially in this conversation.
