# Prompt — Extract equipment bounding boxes from a P&ID

You are extracting **major equipment** and their bounding boxes from a P&ID (PDF or image).
Output is used for downstream geometry, so a wrong coordinate is worse than a missing one.

## Non-negotiable rules

1. **Never invent a tag, a size, or a coordinate.** If you cannot read it, it goes in
   `unresolved[]`, not in `major_equipment[]`.
2. **Never report a box in a coordinate space you did not define and verify.** Every box is
   accompanied by the exact frame it was measured in.
3. **Never emit the result without looking at your own overlay.** Rendering boxes onto the
   drawing and visually checking them is a required step, not a nicety.
4. Prefer the PDF **text layer** over vision for every string (tags, sizes, line numbers).
   Vision decides *where*; text decides *what*.

## Step 1 — Establish and verify the coordinate frame

- Read the page: `mediabox` width/height **and `/Rotate`**. A rotated page is the single most
  common source of transposed or mirrored boxes.
- Render the page to a raster at a stated DPI. Record the render's pixel size.
- **Aspect gate:** `render_w/render_h` must equal the *displayed* page aspect (after rotation)
  within 1%. If it does not, stop — you are about to measure in the wrong space.
- Record the transform you will use, and sanity-check it on one known feature (e.g. plot one
  extracted vector segment and confirm it lands on the drawn line).
- **Legibility gate:** if equipment tags are not readable in the render, increase DPI and
  re-render. Do not extract from a render where the text is illegible; say so instead.

Minimum render width for a full-size sheet (A1/A0): **2500 px**. Below that, tags are guesses.

## Step 2 — Build the equipment inventory from text, before looking for shapes

Sources, in priority order:

1. The **equipment data block** on the sheet (usually top-left): `ITEM No.`, `SERVICE`,
   `SIZE`, design conditions. This is authoritative for tag and size.
2. The **title block** (drawing title often names the unit and service).
3. **In-drawing labels** on the symbol itself.

Cross-check 1 against 3. If they disagree, report both and flag it — do not silently pick one.

Do not carry over an equipment list from any prior extraction, JSON, or conversation. Rebuild
it from this sheet. If a prior list contains an item you cannot find on the sheet, the correct
action is to **delete it and say so**, not to place a box where it "should" be.

Major equipment = vessels, columns, drums, reactors, tanks, pumps, compressors, exchangers,
fired heaters, filters, package units. Not: instruments, valves, in-line fittings, sight
glasses, analyzers, off-page connectors.

### Equipment_type controlled vocabulary

`Equipment_type` is imported into GARNET's pipeline, which only accepts values from a fixed
set (`EQUIPMENT_LABELS` in `backend/garnet/pid_extractor.py`). **Anything outside this list is
skipped by the importer, not guessed at or coerced** — so pick the nearest listed label, or if
none fits, put the item in `unresolved[]` and say why.

Allowed `Equipment_type` values (underscore or space variants both accepted):

```
vessel, column, pump, compressor, blower, heat exchanger, tank, reactor,
mixer, pot, knockout drum, filter, cooler, heater, injection pump
```

Documented synonyms the importer maps for you (use the label directly if you already know it;
these exist so prose equipment names don't get lost) — from `AI_CLASS_MAP` in
`backend/garnet/ai_import.py`:

| If the drawing/sheet calls it...        | Use `Equipment_type`... |
| ---------------------------------------- | ------------------------ |
| static mixer / inline mixer              | `mixer`                  |
| ko drum / knockout drum                  | `knockout drum`          |
| shell and tube exchanger / exchanger     | `heat exchanger`         |
| air cooler                               | `cooler`                 |
| drum / separator / accumulator           | `vessel`                 |

If the equipment is a real vessel type but doesn't fit any row above and isn't already one of
the allowed values verbatim (e.g. "surge tank" → `tank`, "flare knockout drum" → `knockout
drum`), pick the closest allowed value and note your reasoning in `Evidence`. Only fall back to
`unresolved[]` when no allowed value is a reasonable fit.

## Step 3 — Measure each box

- Box = the equipment **symbol outline** only: vessel shell including heads; pump casing plus
  its driver. Exclude nozzles' external piping, tag text, dimension lines, and callouts.
- Measure by cropping and enlarging the region, not by eyeballing the full sheet.
- Give every box in **both** forms:
  - `bounding_box_px` in the frame defined in Step 1
  - `bounding_box_norm` = `[x0/W, y0/H, x1/W, y1/H]`, 4 decimals — resolution independent

## Step 3b — Mark the ports (pipe connection points)

A port is a point, not a box: where a process line meets the equipment outline.

- One port per line that terminates on the symbol. Include nozzles drawn with a
  stub and a mark, and lines that land directly on the outline.
- Exclude: instrument tappings and impulse lines, lines that only pass behind
  the symbol, internals (distributors, baffles, weirs), and the vessel's own
  level-bridle taps unless they are a drawn process nozzle.
- The point is where the line crosses the outline, not the end of the stub and
  not the valve next to it.
- `mark` is the nozzle letter on the drawing (AV, BO, CI, DN1…). Unreadable or
  absent → `null`. Never invent one.
- `size` and `line_number` come from the adjacent text only. Not inferable →
  `null`. Do not carry a size across from a different line.
- `side` is which edge of the bounding box the port sits on: top | bottom |
  left | right. **This is required** — the importer silently drops any port
  missing a usable `side` or a 2-element `point_px`.
- Ports nest inside their equipment item. A line between two pieces of
  equipment on this sheet produces a port on each.

## Step 4 — Verify visually (mandatory)

Render the source page, draw every box and tag onto it, **open the image and look at it**.
Check each box: does it enclose the symbol, and only the symbol? Is the tag next to the right
shape? Adjust and repeat until it is right. State that you did this.

If any box is off, fix it before answering. Never ship an unviewed overlay.

## Step 5 — Output

```json
{
  "source_drawing": "<drawing number + revision from the title block>",
  "coordinate_frame": {
    "width_px": 0, "height_px": 0, "dpi": 0,
    "page_pt": [0, 0], "page_rotate_deg": 0,
    "note": "how the raster relates to the PDF page"
  },
  "objects": [
    {
      "Index": 1,
      "Object": "",
      "Tag": "",
      "Equipment_type": "",
      "Service": "",
      "Size": "",
      "Evidence": "equipment data block | in-drawing label | both (agree) | CONFLICT: ...",
      "Left": 0,
      "Top": 0,
      "Width": 0,
      "Height": 0,
      "Bounding_box_px": {"x_min":0,"y_min":0,"x_max":0,"y_max":0},
      "Bounding_box_norm": [0,0,0,0],
      "Score": 1.0,
      "Ports": [
        {
          "mark": "",
          "size": null,
          "line_number": null,
          "side": "top",
          "point_px": [0, 0],
          "point_norm": [0, 0]
        }
      ]
    }
  ],
  "unresolved": [
    {"what": "", "why": "", "location_hint_px": [0,0]}
  ],
  "verification": "overlay rendered and visually checked: yes/no"
}
```

`Object` is same as `Equipment_type`.

Omit `confidence` unless you can state what the number measures. A decorative confidence
score reads as diligence and is worse than no field at all.

## Self-check before answering

- [ ] Render aspect matches the rotated page aspect
- [ ] Transform validated against a known feature
- [ ] Every tag traced to a text source on this sheet
- [ ] No item carried over from a previous list without re-finding it here
- [ ] Every box has both px and normalised coordinates
- [ ] Every port has a `side` and a `point_px`; unreadable marks are `null`, not invented
- [ ] Every `Equipment_type` is one of the allowed values, or the item is in `unresolved[]`
- [ ] Overlay rendered **and viewed**; boxes sit on the symbols
- [ ] Anything unreadable is in `unresolved[]`, not guessed

## Changes from the original

This copy (tracked at `backend/garnet/OCR_prompts/pid_equipment_bbox_prompt.md`) differs from
the original at `backend/output/claude-output/pid_equipment_bbox_prompt.md` (gitignored, not
tracked) in two places, to match what `backend/garnet/ai_import.py` actually requires:

1. The Step 5 output template now includes the `Ports` array (was described in Step 3b but
   missing from the JSON template — `ai_import.py` requires it to place nozzle connections).
2. Added the "Equipment_type controlled vocabulary" section under Step 2, and a matching
   self-check item, documenting `EQUIPMENT_LABELS` / `AI_CLASS_MAP` from
   `backend/garnet/pid_extractor.py` and `backend/garnet/ai_import.py`. Out-of-vocabulary
   `Equipment_type` values are silently skipped by the importer, so the prompt now says so.
