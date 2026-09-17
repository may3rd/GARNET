# Stage 5b trace review prompt (vision adjudication — residual cases only)

Version: 2. This prompt is sent ONLY for walks the deterministic screen could not decide.
See `review_rules.md` for the rules and `review_gate.py` for what reaches the model.

The gate has already decided everything measurable: whether ink is under the path, whether the
path is centred on it, and whether each interruption is inside a known symbol. What it cannot
decide from pixels is the single question below.

---

You are reviewing ONE traced path from an automatically digitised P&ID (piping and
instrumentation diagram). A computer-vision walker followed dark pixels and reported the path
below.

## Your single question

The measurements below are already agreed; do not re-derive them. The open question is:

**Where the path crosses a gap in the ink, is that gap caused by something DRAWN on the pipe
(a valve, instrument, or symbol body), or is the path running over a text label / blank paper
with no pipe there?**

That distinction is the only reason this crop is being shown to you.

## What is drawn on this drawing

- **Process pipe lines are drawn as SOLID black lines.**
- **Dashed, dotted, or broken lines are NOT process pipes.** They are signal lines,
  instrument impulse lines, off-page continuation marks, or leader/dimension lines.
- A solid pipe may be **interrupted by printed text** — a line number, an equipment tag, or a
  valve body sitting on the line. That interruption is normal and does not make the line
  dashed.
- A real pipe line **ends at a connection**: equipment, an off-page/page connector, a tee or
  junction, or an inline instrument.

## The path under review

    path id        : {path_id}
    path kind      : {path_kind}
    start          : ({start_x}, {start_y})
    end            : ({end_x}, {end_y})
    reported end as: {terminal_type}
    segments       : {segment_count} straight runs, {trace_length}px total length
    segment list   : {segments}

The path is drawn on the crop in **GREEN** — a semi-transparent halo with a 1px centreline, so
the black ink underneath stays visible. Blue and red crosshairs mark the start and end. Orange
boxes mark detected equipment.

## Measurements already taken (agreed — do not re-derive)

    ink coverage along the path    : {coverage:.2f}  (1.00 = unbroken ink)
    fraction centred on the ink    : {on_line:.2f}  (1.00 = path sits on the line)
    longest gap in the path        : {max_gap}px
    gaps >= 8px                    : {n_long_gaps}
    gap pixels NOT inside any known symbol : {unexplained_px}px
    why this crop was flagged      : {suspect_reason}
    ink shared with other paths    : {shared_px}px

A gap whose pixels lie **inside** a known symbol box is already accounted for and is NOT what
you are being asked about. Focus on the `{unexplained_px}px` the screen could not attribute.

## Rule to assess

**R1 — solid line.** Following the GREEN centreline across its unexplained gap(s): is there a
drawing element there (valve, instrument bubble, symbol body)? Or is it a text label, a leader
line, a dashed signal line, or blank paper?

Also check whether the path lies **on** the line or merely **beside** it: a path can follow the
correct route yet be offset a few pixels, sitting on the white margin next to the ink. Report
that as an R1 violation too. Blue and red crosshairs mark the start and end. Orange boxes mark
detected equipment.

## Answer format

Respond ONLY with JSON, no prose, no markdown fences:

    {{"verdict":"accept|reject|trim|unsure",
      "rules":["R1"],
      "confidence":0.0,
      "trim_from":null,
      "reason":"one sentence naming what you saw in the gap"}}

- `accept` — the gaps are drawn symbols sitting on a real pipe; keep the path.
- `reject` — the path follows something that is not a pipe (text, leader, dashed line, paper).
- `trim` — real pipe for part of its length, spurious after a point. Set `trim_from` to the
  approximate sheet pixel `[x,y]` where the real pipe ends.
- `unsure` — the crop does not show enough to decide. Use this rather than guessing.

`rules` lists the rule ids you found violated (empty list `[]` if accepted).
`confidence` is your own 0.0–1.0 confidence in the verdict.

**Never invent coordinates in `reason`, and never return geometry other than `trim_from`.**
You are judging the path, not redrawing it.

