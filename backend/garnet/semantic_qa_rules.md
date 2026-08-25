# Chemical Engineering Process-Design Rules — Semantic Graph QA

This document is domain context injected into the semantic-QA stage of the GARNET
pipeline. It is read at runtime by `semantic_graph_qa.py` and supplied to a language
model that flags *semantic* implausibilities in an already-assembled process graph.

## Scope and authority

- **You are reasoning over a graph, not over pixels.** The geometry pipeline
  (`trace_graph_builder`) already decided *where* lines run. Your job is only to
  flag where the *meaning* of the graph is physically or conventionally implausible.
- **Geometry is authoritative.** Do not claim a line does not exist; only claim that
  its endpoints or type violate a process-design rule.
- **Reference only existing node and edge ids.** Never invent a node or edge. If the
  evidence is ambiguous, prefer `severity: "low"` and say so in the message.
- A node or edge you cannot confidently map to a rule is **not** an anomaly. Absence
  of evidence is not evidence of absence.

## Node types you will encounter

- `pump_compressor` / `pump` / `compressor` — rotating machinery
- `tank_vessel` / `vessel` — drums, columns, reactors, tanks
- `valve` — in-line blocking/regulating element (control, isolation, check, relief)
- `instrumentation` / `instrument` — sensor bubble, indicator, controller, switch
- `equipment_general` / `equipment` — generic or unrecognised unit
- `junction` / `tee_junction` — pipe branch or cross; carries no process function
- `source`, `inlet_outlet`, `connection`, `page_connection`, `utility_connection` —
  boundary terminals (feed, product, off-page, utility header)
- `dead_end` / `terminal` — trace that stops without a recognised terminal

## Edge semantics

- `line_style: "solid"` — process / utility piping (material or energy flow).
- `line_style: "non_solid"` (dashed) — **signal** lines: electrical, pneumatic, or
  control wiring. A non-solid edge is never a material stream.
- `directed` — when present, it indicates flow direction (flow arrows).

---

## 1. Tag nomenclature and abbreviations

Tags are the primary semantic signal. Normalise before judging: strip whitespace,
treat `0`/`O`, `1`/`l`/`I` as ambiguous unless context disambiguates.

Common ISA-style tag prefixes:

| Prefix | Meaning | Prefix | Meaning |
|--------|---------|--------|---------|
| `P-` / `PU-` | pump | `V-` / `D-` | vessel / drum |
| `T-` | tank | `E-` / `HX-` | exchanger |
| `C-` / `K-` | compressor | `F-` | furnace / filter (context) |
| `R-` | reactor | `PR-` | pressure-reducing station |
| `FCV` / `FV` | flow control valve | `PCV` / `PV` | pressure control valve |
| `TCV` / `TV` | temperature control valve | `LCV` / `LV` | level control valve |
| `XV` / `MOV` / `HV` | on-off / motor / hand valve | `PSV` / `PRV` / `RV` | relief valve |
| `SDV` | shutdown valve | `BDV` | blowdown valve |
| `FT` / `PT` / `TT` / `LT` | flow / pressure / temp / level transmitter | | |
| `FIC` / `PIC` / `TIC` / `LIC` | indicating controller (same variables) | | |
| `FS` / `PS` / `TS` / `LS` | switch | `FE` / `PE` / `TE` / `LE` | primary element |
| `PI` / `TI` / `LI` / `FI` | indicator | `FA` / `PA` / `TA` / `LA` | alarm |
| `HSV` / `HIC` | hand switch / hand indicator-controller | | |

Line (pipe) numbers encode service, e.g. `50-PG-101-2"-CS` (size-service-sequence-spec).
The service segment is what matters for connectivity (`PG` = process gas, `CW` = cooling
water, `IA` = instrument air, `LS` = low-pressure steam, etc.).

## 2. Physical connectivity patterns (process)

These are *expected* configurations. Flag a **violation**, not the mere absence of the
pattern — a missing piece may simply be on another sheet.

- **Pump / compressor** requires both a suction (inlet) and a discharge (outlet)
  connection. A `pump_compressor` with degree < 2, or with only inlets, is a strong
  anomaly. A pump feeding a pump directly (degree-2 chain of pumps, no vessel between)
  is usually wrong.
- **Vessel / tank** normally has at least one inlet and one outlet nozzle, except a
  drain/vent-only accumulator. A `tank_vessel` that is isolated (degree 0) or that only
  ever *sinks* material (all edges into it, none out) is suspicious.
- **Valve** is an in-line element: it must sit between two other nodes on a process line
  (degree >= 2), unless it is a relief/blowdown/drain valve whose downstream side goes
  to atmosphere, flare, or a closed drain header. A `valve` as a dead-end *terminal* is
  anomalous except in those cases.
- **Relief valve (`PSV`/`PRV`/`RV`)** inlet connects to the *protected* equipment or its
  upstream piping; outlet discharges to flare / atmosphere / a relief header — never
  back into the same process line, and never with a solid line *into* the relief side.
- **Check valve** is directional; a solid directed edge through it should point in the
  process-flow direction.
- **Equipment reachability** — an `equipment_*` node with degree 0 (connected to
  nothing) is a strong anomaly; it is either a detection error or an isolated unit.
- **Signal/process mix-ups** — a `non_solid` edge into or out of process equipment
  (pump, vessel, exchanger) usually indicates a mis-classified line or an instrument
  whose tap was treated as a process connection.

## 3. Control and instrumentation structure

- **Instrumentation interfaces through signal lines, not process lines.** A `solid`
  edge between two `instrumentation` nodes, or a `solid` edge from `instrumentation`
  into `equipment_*` (unless it is an in-line instrument in the pipe), is anomalous.
- **Control loop chain** — a functioning loop links
  *primary element / transmitter → indicator/controller → final control element*
  (e.g. `FT` → `FIC` → `FCV`). A transmitter with no downstream controller, or a
  control valve with no upstream controller/signal source, is worth flagging.
- **Signal directionality** — signal flows from sensor toward actuator. A directed
  `non_solid` edge pointing the wrong way (controller → sensor, or valve → controller)
  is suspicious.
- **Control valve placement** — a control valve belongs *on* the process line it
  regulates (solid edges through it), with a *separate* signal line from its controller.
  It should not appear as an isolated bubble attached only by signal lines.

---

## Output discipline

For each anomaly, state:
- `rule_id` — the rule that fired (short, e.g. `pump_missing_discharge`);
- `severity` — `high` (physically impossible / strong violation), `medium`
  (improbable), `low` (worth a human glance);
- the **existing** `node_id` and/or `edge_id` it concerns;
- a one-sentence `message` naming the affected tags and why it looks wrong.

Flag fewer, higher-confidence anomalies over many weak ones. A clean graph should
yield an empty or near-empty list.
