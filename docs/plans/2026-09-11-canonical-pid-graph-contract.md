# Canonical P&ID graph contract and Phase 1 acceptance plan

This plan defines the information the P&ID extractor must eventually preserve
for line lists, test-package candidates, process descriptions, and HAZOP input.
It is a contract and acceptance reference; it does not add a new runtime graph
format in this phase.

## Contract vocabulary

`drawing` identifies the source document and coordinate frame:

```json
{
  "drawing_id": "P-101",
  "revision": "B",
  "content_sha256": "...",
  "pixel_dimensions": {"width": 4963, "height": 3509},
  "coordinate_system": "image_pixel_origin_top_left"
}
```

`equipment` and `port` are separate entities. A port is the attachable point;
the equipment tag is the stable process object. `segment` retains an ordered
pixel polyline and may reference a port, junction, terminal, or connector at
each endpoint. Segment IDs must remain distinct when routes are parallel,
including a bypass with the same endpoints as a main route.

`line` is a canonical process-line identity. `line_number_occurrence` is an
OCR observation and must retain its text, normalized text, bounding box,
confidence, and association method. Multiple occurrences may support one line;
one ambiguous occurrence must not silently create a canonical line.

`instrument` relationships are typed and evidence-backed: `measures`,
`controls`, and `actuates` are different from merely being spatially attached to
a segment. `boundary` and `test_package` entities record included members,
cut points, isolation elements, exclusions, and review state.

Each entity and relationship has `provenance` (source artifact and geometry),
`confidence`, and `state`. Use `observed`, `inferred`, `reviewed`, `unresolved`,
or `rejected`; unresolved evidence is exportable for review but is not released
as accepted process truth.

## Connectivity and direction

The physical graph is a directed multigraph only in views that have direction
evidence. Its base relation is undirected `connects_to`; flow direction is a
separate attribute with values `forward`, `reverse`, `bidirectional`, `unknown`,
or `conflicting`. Arrow geometry, connector direction, and a reviewed override
are evidence sources. Trace walking order is never evidence of process flow.

## Phase 1 acceptance fixtures

The synthetic fixtures under `backend/tests/fixtures/canonical_pid_graph/`
cover these invariants:

- a main route and a bypass with identical endpoints remain two segments;
- a crossing without a junction marker is unresolved/non-connecting, while a
  marked tee is a shared junction with three incident segments;
- repeated OCR line labels remain occurrences of one drawing-scoped line, and
  the same textual key on two drawings remains sheet-scoped until a connector
  match is evidenced;
- missing line identity, missing direction, conflicting direction, and weak
  connector matches remain explicit uncertainty states;
- every segment has pixel coordinates and source drawing identity.

The test checks structure and invariants only. It does not assert that current
detectors can produce the fixture automatically.

## Benchmark inventory and limits

Candidate read-only benchmark material currently present in the repository:

| Set | Available material | Use and limitation |
|---|---|---|
| `backend/test/ppcl` | 9 JPEG drawings, paired JSON for most cases; examples include `Test-00001.jpg` (3961x3224) and `Test-00008.jpg` (4963x3509) | Good single-sheet geometry/route cases; annotation semantics and revision metadata are not a complete gold graph. |
| `backend/test/pttep` | 5 PNG drawings, including `FUCPP_41.png` (2481x1754) | Useful varied raster inputs; connector and line truth are not fully documented. |
| `backend/datasets/iso_inputs/PTT tank` | 147 PNG drawings, up to `iso_Page1.png` (9917x7017) plus JSON annotations for a subset | Broad scale and multi-sheet candidate; annotation coverage and cross-sheet ground truth need sampling. |

No generated artifacts or model outputs are part of this inventory. Before
benchmark thresholds are set, select a small manually reviewed subset spanning
tees, crossings, bypasses, repeated labels, and off-page connectors. Measure
route preservation, false connections, line assignment, uncertainty retention,
and cross-sheet continuity separately.

## Scope status

Phase 1 defines this contract, fixtures, and benchmark inventory. Phase 2 will
fix route deduplication and unsupported line assignment. Phase 3 will make
review corrections and downstream artifacts share one graph revision. Phases
4–9 (identity strengthening, flow inference, multi-sheet contract migration,
review gates, boundaries/package views, and end-to-end release validation) are
deferred and are not claimed by this document.
