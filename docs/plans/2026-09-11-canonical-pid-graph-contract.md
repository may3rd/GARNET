# Canonical P&ID graph contract and acceptance plan

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

## Phase 4 identity projection

Phase 4 implements the identity portion of this contract as additive fields in
the existing Stage 7 graph, graph-v1 export, and Stage 10 process exports:

- equipment and ports receive drawing-scoped IDs, with each port retaining its
  traced pixel position and the graph node that supplied it;
- edge endpoints reference equipment and port IDs without changing the traced
  polyline;
- canonical process lines are separate from OCR occurrences, so two reviewed
  labels with the same normalized number can support one drawing-scoped line;
- physical inline objects receive drawing-scoped identities and ordered route
  occurrences with pixel distance along the edge;
- instruments are canonicalized separately from their OCR occurrences;
- `connects_to`, `has_port`, `has_inline_object`, `measures`, `controls`, and
  `actuates` relationships are emitted only when their supporting evidence is
  present. A spatial instrument attachment without functional evidence remains
  an unresolved `instrument_association`.

Legacy graph-v1 node, edge, attachment, and line-number fields remain in place.
The new catalogs and relationships are projections over the same reviewed graph
revision and therefore follow the existing Stage 9 invalidation rules.

## Phase 5 flow-direction projection

Phase 5 implements flow as evidence attached to the existing physical graph:

- Stage 6 normalizes explicit arrow vectors, tip-tail geometry, and conservative
  raster-crop orientation, then compares each arrow with the local route tangent;
- each edge records `forward`, `reverse`, `bidirectional`, `unknown`, or
  `conflicting`, plus confidence, evidence, and review state;
- route splits localize arrow observations and recompute direction, while
  duplicate routes merge their arrow evidence before resolving agreement or
  conflict;
- Stage 8 promotes unknown and conflicting direction to review items, and Stage
  9 applies audited human `set_flow_direction` decisions;
- graph-v1 adds flow-oriented endpoints only for resolved forward/reverse edges
  and keeps legacy endpoint fields intact;
- Stage 10 retains per-edge direction. Multi-edge line aggregates remain
  conservative because `forward` and `reverse` are relative to each edge's
  stored coordinate order.

Physical connectivity remains undirected. A trace's source-to-target walking
order is never sufficient evidence for process flow.

## Phase 6 multi-sheet projection

Phase 6 preserves the reviewed graph contract when individual drawings are
assembled into a system graph:

- the legacy graph-v2 merge summaries remain available for existing consumers;
- the additive `combined_graph` carries drawing metadata and qualified nodes,
  pixel-route edges, equipment, ports, lines, inline objects, instruments, and
  relationships from every input sheet;
- projected IDs include entity type and drawing scope, preventing equal local
  IDs on separate sheets or in separate catalogs from colliding;
- each uniquely resolved connector pair creates one typed `cross_sheet_continues`
  relationship between explicit connector entities, with match evidence and
  source provenance;
- unmatched, rejected, non-reciprocal, or ambiguous connectors remain merge
  issues and do not create physical continuity;
- boundary flow is recorded separately as incoming, outgoing, bidirectional,
  or unknown based on the local edge's reviewed direction and the connector's
  source or target terminal.

The combined projection is deterministic under input sheet reordering and the
API persists it as strict JSON. Embedded per-sheet graph-v1 payloads remain in
the system artifact for compatibility and audit.

## Phase 7 reviewed topology and release

Phase 7 makes review completion an explicit condition of process-data release:

- Stage 8 classifies review items as release-blocking or informational and
  promotes existing node, edge, trace, endpoint, and pixel-route evidence into
  decision-friendly targets;
- Stage 9 applies validated topology decisions for node merging, endpoint
  reconnection, route splitting, edge deletion, and node reclassification;
- topology mutations are atomic, route splits preserve ordered pixel geometry,
  and physical attachment observations remain on one resulting segment;
- every applied mutation records deterministic before/after evidence, affected
  IDs, the source review item, the decision, and reviewer-supplied context;
- missing, deferred, invalid, duplicate, orphaned, and unsupported decisions
  remain unresolved rather than being accepted by assumption;
- `stage9_release_gate.json` controls access to the reviewed graph and its
  process-facing derivatives while leaving review items, resolutions, and audit
  artifacts available;
- multi-sheet release additionally requires an explicit connector-review
  revision and no remaining merge issues. Released cross-sheet relationships
  carry the connector review revision and reviewer provenance.

The Stage 7 source graph remains immutable. Stage 9 writes a corrected graph
revision and regenerates graph-v1 and later projections from that revision.

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

Phases 1–3 established the contract, route preservation, and authoritative
review revision. Phase 4 implements drawing-scoped engineering identities.
Phase 5 adds evidence-based flow direction and its review path. Phase 6 adds a
qualified system graph and explicit cross-sheet continuity. Phase 7 implements
reviewed topology changes, audit history, and release gates. Phases 8–9
(boundaries/package views and end-to-end release validation) remain deferred.
