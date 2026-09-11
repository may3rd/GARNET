# GARNET P&ID digitizing roadmap

This document is the roadmap for turning P&ID drawings into a reviewable,
traceable connectivity graph. It describes the target contract and the order in
which capabilities should be delivered. It does not imply that deferred
capabilities already exist in the runner.

## Current runner

The live `pid_extractor.py` sequence is:

`1 input normalization -> 2 OCR discovery -> 4 object detection/fusion -> 5
pipe mask -> 5b pipe trace -> 6 trace associations -> 7 geometric graph
assembly -> 7c page connector labeling -> 7b graph export -> 8 graph QA -> 9
review decisions -> 10 process exports -> 11 connection overlay`.

Stage 3 is external HITL input. The stage numbers are intentionally sparse;
they are public artifact names and should not be renumbered casually. Older
documents that call graph normalization or review packaging Stage 12 or 13 are
historical design notes, not the current runner order.

Stages 1–3 of the roadmap in the delivery plan are the first implementation
slice: define the contract and fixtures, preserve distinct routes, and make
review corrections produce a consistent graph revision. Later roadmap phases
remain deferred until that foundation is accepted.

## Target graph contract

The canonical data model has drawing/revision identity, pixel geometry, physical
connectivity, semantics, evidence, and review state. A physical network is a
multigraph: two routes with the same endpoints may be separate bypasses.

- A drawing has a stable `drawing_id`, source filename, revision (when present),
  content hash, pixel width and height, and a declared coordinate system.
- Equipment has a stable drawing-scoped identity and explicit ports. A port has
  a pixel position and retains the detection/OCR evidence that led to it.
- A physical segment has a stable ID, drawing ID, ordered pixel polyline,
  endpoint references, and any inline objects encountered along the route.
- A canonical line identity is distinct from each OCR line-number occurrence.
  Occurrences retain text, bounding box, confidence, and association evidence.
- Relationships are typed. At minimum: `connects_to`, `branches_from`,
  `cross_sheet_continues`, `has_inline_object`, `measures`, `controls`, and
  `actuates`.
- Physical connectivity and process-flow direction are separate facts. Flow is
  `forward`, `reverse`, `bidirectional`, `unknown`, or `conflicting`, with
  evidence and review state. Walking order never establishes process flow.
- Boundaries and test packages are graph-linked entities with members, cut
  points, isolation elements, exclusions, and review state. They are deferred
  until the graph foundation is stable.
- Every promoted fact carries provenance and state such as `observed`,
  `inferred`, `reviewed`, `unresolved`, or `rejected`. Unsupported guesses must
  remain candidates or unresolved evidence.

The existing graph-v1 payload remains the compatibility surface during this
roadmap. The richer contract is additive design guidance until an implementation
slice explicitly extends and versions that payload.

## Delivery phases

1. Define the canonical contract, acceptance fixtures, benchmark inventory, and
   versioning rules.
2. Preserve route identity: compare complete routes before deduplication and
   keep parallel/bypass paths; keep unsupported line assignments unresolved.
3. Make review corrections authoritative and regenerate all dependent artifacts
   from one corrected graph revision.
4. Strengthen equipment/port, line, inline-object, and instrument identities.
5. Infer flow direction from arrow evidence and geometry, retaining unknown and
   conflicting cases.
6. Preserve the contract across sheets and represent matched connectors as
   explicit cross-sheet relationships.
7. Complete topology review operations, audit history, and release gates.
8. Add process boundaries, test-package views, and LLM-oriented projections.
9. Validate end to end on a representative benchmark and version the export.

Phases 1–3 are the active work slice. Phases 4–9 are planned work only.

## Acceptance principle

An output is useful downstream only when a reviewer can locate every promoted
route and relationship on the source drawing, distinguish observed evidence
from inference, and see unresolved ambiguity instead of having it silently
converted into graph truth.
