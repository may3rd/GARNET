# Full revised plan with the added suggestions integrated

The comment is correct. The revised pipeline was stronger than the original, but it still needed four upgrades:

1. **morphological sealing before skeletonization**
2. **explicit flow-arrow detection and edge direction assignment**
3. **graph-native QA using NetworkX algorithms**
4. **polyline simplification before export**

Those are not optional refinements. They materially improve robustness.

---

# 0. Governing rule

All early outputs are **provisional evidence**.

Nothing becomes graph truth until it survives:

1. geometric refinement
2. association scoring
3. topology validation
4. graph QA

The pipeline is not “detect objects then connect them.”
It is:

> **extract evidence → build pipe geometry → derive topology → attach semantics → validate graph**

---

# 1. System objective

Convert a raster/PDF P&ID into a structured, reviewable, confidence-scored graph with:

## Nodes

- equipment
- valves
- instruments
- junctions
- endpoints
- off-page connectors
- optional arrow nodes

## Edges

- pipe segments
- logical continuations
- branch connections
- directed flow edges where evidence exists

## Attributes

- geometry
- OCR text
- line numbers
- classes
- confidence bundle
- provenance
- review status

---

# 2. Core design principles

## Principle 1 — Geometry first, semantics second

Pipe geometry is the backbone.
Text and symbol detection decorate and constrain that backbone.

## Principle 2 — No permanent early masking

Never destroy master evidence early.
Create task-specific derivative views only.

## Principle 3 — Skeletons are fragile

Skeletonization is not a neutral conversion.
It amplifies binary mask defects.
So pre-skeleton cleanup is mandatory.

## Principle 4 — Association must be multi-signal

Nearest distance alone is not enough.

## Principle 5 — Graph QA is part of extraction, not an afterthought

The graph is the validation surface.

## Principle 6 — Export geometry must be compressed

Pixel-by-pixel edge paths are too large and too noisy for practical downstream use.

---

# 3. Full revised architecture

---

## Phase A — Evidence extraction

This phase extracts candidate objects and text without forcing early topological assumptions.

### A1. Input normalization

Create parallel working representations:

- original raster
- grayscale
- multiple binarized views
- OCR-friendly enhanced image
- line-enhanced image
- high-resolution tiles
- lower-resolution full sheet
- optional denoised variants

### A2. Full-page OCR and text localization

Run full-page text discovery to obtain:

- text boxes
- raw OCR
- rough text class
- confidence
- rotation
- legibility

This stage is for **coverage**, not final truth.

### A3. Crop OCR refinement

Run crop OCR only on:

- low-confidence text
- very small text
- rotated text
- overlapping text
- suspected tags near equipment or lines
- unresolved OCR candidates

This stage is for **local accuracy**.

### A4. Small-object detection

Use SAHI + YOLO for small graph-relevant objects:

- valves
- instruments
- reducers/fittings if detectable
- off-page connectors
- **flow arrows**
- special markers

This is an important addition.
Flow arrows are now a required class, not optional clutter.

### A5. Equipment detection

Use multi-scale equipment detection:

- tiled medium/high-resolution branch
- lower-resolution whole-sheet branch
- result fusion

Target classes:

- pumps
- vessels
- exchangers
- tanks
- drums
- filters
- columns
- other major equipment families

### A6. Equipment refinement

Refine equipment candidates using:

- segmentation
- contour extraction
- morphology
- optional template support
- VLM only for local semantic ambiguity

Outputs:

- refined boundary
- refined bbox
- center
- contour
- candidate nozzle sides

---

## Phase B — Geometry engine

This is the center of gravity.

---

### B1. Pipe mask generation

Generate a binary pipe mask from the most line-faithful views.

This mask is the key geometric artifact.
Its quality determines whether the skeleton becomes useful or corrupted.

Outputs:

- binary pipe mask
- optional confidence map
- optional pipe-region metadata

---

### B1.5. Morphological sealing before skeletonization

This is a new required stage.

Apply morphology **before** skeletonization:

- closing (dilation then erosion) to seal tiny pinholes and narrow gaps
- optional opening in limited cases to remove isolated specks
- component filtering for obvious non-pipe noise

Why:

- pinholes can create false loops
- edge bumps can create spurs
- tiny breaks can split true continuity

This stage reduces the burden on post-skeleton cleanup by preventing avoidable artifacts from entering the skeletonizer.

Constraint:

- make this operation conservative
- do not over-close genuine gaps or small intentional separations

Outputs:

- sealed pipe mask
- morphology audit metrics

---

### B2. Skeleton generation

Convert the sealed pipe mask into a 1-pixel skeleton.

Use thinning/skeletonization only after B1.5.

Outputs:

- raw skeleton
- skeleton pixel map

Risk:

- this step is still fragile
- never trust raw skeleton output without cleanup

---

### B3. Skeleton node detection

Classify skeleton pixels by local degree:

- degree 1 = endpoint
- degree 2 = normal pipe path
- degree ≥3 = candidate junction

This produces node hypotheses, not final graph nodes.

Outputs:

- endpoint pixels
- junction candidate pixels
- normal path pixels

---

### B4. Crossing vs junction disambiguation

Mandatory stage.

Do not assume a crossing is a connection.

Disambiguate using:

- junction symbol evidence
- local intersection geometry
- continuity analysis
- nearby detected markers
- optional local crop reasoning

Outputs:

- confirmed junctions
- non-connecting crossings
- unresolved crossing candidates

---

### B5. Skeleton cleanup

Now clean the raw skeleton using topology-aware rules:

- spur pruning
- tiny branch removal
- short-fragment filtering
- controlled gap bridging
- branch-length thresholds
- neighborhood consistency checks

Important:
B1.5 reduces noise before skeletonization.
B5 cleans what still survives after skeletonization.

Both are needed.

Outputs:

- cleaned skeleton
- removed-artifact log

---

### B6. Node clustering

Collapse raw node pixels into graph-node candidates.

Use clustering such as:

- DBSCAN
- connected neighborhood clustering
- junction region consolidation

Outputs:

- graph node candidates
- node centroids
- clustered node regions

---

### B7. Edge tracing

Trace edge paths between clustered nodes along the cleaned skeleton.

This is actual pipe-edge extraction.

Outputs:

- traced edge polylines
- raw adjacency structure
- edge confidence seeds

---

## Phase C — Attachment and semantic association engine

This phase binds symbols and text to the geometry backbone.

---

### C1. Equipment-to-pipe attachment

Replace simple nearest-neighbor logic with multi-signal scoring.

Signals:

- nearest skeleton distance
- boundary-band intersection
- local direction consistency
- candidate nozzle-side alignment
- edge plausibility
- competition penalty when multiple pipes are nearby

Decision policy:

- high score → attach
- medium score → ambiguous set
- low score → unresolved

Outputs:

- equipment-to-edge links
- attachment confidence
- ambiguous attachment queue

---

### C2. Inline-object-to-pipe association

Associate valves, instruments, reducers, and similar inline objects.

Rules:

- require overlap or centerline continuity
- require local axis consistency
- split edge only when inline-node confidence is sufficient

Outputs:

- inline-node attachments
- inline edge splits
- unresolved inline associations

---

### C3. Flow-arrow association and edge direction assignment

This is a new mandatory stage.

Detect flow arrows from A4 and assign them to traced edges.

For each arrow:

- detect arrow bbox/mask
- estimate orientation vector
- associate to nearest compatible edge segment
- verify alignment with local pipe direction
- convert local vector into edge directionality

Then:

- assign source → target direction to affected edge
- propagate direction through locally unambiguous sequences where justified
- keep edges undirected when direction evidence is absent

Important:
The graph is no longer assumed to be purely undirected.

Outputs:

- directed edge assignments
- direction confidence
- unresolved arrow-edge matches

This is essential for downstream path queries.

---

### C4. Text association

Separate from OCR itself.

Associate text to:

- equipment
- line segments
- valves/instruments
- utility/service paths
- off-page connectors where applicable

Use:

- distance
- orientation
- line alignment
- class priors
- local context
- crop re-check if needed

Outputs:

- text-to-equipment links
- text-to-edge links
- unresolved text links

---

### C5. Off-page connector handling

Represent off-page connectors explicitly as graph nodes.

Store:

- connector id
- label
- page reference if known
- attachment edge
- confidence

This enables later multi-sheet merge.

---

## Phase D — Graph engine

---

### D1. Graph construction

Build the graph in NetworkX or equivalent.

Support:

- undirected structure where direction is unknown
- directed edges where flow-arrow evidence exists
- mixed-mode graph representation if needed

Recommended internal model:

- node table
- edge table
- NetworkX graph/DiGraph/MultiGraph wrapper

---

### D2. Graph schema

Each node should carry:

- id
- node type
- class
- geometry
- confidence bundle
- source provenance
- review status

Each edge should carry:

- source
- target
- polyline
- edge class
- direction state
- direction confidence
- associated text ids
- line number if assigned
- review status

---

### D3. Graph-native QA primitives

This is an upgrade to the old QA stage.

Use graph algorithms directly rather than writing everything as spatial heuristics.

Examples:

- connected components
- degree analysis
- path existence checks
- articulation points
- terminal-node checks
- isolated-subgraph detection

Examples of cheap QA:

- isolated equipment = component size anomaly or degree anomaly
- orphan branch = endpoint without terminal semantics
- disconnected subsystem = unexpected component fragmentation

This makes QA faster, simpler, and more reliable.

---

### D4. Polyline simplification before export

This is now required.

Raw traced polylines can contain thousands of pixel coordinates for visually straight pipes.

Before export:

- apply Ramer-Douglas-Peucker or equivalent
- preserve routing shape
- preserve topological endpoints
- preserve branch/intersection anchors
- keep tolerance configurable

Benefits:

- much smaller JSON/GraphML payloads
- cleaner geometry
- easier downstream rendering and queries

Outputs:

- simplified edge polylines
- compression metrics

---

## Phase E — QA, recovery, and review loop

This is what makes the pipeline operational rather than merely sequential.

---

### E1. Automatic anomaly checks

Run graph and geometry QA checks for:

- isolated equipment
- zero-attachment equipment
- suspicious open ends
- duplicate overlapping nodes
- false tiny components
- crossing/junction conflicts
- conflicting text associations
- contradictory edge directions
- impossible local subgraphs
- terminal-free components
- unexpected disconnected islands

Use graph-native checks first, spatial re-checks second.

---

### E2. Recovery loop

When anomalies appear, do targeted reprocessing:

- re-run crop OCR on suspicious text
- re-run local attachment scoring around isolated equipment
- re-check crossing/junction classification
- re-check flow-arrow assignment
- re-evaluate morphology tolerance around damaged pipe regions
- re-run local segmentation/refinement on equipment boundary

This makes the system iterative where it matters.

---

### E3. Human review boundary

Manual review should handle only unresolved ambiguity, such as:

- ambiguous crossings
- competing attachment candidates
- contradictory OCR near critical tags
- uncertain direction assignments
- inconsistent off-page connectors

Do not send the entire sheet to review.

---

# 4. Stage-by-stage implementation plan

## Stage 1 — Input normalization and scale pyramid

Deliverables:

- normalized image bundle
- tile scheme
- preprocessing metadata

## Stage 2 — Full-page OCR discovery

Deliverables:

- text-region table
- raw OCR hypotheses
- rough classes

## Stage 3 — Crop OCR refinement

Deliverables:

- refined OCR table
- OCR merge output
- unresolved OCR queue

## Stage 4 — Small-object, flow-arrow, and equipment detection

Deliverables:

- small-object candidate table
- flow-arrow candidate table
- equipment candidate table
- off-page connector table

## Stage 5 — Pipe mask generation

Deliverables:

- binary pipe mask
- pipe confidence map

## Stage 6 — Morphological sealing

Deliverables:

- sealed pipe mask
- morphology audit

## Stage 7 — Skeleton generation

Deliverables:

- raw skeleton
- skeleton QA counters

## Stage 8 — Skeleton node detection and clustering

Deliverables:

- endpoints
- junction candidates
- clustered node set

## Stage 9 — Crossing/junction disambiguation

Deliverables:

- confirmed junctions
- non-junction crossings
- unresolved crossing queue

## Stage 10 — Skeleton cleanup and edge tracing

Deliverables:

- cleaned skeleton
- traced edge polylines
- raw graph adjacency

## Stage 11 — Attachment and association

Deliverables:

- equipment-to-edge links
- inline-object links
- text links
- flow-arrow-to-edge links
- unresolved association queues

## Stage 12 — Graph assembly and simplification

Deliverables:

- graph object
- simplified edge polylines
- JSON/GraphML export
- compression metrics

## Stage 13 — QA and recovery loop

Deliverables:

- anomaly report
- retry queue
- manual-review queue

---

# 5. Revised role of models and methods

## OCR / text

- full-page OCR for discovery
- crop OCR for refinement
- VLM only for difficult local ambiguities

## Small objects

- SAHI + YOLO

## Equipment

- multi-scale detector
- segmentation/contours for refinement

## Pipe geometry

- image processing
- morphology
- skeletonization
- graph extraction

## Directionality

- flow-arrow detection
- local vector assignment
- directional edge propagation where justified

## Topology and QA

- deterministic graph logic
- NetworkX checks
- targeted recovery loops

---

# 6. Revised quick-win roadmap

## V1 — Geometry-first local graph

Build:

- pipe mask
- morphology sealing
- skeleton
- node detection
- edge tracing
- equipment detection
- basic equipment attachment

Practical output:

- local connectivity explorer
- open-end detector
- isolated-equipment detector

## V2 — Semantics and direction

Add:

- OCR association
- line-number binding
- flow-arrow detection
- directed-edge assignment
- graph-native QA

Practical output:

- directed path queries
- line-tagged local subsystems
- anomaly dashboard

## V3 — Multi-sheet and review system

Add:

- off-page connector merge
- recovery automation
- review queue UI
- export and subsystem extraction

---

# 7. Ground-truth reference plan

Use this as the definitive version:

1. Normalize the drawing and create parallel working views
2. Run full-page OCR for sheet-wide text discovery
3. Run crop OCR only on difficult or ambiguous regions
4. Detect small objects, **flow arrows**, equipment, and off-page connectors
5. Generate a binary pipe mask
6. Apply **morphological closing/sealing** before skeletonization
7. Convert the sealed pipe mask into a 1-pixel skeleton
8. Detect endpoints and candidate junctions from skeleton degree
9. Cluster raw node pixels into graph-node candidates
10. Resolve crossing vs true junction explicitly
11. Clean the skeleton and trace edges between nodes
12. Attach equipment and inline objects using **multi-signal scoring**, not nearest distance alone
13. Associate **flow arrows** to edges and assign direction where supported
14. Associate text to equipment, edges, and inline objects
15. Build a confidence-scored graph in NetworkX or equivalent
16. Run graph-native anomaly checks using connected components, degree checks, and path checks
17. Simplify edge polylines before export using **Ramer-Douglas-Peucker** or equivalent
18. Run targeted recovery loops for suspicious regions
19. Send only unresolved ambiguity to human review

---

# 8. Final judgment

This version is materially better than the prior revised plan because it fixes the remaining hidden weak points:

- it protects the skeletonizer from bad masks
- it stops treating the graph as implicitly undirected
- it uses graph algorithms as first-class QA tools
- it makes exported geometry practical rather than bloated

This is now a credible **engineering extraction system design**, not just a CV pipeline with graph vocabulary.
