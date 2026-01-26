# Annotation Guideline for P&ID Graph Digitization (graph_v1)

## 1. Goal
Create a **graph ground truth** from P&ID images:
- **Nodes** = symbols + topology helpers (crossing/ankle/border)
- **Edges** = connections (solid / non-solid)

The output must support:
- patch-based training (SAHI tiling)
- later merging of patch predictions into a full-plan graph

---

## 2. Coordinate and Bounding Box Rules
### 2.1 Coordinate system
- Origin: top-left of the image (0,0)
- x increases to the right, y increases downward
- Units: pixels

### 2.2 Bounding box format
Use **xywh**:
- x = left
- y = top
- w = width
- h = height

### 2.3 Bounding box tightness
- For **symbols**: bbox should tightly cover the visible symbol shape.
- Do not include large whitespace around symbols.
- Exact pixel-perfect edges are not required, but keep it consistent.

---

## 3. Node Types and When to Use Them
### 3.1 Symbol nodes
Create a node for each visible symbol:
- equipment_general
- tank_vessel
- pump_compressor
- valve
- instrumentation
- inlet_outlet
- arrow

Rules:
- One symbol = one node.
- If the same symbol appears in overlapping patches, it will be duplicated in patch files; merging will reconcile it later.

### 3.2 Topology helper nodes (mandatory for graph correctness)
These nodes exist to represent line structure, not equipment.

#### 3.2.1 crossing
Use when **two lines cross** and the crossing represents a junction or crossing point in the diagram topology.
- If lines cross visually and are intended to connect: add `crossing`.
- If the drawing uses a "jump/bridge" convention (one line hops over another without connection), do NOT add `crossing` as a junction.
  - If unsure, follow your company drafting convention consistently.

#### 3.2.2 ankle
Use when a line makes a **T-junction** or a **branch point** (one line splits into two or more).
- Place the ankle node at the branch point.
- Connect edges from ankle to each connected symbol/crossing/border.

#### 3.2.3 border (patch-only)
Use only when annotating a **patch** (tiling is_patch=true).
Create a `border` node when a line segment **exits the patch boundary**.

Purpose:
- Allows the model to learn continuity across patches.
- Enables graph stitching later.

Placement:
- Put the border node bbox centered at the intersection point where the line hits the patch boundary.
- Use a small fixed bbox size (recommended 10x10 pixels) unless you have a better standard.

---

## 4. Edge Annotation Rules
### 4.1 What counts as an edge
Create an edge when there is a direct connection between two nodes:
- symbol ↔ symbol
- symbol ↔ ankle
- symbol ↔ crossing
- ankle ↔ crossing
- (patch mode) symbol/ankle/crossing ↔ border

### 4.2 Edge type
- `solid`: continuous process line
- `non_solid`: dashed / dotted / non-continuous line

If a line style changes visually, follow this priority:
1) If the diagram clearly indicates dashed meaning, keep `non_solid`.
2) If uncertain, default to `solid` and note in provenance.notes.

### 4.3 Direction
Set `directed=false` by default.

Only set directed=true if:
- you have reliable arrow semantics and you plan to enforce direction consistently later

Do not mix conventions within a dataset.

### 4.4 Disallowed edges
- No self-loops (src == dst).
- No “shortcuts” that skip topology nodes.
  - Example: if a symbol connects to an ankle, and ankle connects to another symbol, do NOT also connect symbol-to-symbol directly.

---

## 5. Patch (Tile) Annotation Workflow (SAHI-Compatible)
### 5.1 Use SAHI to generate tiles
Recommended settings:
- overlap: **>= 50%** in x and y
- record for each tile:
  - offset_x, offset_y, tile_width, tile_height, overlap_x, overlap_y

### 5.2 Create patch graph files
For each tile (patch):
1) Crop the image for the tile.
2) Copy all nodes whose **global bbox intersects** the tile.
3) Convert global bbox → local patch bbox:
   - local_x = global_x - offset_x
   - local_y = global_y - offset_y
4) Add `border` nodes for every connection that exits the patch boundary.
5) Include edges:
   - between nodes that appear in the patch
   - plus edges from internal nodes to border nodes when the connection exits

### 5.3 Border node consistency rule
For the same physical line crossing the patch boundary:
- Only one border node per crossing point per patch boundary.
- If multiple lines exit at different points: multiple border nodes.

---

## 6. Minimal Provenance Requirements
For every node and edge:
- annotated_by
- annotated_at
- source = manual (for ground truth)

Use provenance.notes for ambiguity:
- jump/bridge uncertainty
- dashed vs solid uncertainty
- symbol class uncertainty

---

## 7. Quality Checks (Must Pass)
### 7.1 Graph integrity
- Every edge endpoint exists as a node in the same file.
- No self-loops.
- No isolated topology nodes unless they are border nodes.

### 7.2 Topology completeness
- All junctions are represented by ankle/crossing nodes (no implicit branches).
- Patch border exits are represented by border nodes (patch files only).

### 7.3 Consistency across annotators
- Use the same rule for jump/bridge crossings everywhere.
- Use the same bbox tightness style everywhere.
- Use the same edge direction setting everywhere.

---

## 8. Recommended Naming
Node IDs:
- Use stable IDs for full-plan annotation, e.g. `N000123`
- For patches you may append tile id, e.g. `N000123_T05R02C03` (optional)

Edge IDs:
- `E000001` etc.

---

## 9. Scope Control (Do Not Add Yet)
Do not annotate these in v1:
- line numbers / tags as mandatory
- instrument signal types
- equipment attributes (materials, sizes)
- semantic flow direction

Keep v1 focused on **topology correctness**.
