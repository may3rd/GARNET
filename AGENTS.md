# Agent instructions (scope: this directory and subdirectories)

## Scope and layout
- This AGENTS.md applies to the repository root and everything below it.
- Keep root guidance global. Put Python pipeline specifics in `backend/garnet/AGENTS.md` and Gemini adapter specifics in `backend/gemini_detector/AGENTS.md`.
- Use `MASTER_PLAN.md` as the architecture roadmap for P&ID digitizing work. Do not restate or contradict it in ad hoc task notes.

## Modules / subprojects

| Module | Type | Path | What it owns | How to run | Tests / checks | Docs | AGENTS |
|--------|------|------|--------------|------------|----------------|------|--------|
| Backend API | fastapi | `backend/` | HTTP API, file upload flow, result serving, runtime config | From `backend/`: `uvicorn api:app --reload --port 8001` | `python -m py_compile api.py garnet/*.py garnet/utils/*.py` | `README.md` | `backend/garnet/AGENTS.md` |
| P&ID pipeline | python package | `backend/garnet/` | Stage-by-stage P&ID rebuild, shared OCR/image utilities, pipeline orchestration | From `backend/`: `python -m garnet.pid_extractor` | See module AGENTS | `MASTER_PLAN.md` | `backend/garnet/AGENTS.md` |
| Gemini detector | python adapter | `backend/gemini_detector/` | Gemini/OpenRouter SAHI detector for text-like classes | Called from backend code | `python -m py_compile gemini_detector/*.py` | module file | `backend/gemini_detector/AGENTS.md` |
| Frontend | react + vite | `frontend/` | Review UI, canvas editing, exports, backend API client | From `frontend/`: `bun run dev` | `bun run build`, `bun run lint` | `README.md` | none |
| DeepLSD | vendored library | `DeepLSD/` | Line-detection experiments and model support code | Follow local README / requirements | Module-specific | `DeepLSD/README.md` | none |
| Design assets | docs/assets | `design/` | Design references and non-runtime materials | Open only when task requires design context | n/a | `design/README.md` | none |

## Cross-domain workflows
- Frontend <-> backend: the Vite app proxies `/api` and `/runs` to `VITE_API_URL`, defaulting to `http://localhost:8001`. Keep backend route changes synchronized with frontend API usage.
- API <-> pipeline: `backend/api.py` is the service entrypoint, but the extraction logic lives in `backend/garnet/`. Change request/response shapes in the API layer only after checking the downstream pipeline output and frontend expectations.
- Pipeline roadmap: for P&ID digitizing features, preserve the stage model in `MASTER_PLAN.md` and the scoped rules in `backend/garnet/AGENTS.md`. The live rebuild is currently Stage 1-only.
- Generated artifacts: keep predictions, runs, temp files, and debug outputs in backend-owned artifact folders. Do not make the frontend depend on developer-local filesystem paths.

## Verification (preferred commands)
- Default rule: run checks from the owning module, keep them narrow first, and widen only after the touched area is stable.
- Backend: from `backend/`, use `python -m py_compile api.py garnet/*.py garnet/utils/*.py`; run targeted runtime checks only when the relevant weights/dependencies are present.
- Frontend: from `frontend/`, use `bun run lint` and `bun run build`.
- Root changes: verify the specific files you added or edited, and confirm scoped AGENTS files do not conflict.

## Docs usage
- Do not open or edit `design/` unless the task needs UI/design context.
- Prefer `README.md`, `MASTER_PLAN.md`, and the nearest scoped `AGENTS.md` before reading deeper project material.

## Global conventions
- Route work to the nearest scoped AGENTS file before making module-specific changes.
- Prefer `bun` over `npm` for frontend install, dev, build, and lint commands unless the user explicitly asks otherwise.
- Keep generated outputs, caches, model weights, and secrets out of commits unless the user explicitly asks for them.
- Preserve existing module boundaries. If a task crosses backend, frontend, and pipeline code, describe the touch points clearly and verify each touched module separately.
- When a new submodule develops its own conventions or risk profile, add a nested `AGENTS.md` there instead of bloating the root file.

## Do not
- Do not put backend pipeline implementation rules in the root file when they belong in `backend/garnet/AGENTS.md`.
- Do not assume frontend, backend, and pipeline changes can be validated with one command.
- Do not treat generated artifacts or local weights as source files.

## Links to module instructions
- `backend/garnet/AGENTS.md`
- `backend/gemini_detector/AGENTS.md`

## Topology reconstruction reference
Repository: GARNET

This reference captures the canonical pipeline used to convert object detection
outputs into a structured P&ID graph.

### 1. System overview

Goal:

Convert detected P&ID content into a structured graph.

Output graph contains:

- equipment nodes
- pipe junction nodes
- pipe edges
- equipment-to-pipe connections

Pipeline structure:

```
Detection (SAHI + YOLO)
↓
Pipe Segmentation
↓
Skeleton Extraction
↓
Node Detection
↓
Equipment-Pipe Association
↓
Edge Tracing
↓
Graph Construction
```

The final graph represents the complete piping connectivity network.

### 2. Pipe skeleton extraction

Input:

```
pipe segmentation mask
```

Convert the mask into a 1-pixel centerline skeleton.

Recommended algorithms:

```
opencv.ximgproc.thinning
or
skimage.morphology.skeletonize
```

Example:

```python
pipe_mask = cv2.imread("pipe_mask.png", 0)
binary = pipe_mask > 0

skeleton = skeletonize(binary).astype(np.uint8) * 255
```

All downstream topology analysis operates on this skeleton.

### 3. Node detection

Each skeleton pixel is analyzed using 8-neighborhood connectivity.

| Degree | Meaning     |
| ------ | ----------- |
| 1      | endpoint    |
| 2      | normal pipe |
| >=3    | junction    |

Implementation example:

```python
kernel = np.array([
    [1, 1, 1],
    [1, 10, 1],
    [1, 1, 1],
])

neighbor_count = nd.convolve((skeleton > 0).astype(int), kernel)

junctions = np.where(neighbor_count >= 13)
endpoints = np.where(neighbor_count == 11)
```

Explanation:

```
10 = center pixel
+1 for each neighbor
```

Therefore:

```
11 -> endpoint
12 -> normal pipe
13+ -> junction
```

### 4. Equipment-to-pipe association

Object detection produces equipment:

```
bbox
class
center
```

Each equipment must connect to the nearest pipe centerline.

Approach:

```
KDTree nearest neighbor search
```

Example:

```python
from scipy.spatial import KDTree

skeleton_points = np.column_stack(np.where(skeleton > 0))
tree = KDTree(skeleton_points)

distance, index = tree.query([equip_y, equip_x])
nearest_pipe_point = skeleton_points[index]
```

Create an equipment connection node.

Constraint:

```
distance < threshold (20-40 px typical)
```

### 5. Node consolidation

Skeleton nodes are pixel-dense.

Cluster nearby nodes into single graph nodes.

Recommended method:

```
DBSCAN clustering
```

Example:

```python
clustering = DBSCAN(eps=6, min_samples=1).fit(nodes)
```

Each cluster centroid becomes a graph node.

Node types:

```
junction
endpoint
equipment_connection
```

### 6. Edge tracing

Edges represent pipe segments between nodes.

Procedure:

```
1. Start from node
2. Follow skeleton pixels
3. Stop at next node
4. Record path
```

Traversal strategy:

```
Depth-first traversal
```

Pseudo-code:

```python
def trace_edge(start_pixel):
    path = []
    current = start_pixel

    while current not in node_set:
        path.append(current)
        next_pixel = find_next_neighbor(current)
        current = next_pixel

    return path, current
```

Edge representation:

```
(nodeA, nodeB, polyline)
```

### 7. Graph construction

Use `NetworkX`.

```python
import networkx as nx

G = nx.Graph()

G.add_node(node_id, type="junction", pos=(x, y))
G.add_edge(nodeA, nodeB, polyline=polyline)
```

Graph supports connectivity queries, flow path tracing, automatic line lists,
and equipment connectivity matrices.

### 8. Handling pipe crossings

Skeleton methods falsely interpret pipe crossings as junctions.

P&IDs often contain non-connected crossings.

Rule:

```
crossing without junction symbol -> NOT connected
```

Detect junction symbols:

```
dot
node
tee
```

If no junction symbol exists, split edges to prevent false graph connections.

### 9. Off-page connectors

Detect connectors as special nodes with:

```
connector_id
page_reference
```

Graphs from multiple pages are merged via connector IDs.

### 10. Data model

Node:

```text
{
 id
 type (pump, valve, junction)
 tag
 position
}
```

Edge:

```text
{
 source
 target
 polyline
 pipe_class
 line_number
}
```

### 11. Graph export

Supported formats:

```
GraphML
JSON
Neo4j
```

GraphML is preferred for interoperability.

### 12. Large drawing optimization

Typical P&IDs exceed:

```
30k x 20k pixels
```

Required spatial indexing:

```
KDTree
R-tree
tile-based graph merging
```

### 13. Engineering principle

Detection accuracy is rarely the bottleneck.

The real challenge is topological reconstruction under ambiguous geometry.

Reliable systems combine:

```
computer vision
computational geometry
graph theory
```

### 14. Future improvements

Advanced systems add:

```
relation inference models
graph neural networks
symbol-to-pipe prediction
```

These improve connection accuracy for noisy drawings.

### 15. Canonical pipeline

```
SAHI tiling
YOLO detection
Pipe segmentation
Skeleton extraction
Node detection
KDTree equipment association
Edge tracing
NetworkX graph generation
```
