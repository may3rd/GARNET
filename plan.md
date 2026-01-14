# Plan: Refactor Connectivity Analysis Engine

## Goal
Replace the fragile, line-bag-based connectivity analysis in `garnet/pid_extractor.py` (Stage 6) with a robust, graph-based `ConnectivityEngine` that correctly handles P&ID topology, intersections, and port-based symbol connections.

## Research Findings
- **Current State:**
    - `Stage 5` detects "Ports" on symbol bounding boxes using pixel analysis (Canny/Binary) but these ports are mostly unused in the final graph.
    - `Stage 6` builds the graph by snapping symbol *centers* to the nearest DeepLSD line segment.
    - Line merging (`combine_close_lines`) is naive (pairwise, O(N^2)) and doesn't handle intersections or T-junctions.
    - Result: Disconnected graphs, missed T-junctions, and imprecise symbol connections.
- **Constraints:**
    - Must maintain the output structure (`nodes`, `edges` lists) for `dexpi_exporter.py`.
    - Should reuse the `DeepLSD` lines and `Stage 5` Ports.

## Analysis
- **Problem:** The current "bag of lines" approach fails to capture the true topology of a P&ID network (junctions, crossings). Snapping symbol centers ignores the precise "port" locations found in Stage 5.
- **Solution:** Implement a `ConnectivityEngine` that:
    1.  **Ingests Lines:** Converts DeepLSD segments into a proper geometric graph by snapping endpoints and handling intersections.
    2.  **Ingests Ports:** Uses the explicit `NodeType.PORT` nodes from Stage 5.
    3.  **Connects:** Snaps Ports to the nearest *Edge* of the line graph, splits the edge, and inserts the port node.
    4.  **Exports:** Produces the standard Node/Edge list.

## Implementation Steps

### 1. Create `garnet/connectivity_graph.py`
This new module will contain the `ConnectivityEngine` class.
- **`LineGraphBuilder` Class:**
    -   `add_lines(lines)`:
        -   Index endpoints using `scipy.spatial.KDTree`.
        -   Cluster endpoints within a radius (`merge_dist`) to form graph nodes.
        -   Create initial graph edges from segments.
    -   `merge_collinear()`:
        -   Identify nodes with degree 2.
        -   If neighbors are collinear, merge the edges.
    -   `add_ports(port_nodes)`:
        -   For each port, find the nearest graph edge (segment).
        -   Project port onto the segment.
        -   Split the edge at the projection point: `u-v` becomes `u-port-v`.
-   **Helper Geometry Functions:**
    -   `point_to_segment_dist`
    -   `project_point_on_segment`

### 2. Update `garnet/pid_extractor.py`
-   Import `ConnectivityEngine` from `garnet.connectivity_graph`.
-   **Refactor `stage6_line_graph`:**
    -   Collect `combined_deeplsd_lines`.
    -   Collect `PORT` nodes from `self.nodes` (created in Stage 5).
    -   Instantiate `ConnectivityEngine`.
    -   Run `engine.build_graph(lines, ports, symbols)`.
    -   Replace `self.nodes` and `self.edges` with the engine's output.
    -   Ensure `Symbol` and `Text` nodes are preserved and linked.

### 3. Cleanup
-   Remove the old ad-hoc snapping logic in `stage6_line_graph`.
-   Remove unused imports.

## Verification
-   **Visual Inspection:** The `stage6_graph_overlay` image should show:
    -   Lines connected at junctions (no gaps).
    -   Symbols connected via their ports (blue dots) to the lines.
    -   No "flying" edges connecting symbol centers directly to lines.
