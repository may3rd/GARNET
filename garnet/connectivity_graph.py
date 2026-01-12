"""
Graph-based connectivity engine for P&ID networks.
Handles robust line merging, intersection detection, and port-to-line snapping.
"""
import math
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set, Any

@dataclass
class GraphNode:
    id: int
    x: float
    y: float
    type: str = "endpoint"
    ref_id: Optional[int] = None  # Reference to external ID (e.g. symbol ID)

@dataclass
class GraphEdge:
    u: int
    v: int
    id: int = -1
    path: Optional[List[Tuple[float, float]]] = None
    attrs: Dict[str, Any] = None


class ConnectivityEngine:
    def __init__(self, merge_dist: float = 10.0, snap_dist: float = 15.0, ortho_tol: float = 2.0):
        self.merge_dist = merge_dist
        self.snap_dist = snap_dist
        self.ortho_tol = ortho_tol
        self.graph = nx.Graph()
        self.nodes: Dict[int, GraphNode] = {}
        self.next_node_id = 0
        self.next_edge_id = 0

    def _add_node(self, x: float, y: float, ntype: str = "endpoint", ref_id: Optional[int] = None) -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        self.nodes[nid] = GraphNode(nid, x, y, ntype, ref_id)
        self.graph.add_node(nid, pos=(x, y), type=ntype, ref_id=ref_id)
        return nid

    def _add_edge(self, u: int, v: int, path: Optional[List[Tuple[float, float]]] = None, attrs: Dict[str, Any] = None) -> int:
        eid = self.next_edge_id
        self.next_edge_id += 1
        if attrs is None:
            attrs = {}
        # Ensure path exists
        if path is None:
            p1 = self.nodes[u]
            p2 = self.nodes[v]
            path = [(p1.x, p1.y), (p2.x, p2.y)]
        
        attrs['id'] = eid
        attrs['path'] = path
        self.graph.add_edge(u, v, **attrs)
        return eid

    def _is_orthogonal(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Check if line segment is approximately horizontal or vertical."""
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        length = math.hypot(dx, dy)
        if length == 0:
            return False
        
        # Angle with horizontal
        angle_deg = math.degrees(math.atan2(dy, dx))
        
        # Horizontal (near 0 or 180)
        if angle_deg <= self.ortho_tol or abs(angle_deg - 180.0) <= self.ortho_tol:
            return True
        # Vertical (near 90)
        if abs(angle_deg - 90.0) <= self.ortho_tol:
            return True
            
        return False

    def build_graph(self, lines: List[Tuple[Tuple[float, float], Tuple[float, float]]], ports: List[Dict]) -> Tuple[nx.Graph, Dict[int, GraphNode]]:
        """
        Builds a connected graph from line segments and ports.
        
        Args:
            lines: List of ((x1, y1), (x2, y2)) segments.
            ports: List of dicts {'id': int, 'pos': (x, y), 'parent_id': int, 'type': str}.
        
        Returns:
            (networkx.Graph, Dict[node_id, GraphNode])
        """
        self.graph.clear()
        self.nodes.clear()
        self.next_node_id = 0
        self.next_edge_id = 0

        # 1. Ingest Lines & Merge Endpoints
        # Filter strictly for orthogonal lines
        valid_lines = [line for line in lines if self._is_orthogonal(line[0], line[1])]
        
        # Collect all endpoints from valid lines
        raw_points = []
        for p1, p2 in valid_lines:
            raw_points.append(p1)
            raw_points.append(p2)
        
        if not raw_points:
            return self.graph, self.nodes

        # KDTree to merge close endpoints
        raw_points_arr = np.array(raw_points)
        tree = KDTree(raw_points_arr)
        
        # Mapping from raw point index to canonical node ID
        point_map: Dict[int, int] = {}
        processed_indices = set()
        
        for i in range(len(raw_points)):
            if i in processed_indices:
                continue
            
            # Find all points within merge radius
            indices = tree.query_ball_point(raw_points_arr[i], self.merge_dist)
            
            # Centroid of cluster
            cluster_points = raw_points_arr[indices]
            cx, cy = np.mean(cluster_points, axis=0)
            
            # Create node
            nid = self._add_node(cx, cy, "junction")
            
            for idx in indices:
                point_map[idx] = nid
                processed_indices.add(idx)

        # Create edges from valid lines
        for i, (p1, p2) in enumerate(valid_lines):
            idx1 = i * 2
            idx2 = i * 2 + 1
            u = point_map[idx1]
            v = point_map[idx2]
            
            if u != v: # avoid self-loops from tiny lines
                # Check if edge already exists to avoid duplication
                if not self.graph.has_edge(u, v):
                    self._add_edge(u, v, attrs={'type': 'pipe'})

        # 2. Simplification: Merge Degree-2 Collinear Nodes
        self._simplify_graph()

        # 3. Snap Ports to Nearest Edge
        self._snap_ports(ports)

        return self.graph, self.nodes

    def _simplify_graph(self, angle_tol: float = 15.0):
        """Merges degree-2 nodes that are collinear."""
        nodes_to_remove = []
        
        for n in list(self.graph.nodes()):
            if self.graph.degree(n) == 2:
                neighbors = list(self.graph.neighbors(n))
                u, v = neighbors[0], neighbors[1]
                
                p_n = self.nodes[n]
                p_u = self.nodes[u]
                p_v = self.nodes[v]
                
                # Check angle u-n-v
                vec1 = np.array([p_n.x - p_u.x, p_n.y - p_u.y])
                vec2 = np.array([p_v.x - p_n.x, p_v.y - p_n.y])
                
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    unit1 = vec1 / norm1
                    unit2 = vec2 / norm2
                    dot = np.dot(unit1, unit2)
                    angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
                    
                    if angle < angle_tol:
                        # Merge: add edge u-v, remove n
                        # Preserve total path length/geometry ideally, but for now linear approx
                        self._add_edge(u, v, attrs={'type': 'pipe'})
                        nodes_to_remove.append(n)

        self.graph.remove_nodes_from(nodes_to_remove)
        for n in nodes_to_remove:
            del self.nodes[n]

    def _snap_ports(self, ports: List[Dict]):
        """
        For each port, find the closest edge. If within snap_dist, 
        split the edge and insert the port node.
        """
        if self.graph.number_of_edges() == 0:
            return

        # Build edge index (midpoints for coarse search)
        edge_keys = list(self.graph.edges(keys=True)) if self.graph.is_multigraph() else list(self.graph.edges())
        edge_data = []
        
        for idx, (u, v) in enumerate(edge_keys):
            p1 = np.array([self.nodes[u].x, self.nodes[u].y])
            p2 = np.array([self.nodes[v].x, self.nodes[v].y])
            mid = (p1 + p2) / 2
            edge_data.append({'u': u, 'v': v, 'p1': p1, 'p2': p2, 'mid': mid, 'idx': idx})
            
        if not edge_data:
            return

        midpoints = np.array([e['mid'] for e in edge_data])
        tree = KDTree(midpoints)

        for port in ports:
            px, py = port['pos']
            p_vec = np.array([px, py])
            
            # Query candidate edges (nearest midpoints)
            # Search radius is generous to catch long segments whose midpoint is far
            # but which pass close to the point.
            # Ideally we check all edges or use a spatial index on segments (R-tree).
            # For simplicity, we check K nearest midpoints.
            k = min(10, len(edge_data))
            dists, indices = tree.query(p_vec, k=k)
            
            best_dist = float('inf')
            best_proj = None
            best_edge = None
            
            indices = [indices] if k == 1 else indices
            
            for i in indices:
                edge = edge_data[i]
                dist, proj = self._point_to_segment(p_vec, edge['p1'], edge['p2'])
                if dist < best_dist:
                    best_dist = dist
                    best_proj = proj
                    best_edge = edge
            
            if best_dist <= self.snap_dist:
                # Snap!
                # 1. Create Port Node
                pid = self._add_node(px, py, "port", ref_id=port.get('parent_id'))
                
                # 2. Remove old edge
                u, v = best_edge['u'], best_edge['v']
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)
                    
                    # 3. Add two new edges (u-port, port-v)
                    self._add_edge(u, pid, attrs={'type': 'pipe'})
                    self._add_edge(pid, v, attrs={'type': 'pipe'})
                    
                    # Update edge_data to prevent other ports from trying to snap to the removed edge
                    # (A robust implementation would rebuild the index, but for sparse ports this is ok)
                    # Ideally, we should update the graph incrementally. 
                    # For this pass, assumes ports are sparse enough on one segment.
    
    def _point_to_segment(self, p, a, b):
        """Distance from point p to segment a-b, and the projection point."""
        ap = p - a
        ab = b - a
        norm_ab = np.linalg.norm(ab)
        if norm_ab == 0:
            return np.linalg.norm(ap), a
        
        t = np.dot(ap, ab) / (norm_ab * norm_ab)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        dist = np.linalg.norm(p - proj)
        return dist, proj
