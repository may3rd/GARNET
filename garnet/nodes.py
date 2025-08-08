from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import cv2
import numpy as np

# ----------------------------
# Class groups (tune as needed)
# ----------------------------
VALVE_CLASSES = {'check valve', 'gate valve', 'globe valve', 'control valve', 'three way valve'}
INLINE_COMPONENTS = VALVE_CLASSES | {'strainer', 'reducer'}
SINGLE_PORT_CLASSES = {'instrument tag', 'instrument dcs', 'instrument logic', 'sampling point', 'utility connection', 'page connection'}
PUMP_CLASSES = {'pump'}
NO_PORT_CLASSES = {'line number', 'spectacle blind'}

# For convenience
def v(x, y):
    return np.array([float(x), float(y)], dtype=float)

@dataclass
class Port:
    pt: Tuple[float, float]
    direction: Tuple[float, float]
    kind: str = "main"

@dataclass
class Node:
    sym_id: int
    cls: str
    bbox: Tuple[float, float, float, float]
    score: float = 1.0
    ports: List[Port] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

def binarize_lines(gray: np.ndarray) -> np.ndarray:
    # Preprocess the image
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k)
    return (bw > 0).astype(np.uint8)

def _cluster_1d(coords: np.ndarray, gap: int = 5) -> List[np.ndarray]:
    clusters = []
    if coords.size == 0:
        return clusters
    coords = np.sort(coords)
    start = 0
    for i in range(1, len(coords)):
        if coords[i] - coords[i - 1] > gap:
            clusters.append(coords[start:i])
            start = i
    clusters.append(coords[start:])
    return clusters

def _cluster_centers(vals: np.ndarray) -> List[float]:
    return [float(np.median(c)) for c in _cluster_1d(vals)]

# ----------------------------
# Port template generator
# ----------------------------
def generate_ports_for_bbox(bbox: Tuple[float,float,float,float], cls: str) -> List[Port]:
    """
    Heuristic, bbox-only port templates.
    Returns a list of Port objects with points on the bbox perimeter and nominal directions.
    """
    x, y, w, h = bbox
    cx, cy = x + w/2.0, y + h/2.0

    # Perimeter midpoints
    L = v(x,       cy); R = v(x + w, cy)
    T = v(cx,      y ); B = v(cx,    y + h)

    # Direction unit vectors
    dL = v(-1,  0); dR = v( 1,  0)
    dT = v( 0, -1); dB = v( 0,  1)

    ports: List[Port] = []

    # Inline components: 2 ports on dominant axis, fallback to the other axis
    if cls in INLINE_COMPONENTS:
        if w >= h:
            ports = [Port(tuple(L), tuple(dL), 'in'), Port(tuple(R), tuple(dR), 'out')]
        else:
            ports = [Port(tuple(T), tuple(dT), 'in'), Port(tuple(B), tuple(dB), 'out')]

        # Three-way valve: add third on the perpendicular side nearest to center
        if cls == 'three way valve':
            if w >= h:
                # horizontal main flow, third on top (prefer) then bottom
                ports.append(Port(tuple(T), tuple(dT), 'third'))
            else:
                # vertical main flow, third on left (prefer) then right
                ports.append(Port(tuple(L), tuple(dL), 'third'))

        return ports

    # Pumps: often two opposed nozzles; prefer horizontal if bbox is wider
    if cls in PUMP_CLASSES:
        if w >= h:
            ports = [Port(tuple(L), tuple(dL), 'suction'), Port(tuple(R), tuple(dR), 'discharge')]
        else:
            ports = [Port(tuple(T), tuple(dT), 'suction'), Port(tuple(B), tuple(dB), 'discharge')]
        return ports

    # Single-port classes: one leader/connector from nearest side
    if cls in SINGLE_PORT_CLASSES:
        # If wider than tall, prefer left/right; else top/bottom
        if w >= h:
            ports = [Port(tuple(L), tuple(dL), 'single')]
            # If you want to allow either side, add R as alternative candidate
            ports.append(Port(tuple(R), tuple(dR), 'single_alt'))
        else:
            ports = [Port(tuple(T), tuple(dT), 'single')]
            ports.append(Port(tuple(B), tuple(dB), 'single_alt'))
        return ports

    # Default: treat as inline
    if w >= h:
        ports = [Port(tuple(L), tuple(dL), 'in'), Port(tuple(R), tuple(dR), 'out')]
    else:
        ports = [Port(tuple(T), tuple(dT), 'in'), Port(tuple(B), tuple(dB), 'out')]
    return ports

def generate_ports_for_bbox_from_lines_refined(
    bbox: Tuple[float, float, float, float],
    cls: str,
    line_mask: np.ndarray,
    probe_w: int = 5,
    min_run: int = 1,
) -> List[Port]:
    if cls in NO_PORT_CLASSES:
        return []

    x, y, w, h = map(int, bbox)
    H, W = line_mask.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W - 1, x + w), min(H - 1, y + h)
    dL = (-1.0, 0.0); dR = (1.0, 0.0); dT = (0.0, -1.0); dB = (0.0, 1.0)

    def scan_top():
        xs = []
        if y0 > 0:
            band_out = line_mask[max(0, y0 - probe_w):y0, x0:x1 + 1]
            if band_out.size:
                xs += (np.where(band_out.any(axis=0))[0] + x0).tolist()
        band_in = line_mask[y0:min(H, y0 + probe_w), x0:x1 + 1]
        if band_in.size:
            xs += (np.where(band_in.any(axis=0))[0] + x0).tolist()
        return np.unique(xs)

    def scan_left():
        ys = []
        if x0 > 0:
            band_out = line_mask[y0:y1 + 1, max(0, x0 - probe_w):x0]
            if band_out.size:
                ys += (np.where(band_out.any(axis=1))[0] + y0).tolist()
        band_in = line_mask[y0:y1 + 1, x0:min(W, x0 + probe_w)]
        if band_in.size:
            ys += (np.where(band_in.any(axis=1))[0] + y0).tolist()
        return np.unique(ys)

    def scan_right():
        ys = []
        if x1 < W - 1:
            band_out = line_mask[y0:y1 + 1, x1 + 1:min(W, x1 + 1 + probe_w)]
            if band_out.size:
                ys += (np.where(band_out.any(axis=1))[0] + y0).tolist()
        # This part checks the band *inside* the bbox, just next to the right edge
        band_in = line_mask[y0:y1 + 1, max(0, x1 - probe_w + 1):x1 + 1]
        if band_in.size:
            # Fix: Use band_in here, not band_out
            ys += (np.where(band_in.any(axis=1))[0] + y0).tolist()
        return np.unique(ys)

    def scan_bottom():
        xs = []
        if y1 < H - 1:
            band_out = line_mask[y1 + 1:min(H, y1 + 1 + probe_w), x0:x1 + 1]
            if band_out.size:
                xs += (np.where(band_out.any(axis=0))[0] + x0).tolist()
        # This part checks the band *inside* the bbox, just next to the bottom edge
        band_in = line_mask[max(0, y1 - probe_w + 1):y1 + 1, x0:x1 + 1]
        if band_in.size:
            # Fix: Use band_in here, not band_out
            xs += (np.where(band_in.any(axis=0))[0] + x0).tolist()
        return np.unique(xs)

    def centers_1d(vals):
        vals = np.asarray(vals, dtype=int)
        return [float(np.median(c)) for c in _cluster_1d(vals, gap=6)]

    ports_top = [(cx, float(y0), dT) for cx in centers_1d(scan_top())]
    ports_bot = [(cx, float(y1), dB) for cx in centers_1d(scan_bottom())]
    ports_left = [(float(x0), cy, dL) for cy in centers_1d(scan_left())]
    ports_right = [(float(x1), cy, dR) for cy in centers_1d(scan_right())]

    def to_ports(items, kind='main'):
        return [Port((px, py), dirv, kind) for (px, py, dirv) in items]

    out: List[Port] = []

    if cls in VALVE_CLASSES:
        shorter_is_vertical = (w < h)
        if cls in {'check valve', 'butterfly valve'}:
            # Detect diagonal lines for check valves and butterfly valves
            try:
                img = cv2.imread("test/!test01.png", cv2.IMREAD_GRAYSCALE)
                roi = img[y0:y1, x0:x1]
                _, thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
                lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi / 180, threshold=50,
                                       minLineLength=10, maxLineGap=5)

                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # Check if the line is diagonal
                        if abs(x1 - x2) > 5 and abs(y1 - y2) > 5:
                            # Find intersection points with bounding box edges
                            if x1 == x2 or y1 == y2:
                                continue  # Skip vertical/horizontal lines
                            # Calculate direction vector
                            dx, dy = x2 - x1, y2 - y1
                            norm = (dx**2 + dy**2)**0.5
                            if norm > 0:
                                dx, dy = dx / norm, dy / norm
                            # Assign ports based on direction
                            if shorter_is_vertical:
                                # Vertical valve: ports on top/bottom
                                if dy > 0:
                                    out.append(Port((x0 + (x1 + x2) / 2, y0), (0, -1), 'in'))
                                else:
                                    out.append(Port((x0 + (x1 + x2) / 2, y1), (0, 1), 'out'))
                            else:
                                # Horizontal valve: ports on left/right
                                if dx > 0:
                                    out.append(Port((x0, y0 + (y1 + y2) / 2), (-1, 0), 'in'))
                                else:
                                    out.append(Port((x1, y0 + (y1 + y2) / 2), (1, 0), 'out'))

            except Exception as e:
                print(f"Warning: Diagonal line detection failed for valve {cls} at {bbox}. Error: {e}")

        else:
            # For other valves (e.g., gate valve, globe valve)
            if shorter_is_vertical:
                # Vertical valve: ports on top/bottom
                if ports_top:
                    out.append(to_ports([ports_top[0]], 'in')[0])
                if ports_bot:
                    out.append(to_ports([ports_bot[0]], 'out')[0])
            else:
                # Horizontal valve: ports on left/right
                if ports_left:
                    out.append(to_ports([ports_left[0]], 'in')[0])
                if ports_right:
                    out.append(to_ports([ports_right[0]], 'out')[0])

        # Fallback for valves if line detection fails
        if not out:
            # Use the template-based approach if no ports were detected via lines
            template_ports = generate_ports_for_bbox(bbox, cls)  # Reuse existing template logic
            # Assign 'in'/'out' kinds if they were generic or different
            for p in template_ports:
                if p.kind in ['in', 'out', 'main']:
                    pass  # Keep the kind from the template
            out.extend(template_ports)

    elif cls in SINGLE_PORT_CLASSES:
        side_scores = [
            ('L', len(ports_left), ports_left),
            ('R', len(ports_right), ports_right),
            ('T', len(ports_top), ports_top),
            ('B', len(ports_bot), ports_bot),
        ]
        side_scores.sort(key=lambda t: t[1], reverse=True)
        for _, score, items in side_scores:
            if score > 0:
                return to_ports(items, 'single')
        return []

    elif cls in PUMP_CLASSES:
        if w <= h:
            # Horizontal pump
            if ports_left:
                out.append(to_ports([ports_left[0]], 'suction')[0])
            if ports_right:
                out.append(to_ports([ports_right[0]], 'discharge')[0])
        else:
            # Vertical pump
            if ports_top:
                out.append(to_ports([ports_top[0]], 'suction')[0])
            if ports_bot:
                out.append(to_ports([ports_bot[0]], 'discharge')[0])

    else:
        shorter_is_vertical = (w > h)
        if shorter_is_vertical:
            if ports_left:
                out.append(to_ports([ports_left[0]], 'in')[0])
            if ports_right:
                out.append(to_ports([ports_right[0]], 'out')[0])
            if len(out) < 2:
                if ports_top:
                    out.append(to_ports([ports_top[0]], 'alt')[0])
                if len(out) < 2 and ports_bot:
                    out.append(to_ports([ports_bot[0]], 'alt')[0])
        else:
            if ports_top:
                out.append(to_ports([ports_top[0]], 'in')[0])
            if ports_bot:
                out.append(to_ports([ports_bot[0]], 'out')[0])
            if len(out) < 2:
                if ports_left:
                    out.append(to_ports([ports_left[0]], 'alt')[0])
                if len(out) < 2 and ports_right:
                    out.append(to_ports([ports_right[0]], 'alt')[0])

    if not out and cls in INLINE_COMPONENTS:
        cx, cy = x + w / 2.0, y + h / 2.0
        if shorter_is_vertical:
            out = [Port((x, cy), dL, 'fallback'), Port((x + w, cy), dR, 'fallback')]
        else:
            out = [Port((cx, y), dT, 'fallback'), Port((cx, y + h), dB, 'fallback')]

    if cls == 'three way valve' and len(out) >= 2:
        used = {(round(p.pt[0]), round(p.pt[1])) for p in out}
        for side in [ports_top, ports_bot, ports_left, ports_right]:
            for px, py, dirv in side:
                if (round(px), round(py)) not in used:
                    out.append(Port((px, py), dirv, 'third'))
                    break
            if len(out) >= 3:
                break

    return out

def nodes_from_coco_annotations(line_mask: np.ndarray, annos: List[Dict]) -> List[Node]:
    nodes: List[Node] = []
    for i, a in enumerate(annos):
        cls = a.get('category_name') or a.get('name') or str(a.get('category_id'))
        a1, a2, a3, a4 = a['bbox']
        bbox = (a1, a2, a3, a4)
        score = float(a.get('score', 1.0))
        node = Node(sym_id=i, cls=cls, bbox=bbox, score=score)
        node.ports = generate_ports_for_bbox_from_lines_refined(node.bbox, node.cls, line_mask, probe_w=10, min_run=1)
        nodes.append(node)
    return nodes

def draw_nodes(image, nodes, show_labels=True, show_port_kinds=False,
               box_color=(0, 255, 0), port_color=(0, 0, 255), dir_len=8, thickness=2):
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    for n in nodes:
        x, y, w, h = map(int, n.bbox)
        cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, thickness)
        if show_labels:
            label = n.cls
            if hasattr(n, 'score') and n.score is not None:
                label = f"{label} {n.score:.2f}"
            tsize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx, ty = x, max(0, y - 3)
            bg_pt1 = (tx, max(0, ty - tsize[1] - 4))
            bg_pt2 = (tx + tsize[0] + 4, ty)
            cv2.rectangle(vis, bg_pt1, bg_pt2, (0, 0, 0), -1)
            cv2.putText(vis, label, (tx + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for p in n.ports:
            px, py = map(int, p.pt)
            cv2.circle(vis, (px, py), 3, port_color, -1, lineType=cv2.LINE_AA)
            dx, dy = p.direction
            norm = (dx ** 2 + dy ** 2) ** 0.5
            if norm > 0:
                dx, dy = dx / norm, dy / norm
                qx, qy = int(round(px + dx * dir_len)), int(round(py + dy * dir_len))
                cv2.line(vis, (px, py), (qx, qy), port_color, 1, cv2.LINE_AA)
            if show_port_kinds:
                text = getattr(p, 'kind', '')
                if text:
                    cv2.putText(vis, text, (px + 5, py - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 255), 1, cv2.LINE_AA)
    return vis

if __name__ == "__main__":
    import json
    with open("output/coco_annotation.json", "r") as f:
        coco = json.load(f)
    annos = coco["annotations"]
    img = cv2.imread("test/!test01.png", cv2.IMREAD_GRAYSCALE)
    line_mask = binarize_lines(img)
    nodes = nodes_from_coco_annotations(line_mask, annos)
    vis = draw_nodes(cv2.imread("test/!test01.png", cv2.IMREAD_COLOR), nodes, show_labels=False, show_port_kinds=False)
    cv2.imwrite("vis.png", vis)