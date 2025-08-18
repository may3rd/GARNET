import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import networkx as nx
import numpy as np
from shapely.geometry import LineString, box as shapely_box
from shapely.ops import linemerge
from skimage.morphology import skeletonize
from skimage.util import invert as sk_invert
from tqdm.auto import tqdm

# Optional: OpenCV ximgproc thinning if available
try:
    from cv2 import ximgproc as cv_ximgproc # pyright: ignore[reportAttributeAccessIssue]
except Exception:
    cv_ximgproc = None

# ------------------------------
# A. Inputs and shared data
# ------------------------------

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int
    cls: str
    score: float

@dataclass
class Segment:
    id: int
    polyline: np.ndarray  # shape (N,2), int
    start: Tuple[int, int]
    end: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    length_px: float
    neighbors: List[int]

def load_coco(coco_json_path: str) -> Tuple[str, int, int, List[BBox]]:
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    img_entry = coco['images'][0]
    img_file = img_entry['file_name']
    W, H = img_entry['width'], img_entry['height']

    anns = []
    for a in coco['annotations']:
        x, y, w, h = a['bbox']
        anns.append(BBox(
            x=int(round(x)),
            y=int(round(y)),
            w=int(round(w)),
            h=int(round(h)),
            cls=a.get('category_name', str(a['category_id'])),
            score=float(a.get('score', 1.0))
        ))
    return img_file, W, H, anns

# Helper: Load OCR results (xyxy) and convert to BBox(xywh)
def load_ocr(ocr_json_path: str) -> List[BBox]:
    """
    Load OCR results saved by the OCR pipeline. Expected format:
    {
        "image": {"file_name": str, "width": int, "height": int},
        "annotations": [
            {"bbox": [x1, y1, x2, y2], "text": str, "score": float}, ...
        ]
    }
    Returns a list of BBox with cls='text' and (x,y,w,h) derived from [x1,y1,x2,y2].
    """
    if not os.path.exists(ocr_json_path):
        return []
    with open(ocr_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Fetch image bounds if available for clamping
    W = None; H = None
    try:
        W = int(data.get('image', {}).get('width'))
        H = int(data.get('image', {}).get('height'))
    except Exception:
        W = None; H = None

    anns: List[BBox] = []
    for a in data.get('annotations', []):
        bbox = a.get('bbox') or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(float, bbox)
        # Convert xyxy -> xywh
        x = int(round(min(x1, x2)))
        y = int(round(min(y1, y2)))
        w = int(round(abs(x2 - x1)))
        h = int(round(abs(y2 - y1)))
        # Ensure minimum size
        w = max(1, w)
        h = max(1, h)
        # Clamp to image bounds if provided
        if W is not None and H is not None:
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            if x + w > W:
                w = max(1, W - x)
            if y + h > H:
                h = max(1, H - y)
        anns.append(BBox(x=x, y=y, w=w, h=h, cls='text', score=float(a.get('score', 1.0))))
    return anns

def read_gray(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")
    return img

def merge_collinear_segments(segments: List[Segment], cfg: Dict) -> List[Segment]:
    """Post-process to merge collinear horizontal and vertical line segments."""
    angle_tol_deg = cfg.get('merge_angle_tol', 2.0)
    gap_tol_px = cfg.get('merge_gap_tol', 15)
    y_tol_px = cfg.get('merge_y_tol', 5)  # y-tolerance for horizontal lines
    x_tol_px = cfg.get('merge_x_tol', 5)  # x-tolerance for vertical lines

    horizontals, verticals, others = [], [], []

    # 1. Classify segments
    for seg in segments:
        p1 = seg.start
        p2 = seg.end
        dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
        if dx < 1 and dy < 1: continue # skip zero-length segments

        angle = math.degrees(math.atan2(dy, dx))
        if angle < angle_tol_deg:
            horizontals.append(seg)
        elif angle > (90 - angle_tol_deg):
            verticals.append(seg)
        else:
            others.append(seg)

    # 2. Merge horizontal segments
    horizontals.sort(key=lambda s: (min(s.start[1], s.end[1]), min(s.start[0], s.end[0])))
    merged_horz = []
    if horizontals:
        current_h = horizontals[0]
        for next_h in horizontals[1:]:
            cy_min = min(current_h.start[1], current_h.end[1])
            ny_min = min(next_h.start[1], next_h.end[1])

            cx_max = max(current_h.start[0], current_h.end[0])
            nx_min = min(next_h.start[0], next_h.end[0])

            # Check for alignment and proximity
            if abs(cy_min - ny_min) < y_tol_px and (0 < (nx_min - cx_max) < gap_tol_px):
                # Merge: create a new line from the outer points
                all_x = [current_h.start[0], current_h.end[0], next_h.start[0], next_h.end[0]]
                all_y = [current_h.start[1], current_h.end[1], next_h.start[1], next_h.end[1]]
                new_x1, new_x2 = min(all_x), max(all_x)
                new_y = int(np.mean(all_y)) # Average the y-coordinates
                
                new_poly = np.array([[new_x1, new_y], [new_x2, new_y]], dtype=np.int32)
                current_h = Segment(id=-1, polyline=new_poly, start=(new_x1, new_y), end=(new_x2, new_y),
                                    bbox=bbox_of_polyline(new_poly), length_px=length_of_polyline(new_poly),
                                    neighbors=[])
            else:
                merged_horz.append(current_h)
                current_h = next_h
        merged_horz.append(current_h)
    
    # 3. Merge vertical segments
    verticals.sort(key=lambda s: (min(s.start[0], s.end[0]), min(s.start[1], s.end[1])))
    merged_vert = []
    if verticals:
        current_v = verticals[0]
        for next_v in verticals[1:]:
            cx_min = min(current_v.start[0], current_v.end[0])
            nx_min = min(next_v.start[0], next_v.end[0])
            
            cy_max = max(current_v.start[1], current_v.end[1])
            ny_min = min(next_v.start[1], next_v.end[1])

            if abs(cx_min - nx_min) < x_tol_px and (0 < (ny_min - cy_max) < gap_tol_px):
                all_y = [current_v.start[1], current_v.end[1], next_v.start[1], next_v.end[1]]
                all_x = [current_v.start[0], current_v.end[0], next_v.start[0], next_v.end[0]]
                new_y1, new_y2 = min(all_y), max(all_y)
                new_x = int(np.mean(all_x))
                
                new_poly = np.array([[new_x, new_y1], [new_x, new_y2]], dtype=np.int32)
                current_v = Segment(id=-1, polyline=new_poly, start=(new_x, new_y1), end=(new_x, new_y2),
                                    bbox=bbox_of_polyline(new_poly), length_px=length_of_polyline(new_poly),
                                    neighbors=[])
            else:
                merged_vert.append(current_v)
                current_v = next_v
        merged_vert.append(current_v)
        
    # 4. Recombine and re-index
    final_segments = merged_horz + merged_vert + others
    for i, seg in enumerate(final_segments):
        seg.id = i
        
    return final_segments

# ------------------------------
# B. Binarization and skeletonization
# ------------------------------

def _fill_line_gaps(bin_img, kernel_size=5):
    """Fills gaps in lines using a morphological closing operation."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

def binarize(img_gray: np.ndarray,
             use_adaptive: bool = False,
             block_size: int = 41,
             C: int = 7) -> np.ndarray:
    # Equalize to stabilize thresholds
    eq = cv2.equalizeHist(img_gray)
    if use_adaptive:
        bin_img = cv2.adaptiveThreshold(eq, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        block_size, C)
    else:
        _, bin_img = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    
    # --- ADD THIS LINE ---
    # Fill gaps from thick lines that became parallel
    bin_img = _fill_line_gaps(bin_img, kernel_size=7) # You can tune the kernel_size

    # Despeckle: remove tiny CCs
    nb_components, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    sizes = stats[:, cv2.CC_STAT_AREA]
    clean = np.zeros_like(bin_img)
    min_area = 10  # tune if needed
    for i in range(1, nb_components):
        if sizes[i] >= min_area:
            clean[labels == i] = 255
    return clean


# Fast thinning using OpenCV ximgproc if available, else fallback
def fast_thinning(bin_img: np.ndarray) -> np.ndarray:
    """Use OpenCV ximgproc thinning if available; otherwise fall back to zhang_suen_skeleton."""
    if cv_ximgproc is not None and hasattr(cv_ximgproc, 'thinning'):
        return cv_ximgproc.thinning(bin_img, thinningType=cv_ximgproc.THINNING_GUOHALL)
    return zhang_suen_skeleton(bin_img)

def zhang_suen_skeleton(bin_img: np.ndarray) -> np.ndarray:
    # skimage skeletonize expects foreground as True
    # Our bin_img is 255 for foreground; convert to boolean.
    bool_img = (bin_img > 0).astype(np.uint8)
    skel = skeletonize(bool_img).astype(np.uint8)  # 1 where skeleton
    return (skel * 255).astype(np.uint8)

def prune_spurs(skel: np.ndarray, Lmin: int = 8, max_iter: int = 50) -> np.ndarray:
    # Convert to binary 0/1
    s = (skel > 0).astype(np.uint8)
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]], dtype=np.uint8)

    def neighbor_count(img):
        # Convolution to count neighbors; center weighted to keep center value
        conv = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        # neighbor count = conv - 10*center
        return conv - (img * 10)

    for _ in range(max_iter):
        nbh = neighbor_count(s)
        endpoints = ((s == 1) & (nbh == 1))  # degree-1 points
        if not endpoints.any():
            break
        # grow small branches from endpoints and remove if length < Lmin
        to_remove = np.zeros_like(s)
        ep_coords = np.column_stack(np.where(endpoints))
        for (r, c) in ep_coords:
            # walk along the branch
            path = [(r, c)]
            visited = set(path)
            cur = (r, c)
            for _step in range(Lmin):
                # find neighbors
                nbrs = []
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        if dr == 0 and dc == 0: continue
                        rr, cc = cur[0]+dr, cur[1]+dc
                        if rr < 0 or cc < 0 or rr >= s.shape[0] or cc >= s.shape[1]:
                            continue
                        if s[rr, cc] == 1 and (rr, cc) not in visited:
                            nbrs.append((rr, cc))
                if len(nbrs) != 1:
                    break  # reached junction or end
                cur = nbrs[0]
                visited.add(cur)
                path.append(cur)
            # If we did not hit a junction within Lmin steps, remove the path
            if len(path) < Lmin:
                for (rr, cc) in path:
                    to_remove[rr, cc] = 1
        s[to_remove == 1] = 0
    return (s * 255).astype(np.uint8)

# ------------------------------
# C. Mask out non-line areas
# ------------------------------

def build_symbol_text_mask(H: int, W: int, boxes: List[BBox],
                           inflate: int = 2,
                           erode_ksize: int = 0,
                           erode_iter: int = 0) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    # Treat all classes in annotations as non-line areas to mask
    for b in boxes:
        x0 = max(0, b.x - inflate); y0 = max(0, b.y - inflate)
        x1 = min(W, b.x + b.w + inflate); y1 = min(H, b.y + b.h + inflate)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1) # type: ignore
        
    if erode_ksize > 0 and erode_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_ksize, erode_ksize))
        mask = cv2.erode(mask, k, iterations=erode_iter)
        
    return mask

def apply_mask(skel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = skel.copy()
    out[mask > 0] = 0
    return out

# ------------------------------
# D. Extract continuous lines
# ------------------------------

def scan_horizontal(skel: np.ndarray, min_run: int = 3, gap_tol: int = 2) -> List[Tuple[int,int,int]]:
    H, W = skel.shape
    lines = []
    for y in range(H):
        x = 0
        while x < W:
            # find start of run
            while x < W and skel[y, x] == 0:
                x += 1
            if x >= W:
                break
            x0 = x
            while x < W and skel[y, x] > 0:
                x += 1
            x1 = x - 1
            if x1 - x0 + 1 >= min_run:
                lines.append((y, x0, x1))
    # merge near-collinear runs within gap_tol on same row
    merged = []
    if not lines:
        return merged
    lines.sort(key=lambda t: (t[0], t[1]))
    cur_y, cur_x0, cur_x1 = lines[0]
    for y, x0, x1 in lines[1:]:
        if y == cur_y and x0 - cur_x1 <= gap_tol and x0 >= cur_x0:
            cur_x1 = max(cur_x1, x1)
        else:
            merged.append((cur_y, cur_x0, cur_x1))
            cur_y, cur_x0, cur_x1 = y, x0, x1
    merged.append((cur_y, cur_x0, cur_x1))
    return merged

def scan_vertical(skel: np.ndarray, min_run: int = 3, gap_tol: int = 2) -> List[Tuple[int,int,int]]:
    H, W = skel.shape
    lines = []
    for x in range(W):
        y = 0
        while y < H:
            while y < H and skel[y, x] == 0:
                y += 1
            if y >= H:
                break
            y0 = y
            while y < H and skel[y, x] > 0:
                y += 1
            y1 = y - 1
            if y1 - y0 + 1 >= min_run:
                lines.append((x, y0, y1))
    # merge near-collinear runs within gap_tol on same column
    merged = []
    if not lines:
        return merged
    lines.sort(key=lambda t: (t[0], t[1]))
    cur_x, cur_y0, cur_y1 = lines[0]
    for x, y0, y1 in lines[1:]:
        if x == cur_x and y0 - cur_y1 <= gap_tol and y0 >= cur_y0:
            cur_y1 = max(cur_y1, y1)
        else:
            merged.append((cur_x, cur_y0, cur_y1))
            cur_x, cur_y0, cur_y1 = x, y0, y1
    merged.append((cur_x, cur_y0, cur_y1))
    return merged

def hough_diagonals(skel: np.ndarray,
                    min_len: int = 20,
                    max_gap: int = 3,
                    theta_deg_tol_exclude: int = 10,
                    vote_thresh: int = 40,
                    max_lines_cap: int = 200) -> List[Tuple[int,int,int,int]]:
    # Use probabilistic Hough on skeleton (white=255)
    linesP = cv2.HoughLinesP(skel, rho=1, theta=np.pi/180,
                             threshold=vote_thresh,
                             minLineLength=min_len, maxLineGap=max_gap)
    result = []
    if linesP is None:
        return result
    for l in linesP[:,0,:]:
        x1, y1, x2, y2 = map(int, l.tolist())
        dx, dy = x2 - x1, y2 - y1
        angle = abs(math.degrees(math.atan2(dy, dx)))
        # filter near-horizontal/vertical
        if angle < theta_deg_tol_exclude or angle > (90 - theta_deg_tol_exclude):
            continue
        result.append((x1, y1, x2, y2))
    # Cap output to avoid pathological cases
    if len(result) > max_lines_cap:
        result = sorted(result, key=lambda l: abs(l[2]-l[0]) + abs(l[3]-l[1]), reverse=True)[:max_lines_cap]
    return result


# Helper: Deduplicate near-parallel lines using bbox and length-based gating
def dedup_near_parallel(line_geoms, tol=1.5, overlap_ratio=0.8):
    """Keep longer lines and drop shorter near-duplicates using a fast bbox gate before geometry ops."""
    line_geoms = sorted(line_geoms, key=lambda g: g.length, reverse=True)
    kept = []
    kept_bounds = []  # expanded bounds with tol
    for cand in line_geoms:
        cb = cand.bounds
        cb_exp = (cb[0]-tol, cb[1]-tol, cb[2]+tol, cb[3]+tol)
        drop = False
        for kb, keep in zip(kept_bounds, kept):
            # quick bbox overlap test
            if cb_exp[2] < kb[0] or cb_exp[0] > kb[2] or cb_exp[3] < kb[1] or cb_exp[1] > kb[3]:
                continue
            inter = cand.intersection(keep.buffer(tol))
            olen = getattr(inter, 'length', 0.0)
            if (olen / max(cand.length, 1e-6)) >= overlap_ratio:
                drop = True
                break
        if not drop:
            kept.append(cand)
            kept_bounds.append(cb_exp)
    return kept

def verify_and_trace_diagonal(skel: np.ndarray,
                              line: Tuple[int,int,int,int],
                              band: int = 1,
                              min_overlap_ratio: float = 0.7) -> Tuple[np.ndarray, Tuple[int,int], Tuple[int,int]] or None: # type: ignore
    x1, y1, x2, y2 = line
    # draw candidate with a thin line and test overlap within a band
    H, W = skel.shape
    cand = np.zeros_like(skel)
    cv2.line(cand, (x1, y1), (x2, y2), 255, thickness=1) # type: ignore
    # Dilate candidate to create band
    if band > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*band+1, 2*band+1))
        cand = cv2.dilate(cand, kernel)
    overlap = cv2.bitwise_and(cand, skel)
    ol_len = np.count_nonzero(overlap)
    cand_len = int(math.hypot(x2-x1, y2-y1))
    if cand_len == 0:
        return None
    if (ol_len / cand_len) < min_overlap_ratio:
        return None
    # Trace the actual pixels along overlap to get endpoints
    ys, xs = np.where(overlap > 0)
    if len(xs) < 2:
        return None
    # Use endpoints as extremal points along the candidate direction
    pts = np.column_stack([xs, ys])
    # Project points along candidate direction
    v = np.array([x2-x1, y2-y1], dtype=float)
    v = v / (np.linalg.norm(v) + 1e-8)
    proj = pts @ v
    i_min = int(np.argmin(proj)); i_max = int(np.argmax(proj))
    p0 = tuple(pts[i_min])
    p1 = tuple(pts[i_max])
    # Polyline approximation: sort by projection
    order = np.argsort(proj)
    poly = pts[order]
    return poly, p0, p1

def build_pixel_graph(skel: np.ndarray, progress: bool = False) -> nx.Graph:
    s = (skel > 0).astype(np.uint8)
    H, W = s.shape
    G = nx.Graph()
    # Add nodes for all skeleton pixels; track degree by adding edges among neighbors
    idx = np.where(s > 0)
    coords = list(zip(idx[0], idx[1]))
    it_nodes = tqdm(coords, desc="Add nodes", leave=False) if progress else coords
    for r, c in it_nodes:
        G.add_node((r, c))
    it_edges = tqdm(coords, desc="Build edges", leave=False) if progress else coords
    coords_set = set(coords)
    for r, c in it_edges:
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr == 0 and dc == 0: continue
                rr, cc = r+dr, c+dc
                if (rr, cc) in coords_set:
                    G.add_edge((r, c), (rr, cc))
    return G

def walk_edges_splitted(G: nx.Graph, progress: bool = False) -> List[List[Tuple[int,int]]]:
    # Extract paths between junctions (degree!=2) as atomic segments
    segments = []
    visited_edges = set()

    def is_junction(n):
        return G.degree[n] != 2 # type: ignore

    it_nodes = tqdm(G.nodes, desc="Walk edges", leave=False) if progress else G.nodes
    for n in it_nodes:
        if is_junction(n):
            for nbr in G.neighbors(n):
                edge = tuple(sorted([n, nbr]))
                if edge in visited_edges:
                    continue
                # start path
                path = [n, nbr]
                visited_edges.add(edge)
                prev, cur = n, nbr
                while True:
                    # extend along degree==2 chain
                    next_nodes = [p for p in G.neighbors(cur) if p != prev]
                    if len(next_nodes) != 1:
                        break
                    nxt = next_nodes[0]
                    e = tuple(sorted([cur, nxt]))
                    if e in visited_edges:
                        break
                    path.append(nxt)
                    visited_edges.add(e)
                    prev, cur = cur, nxt
                    if G.degree[cur] != 2: # type: ignore
                        break
                segments.append(path)
    return segments

def polyline_from_path(path: List[Tuple[int,int]]) -> np.ndarray:
    # Convert pixel coordinates (row, col) -> (x, y)
    pts = np.array([[c, r] for (r, c) in path], dtype=np.int32)
    # Optionally simplify later
    return pts

def simplify_polyline(pts: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    if pts.shape[0] <= 2:
        return pts
    ls = LineString(pts)
    simp = ls.simplify(epsilon, preserve_topology=False)
    simp_pts = np.array(simp.coords, dtype=np.int32)
    return simp_pts

def bbox_of_polyline(pts: np.ndarray) -> Tuple[int,int,int,int]:
    x0 = int(np.min(pts[:,0])); y0 = int(np.min(pts[:,1]))
    x1 = int(np.max(pts[:,0])); y1 = int(np.max(pts[:,1]))
    return (x0, y0, x1-x0+1, y1-y0+1)

def length_of_polyline(pts: np.ndarray) -> float:
    diffs = np.diff(pts.astype(float), axis=0)
    return float(np.sum(np.hypot(diffs[:,0], diffs[:,1])))

def extract_lines(img_gray: np.ndarray,
                  boxes: List[BBox],
                  cfg: Dict = None) -> Tuple[List[Segment], np.ndarray, np.ndarray, np.ndarray]: # type: ignore
    if cfg is None:
        cfg = {}
    progress = cfg.get('progress', True)
    def pbar(iterable, desc):
        return tqdm(iterable, desc=desc, leave=False) if progress else iterable
    # Binarize
    use_adap = cfg.get('adaptive', False)
    bin_img = binarize(img_gray, use_adaptive=use_adap,
                       block_size=cfg.get('adaptive_block', 41),
                       C=cfg.get('adaptive_C', 7))

    # C. mask symbols/text FIRST on the binary image, then skeletonize (fewer pixels)
    mask = build_symbol_text_mask(
        img_gray.shape[0], img_gray.shape[1], boxes,
        inflate=cfg.get('mask_inflate', 2),
        erode_ksize=cfg.get('mask_erode_ksize', 3),
        erode_iter=cfg.get('mask_erode_iter', 1)
    )
    bin_img_masked = bin_img.copy()
    bin_img_masked[mask > 0] = 0

    # Skeletonize and prune (fast thinning if available)
    skel_masked = fast_thinning(bin_img_masked)
    skel_masked = prune_spurs(skel_masked, Lmin=cfg.get('spur_len', 8))

    # D1/D2: derive atomic segments from pixel graph (captures H/V and curves)
    G = build_pixel_graph(skel_masked, progress=progress)
    paths = walk_edges_splitted(G, progress=progress)

    # Also capture diagonals via Hough to avoid missing faint diagonals and then merge
    hough_lines = hough_diagonals(
        skel_masked,
        min_len=cfg.get('hough_min_len', 20),
        max_gap=cfg.get('hough_max_gap', 3),
        theta_deg_tol_exclude=cfg.get('hough_exclude', 10),
        vote_thresh=cfg.get('hough_votes', 40),
        max_lines_cap=cfg.get('hough_max_lines', 200)
    )
    diag_polys = []
    for l in pbar(hough_lines, 'Verify diagonals'):
        res = verify_and_trace_diagonal(skel_masked, l,
                                        band=cfg.get('diag_band', 1),
                                        min_overlap_ratio=cfg.get('diag_overlap', 0.7))
        if res is not None:
            poly, p0, p1 = res
            diag_polys.append(poly)

    # Build initial polyline set from graph paths
    polylines = []
    for path in pbar(paths, 'Paths→polylines'):
        pts = polyline_from_path(path)
        if pts.shape[0] >= 2:
            polylines.append(pts)

    # Merge Hough diagonals with graph-derived polylines using shapely linemerge
    # Convert to LineStrings and merge
    line_geoms = [LineString(p) for p in polylines] + [LineString(p) for p in diag_polys if p.shape[0] >= 2]
    if line_geoms:
        merged = linemerge(line_geoms)
        if merged.geom_type == 'LineString':
            line_geoms_merged = [merged]
        else:
            line_geoms_merged = list(merged.geoms) # type: ignore
    else:
        line_geoms_merged = []

    # Deduplicate near-parallel/overlapping lines
    line_geoms_merged = dedup_near_parallel(
        line_geoms_merged,
        tol=cfg.get('dedup_tol', 1.5),
        overlap_ratio=cfg.get('dedup_overlap', 0.8)
    )

    # Post-filter, simplify, and prepare segments
    segments: List[Segment] = []
    seg_id = 0
    min_len_px = cfg.get('min_segment_len', 10)
    epsilon = cfg.get('simplify_epsilon', 1.0)

    for lg in pbar(line_geoms_merged, 'Finalize segments'):
        pts = np.array(lg.coords, dtype=np.int32)
        if pts.shape[0] < 2:
            continue
        # Simplify
        pts = simplify_polyline(pts, epsilon=epsilon)
        if pts.shape[0] < 2:
            continue
        L = length_of_polyline(pts)
        if L < min_len_px:
            continue
        x0, y0 = pts[0,0], pts[0,1]
        x1, y1 = pts[-1,0], pts[-1,1]
        segments.append(Segment(
            id=seg_id,
            polyline=pts,
            start=(x0, y0),
            end=(x1, y1),
            bbox=bbox_of_polyline(pts),
            length_px=L,
            neighbors=[]
        ))
        seg_id += 1

    segments = merge_collinear_segments(segments, cfg)
    
    return segments, skel_masked, bin_img_masked, mask

# Save segments to JSON (convert numpy types to native Python)
def to_native(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o

# ------------------------------
# Demo / Execution
# ------------------------------

if __name__ == "__main__":
    # Adjust paths
    coco_path = "output/coco_annotation.json"
    ocr_path = "output/ocr_results.json"
    img_dir = "test"  # directory containing the image file from COCO
    img_file, W, H, boxes = load_coco(coco_path)
    img_path = os.path.join(img_dir, img_file)

    img_gray = read_gray(img_path)

    cfg = {
        'adaptive': False,          # try True if lighting varies
        'adaptive_block': 41,
        'adaptive_C': 7,
        'spur_len': 8,
        'mask_inflate': 2,
        'hough_min_len': 20,
        'hough_max_gap': 3,
        'hough_exclude': 10,
        'hough_votes': 60,
        'diag_band': 1,
        'diag_overlap': 0.7,
        'min_segment_len': 20,
        'simplify_epsilon': 1.0,
        'dedup_tol': 1.5,
        'dedup_overlap': 0.8,
        'progress': True,
        'mask_erode_ksize': 3,
        'mask_erode_iter': 1,
        'hough_max_lines': 200,
        # --- ADD THESE NEW PARAMETERS FOR MERGING ---
        'merge_angle_tol': 2.0,  # Angle (in degrees) to classify a line as H or V
        'merge_gap_tol': 15,     # Max gap (in pixels) between segments to merge
        'merge_y_tol': 5,        # Max y-difference for merging horizontal lines
        'merge_x_tol': 5,        # Max x-difference for merging vertical lines
    }

    # Load OCR text boxes (mask out text areas as well)
    _, _, _, ocr_boxes = load_coco(ocr_path)
    boxes_all = boxes + ocr_boxes

    segments, skel_masked, bin_img, mask = extract_lines(img_gray, boxes_all, cfg)

    print(f"Extracted {len(segments)} segments")
    # Quick visualization
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for seg in segments:
        pts = seg.polyline.reshape(-1,1,2)
        cv2.polylines(vis, [pts], isClosed=False, color=(0,0,255), thickness=2)

        # --- ADD THESE LINES FOR DEBUGGING ---
        # Draw a filled circle at the start point (green)
        start_point = (int(seg.start[0]), int(seg.start[1]))
        cv2.circle(vis, start_point, radius=5, color=(0, 255, 0), thickness=-1)

        # Draw a filled circle at the end point (blue)
        end_point = (int(seg.end[0]), int(seg.end[1]))
        cv2.circle(vis, end_point, radius=5, color=(255, 0, 0), thickness=-1)
        
    # Save debug images
    cv2.imwrite("output/debug_binary.png", bin_img)
    cv2.imwrite("output/debug_skeleton_masked.png", skel_masked)
    cv2.imwrite("output/debug_mask.png", mask)
    cv2.imwrite("output/debug_segments.png", vis)

    out = []
    for s in segments:
        rec = {
            "id": int(s.id),
            "polyline": s.polyline.astype(int).tolist(),
            "start": [int(s.start[0]), int(s.start[1])],
            "end": [int(s.end[0]), int(s.end[1])],
            "bbox": [int(v) for v in s.bbox],
            "length_px": float(s.length_px),
        }
        out.append(rec)

    with open("output/extracted_segments.json", "w") as f:
        json.dump(out, f, indent=2, default=to_native)