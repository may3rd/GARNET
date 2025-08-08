
import os
import json
import warnings
from contextlib import redirect_stdout, redirect_stderr

import cv2
import numpy as np
import easyocr


# =========================
# Utilities
# =========================
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")


def bbox_from_quad(quad):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def is_valid_text(txt, min_len=2):
    t = txt.strip()
    return len(t) >= min_len and any(ch.isalnum() for ch in t)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def merge_results(items, iou_thresh=0.5):
    """Merge overlapping boxes (from multiple rotations) keeping highest score for same text."""
    if not items:
        return []
    items = sorted(items, key=lambda r: r["score"], reverse=True)

    def norm(t):  # normalize for comparison
        return "".join(ch for ch in t.strip() if ch.isalnum()).upper()

    merged = []
    for r in items:
        if any(iou_xyxy(r["bbox"], m["bbox"]) >= iou_thresh and norm(r["text"]) == norm(m["text"]) for m in merged):
            continue
        merged.append(r)
    return merged


def rotate_back_quad(quad, rot, W, H):
    """
    Map quad points from a rotated image back to original orientation.
    rot: 'none' | 'cw' (90° clockwise) | 'ccw' (90° counterclockwise)
    W,H are the width/height of the *original (upscaled)* image BEFORE rotation.
    """
    q = np.array(quad, dtype=np.float32)
    if rot == "none":
        return q
    if rot == "cw":
        # Forward (orig -> cw):  x_r = H - 1 - y_o ; y_r = x_o
        # Inverse (cw -> orig):  x_o = y_r ; y_o = H - 1 - x_r
        x_r, y_r = q[:, 0], q[:, 1]
        x_o = y_r
        y_o = H - 1 - x_r
        return np.stack([x_o, y_o], axis=1)
    if rot == "ccw":
        # Forward (orig -> ccw): x_r = y_o ; y_r = W - 1 - x_o
        # Inverse (ccw -> orig): x_o = W - 1 - y_r ; y_o = x_r
        x_r, y_r = q[:, 0], q[:, 1]
        x_o = W - 1 - y_r
        y_o = x_r
        return np.stack([x_o, y_o], axis=1)
    return q


def visualize_overlay(img_bgr, items, out_path, alpha=0.35, fill_color=(0, 255, 255)):
    """Draw filled translucent polygons over text regions (no labels)."""
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    overlay = img_bgr.copy()
    for r in items:
        quad = np.array(r.get("quad"), dtype=np.int32)
        if quad.shape != (4, 2):
            x1, y1, x2, y2 = map(int, r["bbox"])
            quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        cv2.fillPoly(overlay, [quad], fill_color)
    vis = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0.0)
    cv2.imwrite(out_path, vis)
    return out_path


def remove_long_lines_bgr(rgb_img, h_kernel=35, v_kernel=35):
    """Optional: remove long horizontal/vertical lines via morphological opening + inpaint."""
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 33, 5)
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel, 1)))
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel)))
    mask = cv2.bitwise_or(horiz, vert)
    inpainted = cv2.inpaint(rgb_img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted


def export_results_to_json(items, out_path, image_size, image_file_name):
    """Write minimal JSON with image info + annotations [{bbox, text, score}]."""
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = {
        "image": {
            "file_name": os.path.basename(image_file_name),
            "width": int(image_size[0]),
            "height": int(image_size[1]),
        },
        "annotations": [],
    }
    for item in items:
        data["annotations"].append(
            {"bbox": [float(x) for x in item["bbox"]], "text": item["text"], "score": float(item["score"])}
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


# =========================
# Public API
# =========================
def text_extract_pid(
    img_path,
    # runtime
    use_gpu=False,
    # OCR tuning
    upscale_factor=1.0,
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/().,\"'×%:° ",
    min_text_len=2,
    min_score=0.20,
    min_height_frac=0.004,
    text_threshold=0.7,
    low_text=0.4,
    link_threshold=0.5,
    canvas_size=2560,
    # preproc
    remove_lines=False,
    h_kernel=35,
    v_kernel=35,
    # merge
    iou_thresh=0.5,
    # outputs
    visualize=False,
    overlay_out=None,
    alpha=0.35,
    fill_color=(0, 255, 255),
    export_json=False,
    json_out=None,
):
    """
    Run EasyOCR on a full P&ID image (with vertical-text handling), merge results, and optionally
    export to JSON and/or visualize overlays.

    Returns:
        merged_items: list of dicts { 'quad': [[x,y]...], 'bbox': [x1,y1,x2,y2], 'text': str, 'score': float }
    """
    # ---- init reader quietly ----
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            reader = easyocr.Reader(["en"], gpu=use_gpu)

    # ---- load image ----
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h0, w0 = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---- optional upscale ----
    if upscale_factor != 1.0:
        rgb_up = cv2.resize(
            rgb,
            (int(w0 * upscale_factor), int(h0 * upscale_factor)),
            interpolation=cv2.INTER_CUBIC if upscale_factor > 1.0 else cv2.INTER_AREA,
        )
    else:
        rgb_up = rgb
    H_up, W_up = rgb_up.shape[:2]

    # ---- optional line removal ----
    if remove_lines:
        rgb_up = remove_long_lines_bgr(rgb_up, h_kernel=h_kernel, v_kernel=v_kernel)

    # ---- OCR function ----
    def ocr(image):
        return reader.readtext(
            image,
            allowlist=allowlist,
            detail=1,
            paragraph=False,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            canvas_size=canvas_size,
        )

    # ---- 3 orientations (original + 90 cw + 90 ccw) ----
    results0 = ocr(rgb_up)
    results_cw = ocr(cv2.rotate(rgb_up, cv2.ROTATE_90_CLOCKWISE))
    results_ccw = ocr(cv2.rotate(rgb_up, cv2.ROTATE_90_COUNTERCLOCKWISE))

    # ---- normalize back to original coords, filter, collect ----
    items = []

    def collect(results, rot_tag):
        for (quad, text, prob) in results:
            if not is_valid_text(text, min_text_len):
                continue
            if float(prob) < float(min_score):
                continue
            q = np.array(quad, dtype=np.float32)
            if rot_tag == "none":
                q0 = q
            elif rot_tag == "cw":
                q0 = rotate_back_quad(q, "cw", W_up, H_up)
            else:  # 'ccw'
                q0 = rotate_back_quad(q, "ccw", W_up, H_up)
            if upscale_factor != 1.0:
                q0 = q0 / upscale_factor
            xyxy = bbox_from_quad(q0.tolist())
            if (xyxy[3] - xyxy[1]) < (min_height_frac * h0):
                continue
            items.append(
                {
                    "quad": [[float(x), float(y)] for x, y in q0.tolist()],
                    "bbox": [float(v) for v in xyxy],
                    "text": text.strip(),
                    "score": float(prob),
                }
            )

    collect(results0, "none")
    collect(results_cw, "cw")
    collect(results_ccw, "ccw")

    merged = merge_results(items, iou_thresh=iou_thresh)

    # ---- optional JSON export ----
    if export_json and json_out:
        export_results_to_json(merged, json_out, (w0, h0), img_path)

    # ---- optional visualization ----
    if visualize and overlay_out:
        visualize_overlay(img, merged, overlay_out, alpha=alpha, fill_color=fill_color)

    return merged


# Optional CLI entry for quick testing:
if __name__ == "__main__":
    # Example usage; adjust paths as needed
    results = text_extract_pid(
        img_path="test/!test01.png",
        use_gpu=False,
        upscale_factor=1.0,
        visualize=True,
        overlay_out="output/ocr_overlay.png",
        export_json=True,
        json_out="output/ocr_results.json",
    )
    print(f"Detections: {len(results)}")