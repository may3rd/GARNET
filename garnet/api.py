#!/usr/bin/env python
"""FastAPI service for the frontend — detection path only.

Mirrors the request/response shapes of `backend/api.py`'s legacy `/api/detect` path so the
existing React app works against this tree unchanged, but it wires ONLY the detection path:
detection plus `text_OCR` over the text-bearing boxes. No pipeline jobs, no review state, no
Excel/PDF export. Detection runs through `garnet.detect_objects.detect`, the same code the CLI
uses; `text_OCR=true` then reads instrument tags, line numbers and connections through
`garnet.ocr_objects`.

Run it with `python garnet/main.py`, or directly:
    .venv/bin/python -m uvicorn garnet.api:app --reload --port 8090

Port 8090 is the one `backend/api.py` uses, so the frontend reaches this app with no proxy
change — but only ONE of the two can be running. The pipeline/review/export views have no
routes here and will 404 while this app holds the port.

Endpoints:
    POST   /api/detect                             run detection (+ optional OCR), store it
    GET    /api/results/{id}                       fetch a stored result
    POST   /api/results/{id}/objects               add a box
    PATCH  /api/results/{id}/objects/{index}       edit a box
    DELETE /api/results/{id}/objects/{index}       remove a box
    GET    /api/health, /api/models, /api/weight-files, /api/config-files
    GET    /runs/detect/<file>                     the stored raster (static)
"""

from __future__ import annotations

import logging
import math
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from garnet.detect_objects import DEFAULT_WEIGHT, REPO_ROOT, detect
from garnet.ocr_objects import read_object_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("garnet.api")

WEIGHTS_DIR = REPO_ROOT / "backend" / "yolo_weights"
RUNS_DIR = REPO_ROOT / "output" / "api"
DETECT_DIR = RUNS_DIR / "detect"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Results live in memory for the editing session, exactly as the backend does it. Capped so a
# long-lived dev server cannot grow without bound; oldest result is evicted first.
MAX_RESULTS = 50
RESULTS: "OrderedDict[str, dict]" = OrderedDict()
RESULTS_LOCK = threading.Lock()

app = FastAPI(title="GARNET detection API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
DETECT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


class CreateObjectRequest(BaseModel):
    Object: str = Field(min_length=1)
    Left: float
    Top: float
    Width: float = Field(gt=0)
    Height: float = Field(gt=0)
    Text: str = ""
    CategoryID: int | None = None
    ObjectID: int | None = None
    Score: float = Field(default=1.0, ge=0.0, le=1.0)


class UpdateObjectRequest(BaseModel):
    Object: str | None = None
    Left: float | None = None
    Top: float | None = None
    Width: float | None = None
    Height: float | None = None
    Text: str | None = None
    ReviewStatus: str | None = None


def _weight_files() -> list[str]:
    """Weight files, repo-root-relative so they can be handed straight back to --weight."""
    if not WEIGHTS_DIR.is_dir():
        return []
    found = [p for p in WEIGHTS_DIR.iterdir() if p.suffix in {".pt", ".onnx"}]
    return sorted(str(p.relative_to(REPO_ROOT)) for p in found)


def _get_result(result_id: str) -> dict:
    result = RESULTS.get(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


def _to_frontend_objects(objects: list[dict]) -> list[dict]:
    """detect() objects -> the row shape the React table/canvas expects.

    ObjectID restarts at 1 per class, so the rows read "gate valve - no. 3" the way the backend
    numbers them. `Text` carries that label because nothing here reads tag text — OCR is not
    wired into this app.
    """
    per_class: dict[int, int] = {}
    rows = []
    for obj in objects:
        bb = obj["bbox"]
        category_id = int(obj["class_id"])
        object_id = per_class.get(category_id, 0) + 1
        per_class[category_id] = object_id
        rows.append(
            {
                "Index": 0,
                "Object": obj["class_name"],
                "CategoryID": category_id,
                "ObjectID": object_id,
                "Left": int(bb["x_min"]),
                "Top": int(bb["y_min"]),
                "Width": int(bb["x_max"] - bb["x_min"]),
                "Height": int(bb["y_max"] - bb["y_min"]),
                "Score": round(float(obj["confidence"]), 3),
                "Text": f"{obj['class_name']} - no. {object_id}",
                "ReviewStatus": None,
            }
        )
    rows.sort(key=lambda r: (r["CategoryID"], r["ObjectID"]))
    for index, row in enumerate(rows, start=1):
        row["Index"] = index
    return rows


@app.get("/")
async def root() -> dict:
    return {"service": "garnet-detect-api", "docs": "/docs", "detect": "/api/detect"}


@app.get("/api/health")
async def health() -> dict:
    weights = _weight_files()
    return {
        "status": "ok",
        "service": "garnet-detect-api",
        "models_loaded": bool(weights),
        "models_available": len(weights),
    }


@app.get("/api/models")
async def models() -> list[str]:
    # Only the Ultralytics route exists here; the backend's gemini detector is not wired.
    return ["ultralytics"]


@app.get("/api/weight-files")
async def weight_files() -> list[str]:
    return _weight_files()


@app.get("/api/config-files")
async def config_files() -> list[str]:
    # Detection classes come from the weight file, so there is no dataset YAML to choose.
    return []


@app.post("/api/detect")
# Sync, deliberately: FastAPI runs `async def` handlers ON the event loop, so seconds of
# blocking YOLO + OCR would stall every other request (health included). A plain `def` is
# dispatched to the threadpool, which is also what makes the model/reader locks meaningful.
def api_detect(
    file_input: UploadFile = File(...),
    selected_model: str = Form("ultralytics"),
    weight_file: str = Form(""),
    config_file: str = Form(""),
    conf_th: float = Form(0.8),
    image_size: int = Form(640),
    overlap_ratio: float = Form(0.2),
    postprocess_type: str = Form("GREEDYNMM"),
    postprocess_match_metric: str = Form("IOS"),
    postprocess_match_threshold: float = Form(0.1),
    text_OCR: bool = Form(False),
) -> dict:
    """Run SAHI + YOLO detection on one uploaded raster and store the editable result."""
    if selected_model != "ultralytics":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{selected_model}'. This service only wires ultralytics.",
        )
    if not 0.0 <= conf_th <= 1.0:
        raise HTTPException(status_code=400, detail="conf_th must be in [0, 1]")
    if not 0.0 <= overlap_ratio < 1.0:
        raise HTTPException(status_code=400, detail="overlap_ratio must be in [0, 1)")
    if not 0.0 <= postprocess_match_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="postprocess_match_threshold must be in [0, 1]")
    if image_size < 1:
        raise HTTPException(status_code=400, detail="image_size must be >= 1")
    if postprocess_type not in {"GREEDYNMM", "NMM", "NMS"}:
        raise HTTPException(status_code=400, detail=f"Unsupported postprocess_type '{postprocess_type}'")
    if postprocess_match_metric not in {"IOS", "IOU"}:
        raise HTTPException(status_code=400, detail=f"Unsupported match metric '{postprocess_match_metric}'")

    suffix = Path(file_input.filename or "").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported image type '{suffix or file_input.filename}'")

    payload_bytes = file_input.file.read()
    file_input.file.close()
    if not payload_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(payload_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
        )

    resolved_weight = Path(weight_file) if weight_file else Path(DEFAULT_WEIGHT)
    if not resolved_weight.is_absolute():
        resolved_weight = REPO_ROOT / resolved_weight
    # Confine weights to the weights dir: `weight_file` is client-supplied and otherwise names
    # any file on disk for torch to load.
    try:
        resolved_weight.resolve().relative_to(WEIGHTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"weight_file must live under {WEIGHTS_DIR.name}/")
    if not resolved_weight.is_file():
        raise HTTPException(status_code=400, detail=f"Weight file not found: {weight_file or DEFAULT_WEIGHT}")

    result_id = uuid.uuid4().hex
    image_name = f"prediction_{result_id}{suffix}"
    image_path = DETECT_DIR / image_name
    image_path.write_bytes(payload_bytes)

    logger.info("detect: %s (%d bytes) weight=%s", file_input.filename, len(payload_bytes), resolved_weight.name)
    try:
        detection, image_bgr = detect(
            image_path,
            weight_path=resolved_weight,
            image_size=(int(math.ceil((image_size + 1) / 32)) - 1) * 32,
            overlap_ratio=overlap_ratio,
            conf_th=conf_th,
            postprocess_type=postprocess_type,
            postprocess_match_metric=postprocess_match_metric,
            postprocess_match_threshold=postprocess_match_threshold,
            device=None,
        )
    except FileNotFoundError as exc:
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid image file. Could not decode image.") from exc
    except Exception as exc:
        image_path.unlink(missing_ok=True)
        logger.exception("detection failed")
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

    objects = _to_frontend_objects(detection["objects"])
    if text_OCR:
        try:
            read_object_text(image_bgr, objects)
        except Exception as exc:
            # The request asked for OCR; returning boxes with placeholder labels would look
            # like a successful read of empty tags.
            logger.exception("OCR failed")
            raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc

    result = {
        "id": result_id,
        "objects": objects,
        "image_url": f"/runs/detect/{image_name}",
        "image_width": detection["image_width"],
        "image_height": detection["image_height"],
        "count": len(objects),
    }
    with RESULTS_LOCK:
        RESULTS[result_id] = result
        while len(RESULTS) > MAX_RESULTS:
            old_id, old = RESULTS.popitem(last=False)
            (DETECT_DIR / Path(old["image_url"]).name).unlink(missing_ok=True)
            # The result itself is gone too, so a client still holding it 404s either way —
            # but say it, rather than letting an open canvas lose its raster silently.
            logger.warning("evicted result %s and its raster (cap %d)", old_id, MAX_RESULTS)
    return result


@app.get("/api/results/{result_id}")
async def api_get_result(result_id: str) -> dict:
    with RESULTS_LOCK:
        return _get_result(result_id)


@app.post("/api/results/{result_id}/objects")
async def api_create_object(result_id: str, payload: CreateObjectRequest) -> dict:
    with RESULTS_LOCK:
        result = _get_result(result_id)
        objects = result["objects"]

        category_id = payload.CategoryID
        if category_id is None:
            match = next((o for o in objects if o["Object"] == payload.Object), None)
            category_id = match["CategoryID"] if match else 0
        object_id = payload.ObjectID
        if object_id is None:
            object_id = sum(1 for o in objects if o["CategoryID"] == category_id) + 1

        new_obj = {
            "Index": max((o["Index"] for o in objects), default=0) + 1,
            "Object": payload.Object,
            "CategoryID": int(category_id),
            "ObjectID": int(object_id),
            "Left": int(math.floor(payload.Left)),
            "Top": int(math.floor(payload.Top)),
            "Width": int(math.ceil(payload.Width)),
            "Height": int(math.ceil(payload.Height)),
            "Score": float(payload.Score),
            "Text": payload.Text or payload.Object,
            "ReviewStatus": None,
        }
        objects.append(new_obj)
        result["count"] = len(objects)
        return new_obj


@app.patch("/api/results/{result_id}/objects/{obj_index}")
async def api_patch_object(result_id: str, obj_index: int, payload: UpdateObjectRequest) -> dict:
    with RESULTS_LOCK:
        result = _get_result(result_id)
        target = next((o for o in result["objects"] if o["Index"] == obj_index), None)
        if target is None:
            raise HTTPException(status_code=404, detail="Object not found")
        updates = payload.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No valid fields to update")
        target.update(updates)
        return target


@app.delete("/api/results/{result_id}/objects/{obj_index}")
async def api_delete_object(result_id: str, obj_index: int) -> dict:
    with RESULTS_LOCK:
        result = _get_result(result_id)
        remaining = [o for o in result["objects"] if o["Index"] != obj_index]
        if len(remaining) == len(result["objects"]):
            raise HTTPException(status_code=404, detail="Object not found")
        result["objects"] = remaining
        result["count"] = len(remaining)
        return {"status": "deleted"}
