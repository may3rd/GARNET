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


# --------------------------------------------------------------------------------------------
# garnet/ extraction path: stage fixtures -> (detect) -> trace -> review gate -> Stage 6.
#
# Each step runs as the same CLI the docs show, in a subprocess, so the API cannot drift from
# what `python -m garnet.stage5b.run` does. Any of the three JSONs can be uploaded instead of
# computed: an uploaded file wins and its ML step is skipped.
# --------------------------------------------------------------------------------------------

import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import sys  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

JOBS_DIR = RUNS_DIR / "jobs"
STEM = "sheet"
# ponytail: in-memory job table, lost on restart; persist job.json if jobs must outlive the server
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
# Accepted upload name -> the fixture filename `host.load_fixtures` reads.
JSON_INPUTS = {
    "objects_json": f"{STEM}_objects.json",
    "equipment_json": f"{STEM}_equipment_bboxes.json",
    "line_numbers_json": f"{STEM}_line_numbers.json",
}


def _plan(job_dir: Path, have: set[str], review_llm: bool, weight: Path, ocr_instruments: bool) -> list[dict]:
    """Ordered steps. Skipped ML steps stay in the list so the UI shows why they did not run."""
    indir, out = job_dir / "input", job_dir / "out"
    image = indir / f"{STEM}.jpg"
    py = [sys.executable, "-m"]
    steps = [
        {"name": "detect_objects", "skipped": "objects_json" in have,
         "cmd": py + ["garnet.detect_objects", "--image", str(image), "--weight", str(weight),
                      "--out", str(indir / JSON_INPUTS["objects_json"])]},
        {"name": "detect_equipment", "skipped": "equipment_json" in have,
         "cmd": py + ["garnet.detect_equipment_bboxes", "--image", str(image),
                      "--out", str(indir / JSON_INPUTS["equipment_json"])]},
        # No line-number script exists (agent work); without an upload Stage 6 runs unlabelled.
        {"name": "line_numbers", "skipped": True, "cmd": None,
         "note": "uploaded" if "line_numbers_json" in have else "not provided — traces stay unlabelled"},
        {"name": "ocr_instruments", "skipped": not ocr_instruments, "cmd": None,
         "fn": lambda: _ocr_instruments(indir),
         "note": "" if ocr_instruments else "off", "gate_after": 1},
        {"name": "trace", "skipped": False,
         "cmd": py + ["garnet.stage5b.run", "--stem", STEM, "--indir", str(indir), "--out", str(out)]},
        {"name": "review_gate", "skipped": False,
         "cmd": py + ["garnet.stage5b.review_gate", "--stem", STEM, "--indir", str(indir), "--dir", str(out)]
         + ([] if review_llm else ["--no-llm"]), "gate_after": 2},
        # Stage 6 reads a gate-2 view of the trace files (rejected traces removed), never the
        # 5b/5c originals edited in place.
        {"name": "associations", "skipped": False, "cmd": None,
         "fn": lambda: _run_associations(job_dir), "gate_after": 3},
    ]
    for s in steps:
        s["status"] = "skipped" if s["skipped"] else "pending"
        s.setdefault("note", "uploaded" if s["skipped"] else "")
        s["log"] = ""
    return steps


def _run_job(job_id: str, start: int = 0) -> None:
    job = JOBS[job_id]
    job["status"], job["gate"] = "running", None
    for i, step in enumerate(job["steps"]):
        if i < start:
            continue
        if step["status"] == "skipped":
            gate = step.get("gate_after")
            if gate and job["pause_at_gates"]:
                job["status"], job["gate"], job["resume_at"] = "awaiting_gate", gate, i + 1
                return
            continue
        step["status"] = "running"
        if step.get("fn"):
            try:
                step["log"] = step["fn"]()
                returncode = 0
            except Exception as exc:  # the step's failure is the job's failure; keep the reason
                logger.exception("job %s: %s", job_id, step["name"])
                step["log"], returncode = f"{type(exc).__name__}: {exc}", 1
        else:
            proc = subprocess.run(step["cmd"], cwd=REPO_ROOT, capture_output=True, text=True)
            step["log"] = (proc.stdout + proc.stderr)[-4000:]
            returncode = proc.returncode
        if returncode != 0:
            step["status"] = "failed"
            job["status"] = "failed"
            job["error"] = f"{step['name']} exited {returncode}"
            logger.error("job %s: %s failed\n%s", job_id, step["name"], step["log"])
            return
        step["status"] = "completed"
        gate = step.get("gate_after")
        if gate and job["pause_at_gates"]:
            job["status"], job["gate"], job["resume_at"] = "awaiting_gate", gate, i + 1
            return
    job["status"] = "completed"


def _job_view(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = JOBS_DIR / job_id
    files = sorted(p for p in job_dir.rglob("*") if p.is_file()) if job_dir.is_dir() else []
    return {
        "id": job_id,
        "status": job["status"],
        "gate": job.get("gate"),
        "error": job.get("error"),
        "steps": [{k: s[k] for k in ("name", "status", "note", "log")} for s in job["steps"]],
        "edits": _read_edits(job_dir)["edits"],
        "artifacts": [
            {"name": str(p.relative_to(job_dir)), "url": f"/runs/jobs/{job_id}/{p.relative_to(job_dir)}"}
            for p in files
        ],
    }


@app.post("/api/garnet/jobs")
def api_garnet_job(
    file_input: UploadFile = File(...),
    objects_json: UploadFile | None = File(None),
    equipment_json: UploadFile | None = File(None),
    line_numbers_json: UploadFile | None = File(None),
    review_llm: bool = Form(False),
    ocr_instruments: bool = Form(True),
    pause_at_gates: bool = Form(True),
    weight_file: str = Form(""),
) -> dict:
    """Stage one sheet plus any supplied JSON, then run the garnet/ chain in a background thread."""
    suffix = Path(file_input.filename or "").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported image type '{suffix or file_input.filename}'")
    raw = file_input.file.read()
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image empty or over the size limit")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file. Could not decode image.")

    # Parse every JSON before writing anything: a malformed upload must fail the request, not
    # surface minutes later as a tracer crash.
    uploads = {"objects_json": objects_json, "equipment_json": equipment_json,
               "line_numbers_json": line_numbers_json}
    parsed: dict[str, dict] = {}
    for key, up in uploads.items():
        if up is None or not up.filename:
            continue
        try:
            payload = json.loads(up.file.read())
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"{key}: not valid JSON ({exc})") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
            raise HTTPException(status_code=400, detail=f"{key}: expected an object with an 'objects' list")
        parsed[key] = payload

    weight = Path(weight_file) if weight_file else Path(DEFAULT_WEIGHT)
    if not weight.is_absolute():
        weight = REPO_ROOT / weight
    try:
        weight.resolve().relative_to(WEIGHTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"weight_file must live under {WEIGHTS_DIR.name}/")
    if "objects_json" not in parsed and not weight.is_file():
        raise HTTPException(status_code=400, detail=f"Weight file not found: {weight_file or DEFAULT_WEIGHT}")

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    indir = job_dir / "input"
    indir.mkdir(parents=True)
    # The tracer reads `<stem>.jpg` only. Re-encoding keeps the pixel frame, which is what the
    # uploaded JSON coordinates are in. Max quality so the mask sees what the user uploaded.
    cv2.imwrite(str(indir / f"{STEM}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    for key, payload in parsed.items():
        (indir / JSON_INPUTS[key]).write_text(json.dumps(payload))

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "pause_at_gates": pause_at_gates, "steps": _plan(job_dir, set(parsed), review_llm, weight, ocr_instruments)}
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    logger.info("garnet job %s: uploaded=%s review_llm=%s", job_id, sorted(parsed), review_llm)
    return _job_view(job_id)


@app.get("/api/garnet/jobs/{job_id}")
def api_garnet_job_status(job_id: str) -> dict:
    return _job_view(job_id)


INSTRUMENT_CLASSES = {"instrument tag", "instrument dcs", "instrument logic"}


def _ocr_instruments(indir: Path) -> str:
    """Fill `text` on instrument objects that have none, by EasyOCR on their crop.

    The objects file as it arrived (upload or YOLO) is kept as `.source.json`; the staged file
    the tracer reads gets the text plus `text_source: "easyocr"`. Existing text is never
    overwritten — a transcribed tag outranks a machine read.
    """
    path = indir / JSON_INPUTS["objects_json"]
    source = indir / f"{STEM}_objects.source.json"
    if not source.exists():
        source.write_text(path.read_text())
    payload = json.loads(source.read_text())
    image = cv2.imread(str(indir / f"{STEM}.jpg"))
    if image is None:
        raise RuntimeError("staged raster unreadable")

    rows, owners = [], []
    for obj in payload["objects"]:
        if obj.get("class_name") not in INSTRUMENT_CLASSES or str(obj.get("text") or "").strip():
            continue
        bb = obj["bbox"]
        rows.append({"Index": obj["id"], "Object": obj["class_name"], "Text": "",
                     "Left": int(bb["x_min"]), "Top": int(bb["y_min"]),
                     "Width": int(bb["x_max"] - bb["x_min"]), "Height": int(bb["y_max"] - bb["y_min"])})
        owners.append(obj)
    read = read_object_text(image, rows)
    for row, obj in zip(rows, owners):
        if row["Text"]:
            obj["text"] = row["Text"]
            obj["text_source"] = "easyocr"
    path.write_text(json.dumps(payload))
    return f"read {read} of {len(rows)} instrument objects without text"


# --- Text edits ------------------------------------------------------------------------------
# A human correction is an overlay, never a rewrite: Stage 6 artifacts stay as produced,
# `edits.json` holds the current map, `edits_history.jsonl` every save, and
# `out/reviewed_associations.json` is Stage 6 with the edits applied. Stage 6 uses text only
# for display (matching is geometric), so applying an edit needs no re-run.

class EditsRequest(BaseModel):
    # object id -> corrected text. An empty string is a deliberate "this box has no text".
    edits: dict[str, str]


def _read_edits(job_dir: Path) -> dict:
    p = job_dir / "edits.json"
    return json.loads(p.read_text()) if p.is_file() else {"version": 0, "edits": {}}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _known_ids(job_dir: Path) -> set[str]:
    """Every object id a review may name: staged objects, stage 4 objects, Stage 6 items."""
    known: set[str] = set()
    for rel in (f"input/{JSON_INPUTS['objects_json']}", "out/stage4_objects.json"):
        path = job_dir / rel
        if path.is_file():
            known |= {o["id"] for o in json.loads(path.read_text())["objects"] if "id" in o}
    stage6 = job_dir / "out" / "stage6_trace_associations.json"
    if stage6.is_file():
        for group in json.loads(stage6.read_text())["associations"].values():
            for status in ("accepted", "rejected"):
                known |= {item["id"] for item in group.get(status, []) if "id" in item}
    return known


def _check_ids(ids, known: set[str]) -> None:
    unknown = sorted(set(ids) - known)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown id(s): {unknown[:10]}")


def _append_edits(job_dir: Path, edits: dict[str, str]) -> None:
    prev = _read_edits(job_dir)
    record = {"version": prev["version"] + 1, "saved_at": _now(), "edits": edits}
    (job_dir / "edits.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    with (job_dir / "edits_history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_gates(job_dir: Path) -> dict:
    p = job_dir / "gates.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def _record_gate(job_dir: Path, gate: int, decision: dict) -> None:
    gates = _read_gates(job_dir)
    record = {"gate": gate, "confirmed_at": _now(), **decision}
    gates[str(gate)] = record
    (job_dir / "gates.json").write_text(json.dumps(gates, indent=2, ensure_ascii=False))
    with (job_dir / "gates_history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_reviewed(job_dir: Path) -> None:
    """Stage 6 + text edits + Gate 3 line-number decisions -> out/reviewed_associations.json."""
    stage6 = json.loads((job_dir / "out" / "stage6_trace_associations.json").read_text())
    edits = _read_edits(job_dir)
    decisions = _read_gates(job_dir).get("3", {}).get("decisions", {})
    for name, group in stage6["associations"].items():
        for status in ("accepted", "rejected"):
            for item in group.get(status, []):
                if item.get("id") in edits["edits"]:
                    item["text_original"] = item.get("text", "")
                    item["text"] = edits["edits"][item["id"]]
                    item["normalized_text"] = item["text"]
                    item["text_source"] = "human_edit"
        if name != "line_numbers":
            continue
        # A human decision moves an item between the lists; the system's verdict is kept.
        items = [(st, it) for st in ("accepted", "rejected") for it in group.get(st, [])]
        group["accepted"], group["rejected"] = [], []
        for st, it in items:
            verdict = decisions.get(it.get("id"))
            if verdict:
                it["system_status"] = st
                it["review_state"] = verdict
                it["review_source"] = "human"
                st = "accepted" if verdict == "accepted" else "rejected"
            group[st].append(it)
    stage6["edits_version"] = edits["version"]
    stage6["gate3_confirmed"] = "3" in _read_gates(job_dir)
    (job_dir / "out" / "reviewed_associations.json").write_text(json.dumps(stage6, indent=2, ensure_ascii=False))


def _check_boxes(job_dir: Path, boxes: list[Box]) -> None:
    """A reviewer's box must have area and lie on the raster; anything else is a UI bug."""
    h, w = cv2.imread(str(job_dir / "input" / f"{STEM}.jpg"), cv2.IMREAD_GRAYSCALE).shape[:2]
    for b in boxes:
        if not (0 <= b.x_min < b.x_max <= w and 0 <= b.y_min < b.y_max <= h):
            raise HTTPException(status_code=400, detail=f"Box {b.model_dump()} is empty or off the {w}x{h} raster")


def _apply_gate1(job_dir: Path, rejected: list[str], edits: dict[str, str],
                 added: list[AddedObject] = (), moved: dict[str, Box] | None = None,
                 reclassed: dict[str, str] | None = None) -> None:
    """Gate 1 onto the staged objects: drop rejected, retext, re-box moved, append added.

    The pre-gate file is kept. Moved boxes carry `bbox_original`; added ones get `human_NNNNNN`
    ids and `source: "human"`, so a reviewer's box is never mistaken for a detection.
    """
    moved = moved or {}
    reclassed = reclassed or {}
    path = job_dir / "input" / JSON_INPUTS["objects_json"]
    keep = job_dir / "input" / f"{STEM}_objects.pre_gate1.json"
    if not keep.exists():
        keep.write_text(path.read_text())
    payload = json.loads(keep.read_text())
    drop = set(rejected)
    objects = []
    for obj in payload["objects"]:
        if obj["id"] in drop:
            continue
        if obj["id"] in edits:
            obj["text_original"] = obj.get("text", "")
            obj["text"] = edits[obj["id"]]
            obj["text_source"] = "human_edit"
        if obj["id"] in reclassed and reclassed[obj["id"]].strip() != obj["class_name"]:
            obj["class_name_original"] = obj["class_name"]
            obj["class_name"] = reclassed[obj["id"]].strip()
            obj["class_source"] = "human_edit"
        if obj["id"] in moved:
            obj["bbox_original"] = obj["bbox"]
            obj["bbox"] = moved[obj["id"]].model_dump()
            obj["bbox_source"] = "human_edit"
        objects.append(obj)
    for n, a in enumerate(added, start=1):
        obj = {"id": f"human_{n:06d}", "class_name": a.class_name, "confidence": 1.0,
               "bbox": a.bbox.model_dump(), "source": "human"}
        if a.text:
            obj["text"], obj["text_source"] = a.text, "human_edit"
        objects.append(obj)
    payload["objects"] = objects
    path.write_text(json.dumps(payload))


TRACE_FILES = ("stage5b_trace_results.json", "stage5c_branch_trace_results.json",
               "stage5b_branch_trace_results.json")


def _run_associations(job_dir: Path) -> str:
    """Stage 6 against out/gate2/: copies of the trace inputs minus Gate 2's rejected traces."""
    out, view = job_dir / "out", job_dir / "out" / "gate2"
    view.mkdir(exist_ok=True)
    rejected = set(_read_gates(job_dir).get("2", {}).get("rejected", []))
    for name in ("stage4_objects.json", "stage5_connection_ports.json", *TRACE_FILES):
        src = out / name
        if not src.is_file():
            continue
        payload = json.loads(src.read_text())
        if name == "stage5b_trace_results.json":
            payload = {k: v for k, v in payload.items() if k not in rejected}
        elif name in TRACE_FILES:
            payload["branches"] = {k: v for k, v in payload["branches"].items() if k not in rejected}
        (view / name).write_text(json.dumps(payload))
    cmd = [sys.executable, "-m", "garnet.stage5b.run_associations", "--stem", STEM,
           "--indir", str(job_dir / "input"), "--dir", str(view), "--out", str(out)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    log = (proc.stdout + proc.stderr)[-4000:]
    if proc.returncode != 0:
        raise RuntimeError(f"run_associations exited {proc.returncode}\n{log}")
    _write_reviewed(job_dir)
    return f"{len(rejected)} trace(s) rejected at Gate 2\n{log}"


def _trace_ids(job_dir: Path) -> set[str]:
    out = job_dir / "out"
    ids: set[str] = set()
    for name in TRACE_FILES:
        p = out / name
        if p.is_file():
            payload = json.loads(p.read_text())
            ids |= set(payload if name == "stage5b_trace_results.json" else payload.get("branches", {}))
    return ids


class Box(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class AddedObject(BaseModel):
    class_name: str = Field(min_length=1)
    bbox: Box
    text: str = ""


class GateRequest(BaseModel):
    # Gate 1: object ids to drop before tracing. Gate 2: trace ids to drop before Stage 6.
    rejected: list[str] = []
    # Gate 1 only: boxes a reviewer drew, and new geometry for existing ids.
    added: list[AddedObject] = []
    moved: dict[str, Box] = {}
    # Gate 1 only: object id -> corrected class.
    reclassed: dict[str, str] = {}
    # Gates 1 and 3: object id -> corrected text.
    edits: dict[str, str] = {}
    # Gate 3: line-number id -> "accepted" | "rejected".
    decisions: dict[str, str] = {}


@app.post("/api/garnet/jobs/{job_id}/gates/{gate}")
def api_garnet_gate(job_id: str, gate: int, payload: GateRequest) -> dict:
    """Confirm the gate the job is parked on, apply its decisions, and resume."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != "awaiting_gate" or job.get("gate") != gate:
            raise HTTPException(status_code=409, detail=f"Job is not waiting at Gate {gate}")
        job_dir = JOBS_DIR / job_id
        if gate == 2:
            _check_ids(payload.rejected, _trace_ids(job_dir))
        else:
            _check_ids([*payload.rejected, *payload.edits, *payload.decisions, *payload.moved, *payload.reclassed],
                       _known_ids(job_dir))
        if gate != 1 and (payload.added or payload.moved or payload.reclassed):
            raise HTTPException(status_code=400, detail="Boxes can be added, moved or reclassed at Gate 1 only")
        if any(not c.strip() for c in payload.reclassed.values()):
            raise HTTPException(status_code=400, detail="reclassed: class names must be non-empty")
        if gate == 1:
            _check_boxes(job_dir, [a.bbox for a in payload.added] + list(payload.moved.values()))
        bad = {v for v in payload.decisions.values()} - {"accepted", "rejected"}
        if bad:
            raise HTTPException(status_code=400, detail=f"decisions must be accepted/rejected, got {sorted(bad)}")
        if gate == 2 and payload.edits:
            raise HTTPException(status_code=400, detail="Gate 2 takes trace rejections only")

        _record_gate(job_dir, gate, payload.model_dump())
        if gate == 1:
            _apply_gate1(job_dir, payload.rejected, payload.edits, payload.added, payload.moved, payload.reclassed)
        if gate == 3:
            if payload.edits:
                _append_edits(job_dir, {**_read_edits(job_dir)["edits"], **payload.edits})
            _write_reviewed(job_dir)
        job["status"] = "running"
    threading.Thread(target=_run_job, args=(job_id, job["resume_at"]), daemon=True).start()
    return _job_view(job_id)


@app.put("/api/garnet/jobs/{job_id}/edits")
def api_garnet_edits(job_id: str, payload: EditsRequest) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail="Job has not completed")
        job_dir = JOBS_DIR / job_id
        _check_ids(payload.edits, _known_ids(job_dir))
        _append_edits(job_dir, payload.edits)
        _write_reviewed(job_dir)
    return _job_view(job_id)
