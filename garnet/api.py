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
from typing import Literal

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
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import sys  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from garnet.stage5b.host import LINE_NUMBER_CLASSES, load_fixtures  # noqa: E402

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


def _persist(job_id: str) -> None:
    """Write the job's state next to its files, so a saved bundle (or the folder) can be reloaded.

    Steps are stored without their callables; _plan rebuilds those from `plan`.
    """
    job = JOBS[job_id]
    state = {k: v for k, v in job.items() if k != "steps"}
    state["steps"] = [{k: st[k] for k in ("name", "status", "note", "log")} for st in job["steps"]]
    state["version"] = 1
    (JOBS_DIR / job_id / "job.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))


def _run_job(job_id: str, start: int = 0) -> None:
    try:
        _run_job_inner(job_id, start)
    finally:
        _persist(job_id)


def _write_gate1_line_numbers(job_dir: Path) -> None:
    """Gate 1's read-only line-number view, merged by the tracer's own `load_fixtures`.

    Built exactly as Stage 5b will see it: with an upload, its labels are the master list and
    detector line-number boxes are dropped; without one, the detector boxes stand.
    """
    (job_dir / "input" / "gate1_line_numbers.json").write_text(json.dumps({"objects": _line_number_rows(job_dir)}))


def _is_ln(class_name) -> bool:
    return str(class_name or "").strip().lower() in LINE_NUMBER_CLASSES


def _line_number_rows(job_dir: Path) -> list[dict]:
    objects, _ = load_fixtures(job_dir / "input", STEM)
    return [o for o in objects if _is_ln(o.get("class_name"))]


def _run_job_inner(job_id: str, start: int = 0) -> None:
    job = JOBS[job_id]
    job["status"], job["gate"] = "running", None
    for i, step in enumerate(job["steps"]):
        if i < start:
            continue
        if step.get("gate_after") == 1:
            _write_gate1_line_numbers(JOBS_DIR / job_id)
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
        "source_name": job.get("source_name", ""),
        "error": job.get("error"),
        "steps": [{**{k: s[k] for k in ("name", "status", "note", "log")}, "gate_after": s.get("gate_after")}
                  for s in job["steps"]],
        "edits": _read_edits(job_dir)["edits"],
        "gates_confirmed": sorted(int(k) for k in _read_gates(job_dir)),
        "history": _read_history(job_dir),
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
        JOBS[job_id] = {
            "status": "queued", "pause_at_gates": pause_at_gates,
            "source_name": file_input.filename or "",
            # Everything _plan needs, so a reloaded job gets the same steps back.
            "plan": {"have": sorted(parsed), "review_llm": review_llm, "ocr_instruments": ocr_instruments,
                     "weight": str(weight.relative_to(REPO_ROOT)) if weight.is_relative_to(REPO_ROOT) else str(weight)},
            "steps": _plan(job_dir, set(parsed), review_llm, weight, ocr_instruments),
        }
        _persist(job_id)
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    logger.info("garnet job %s: uploaded=%s review_llm=%s", job_id, sorted(parsed), review_llm)
    return _job_view(job_id)


@app.get("/api/garnet/jobs/{job_id}")
def api_garnet_job_status(job_id: str) -> dict:
    return _job_view(job_id)


# Classes whose box text the OCR step reads. Connectors carry the drawing they continue on,
# which is what links sheets. (Step keeps its historical name `ocr_instruments` so saved bundles
# still load.)
INSTRUMENT_CLASSES = {"instrument tag", "instrument dcs", "instrument logic"}
CONNECTOR_CLASSES = {"page connection", "utility connection"}
OCR_CLASSES = INSTRUMENT_CLASSES | CONNECTOR_CLASSES


def _ocr_instruments(indir: Path) -> str:
    """Fill `text` on instrument and connector objects that have none, by EasyOCR on their crop.

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
        if obj.get("class_name") not in OCR_CLASSES or str(obj.get("text") or "").strip():
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
    return f"read {read} of {len(rows)} instrument / connector objects without text"


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
    for rel in (f"input/{JSON_INPUTS['objects_json']}", "input/gate1_line_numbers.json", "out/stage4_objects.json"):
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


# Association groups whose items a reviewer may move to another trace at Gate 3.
REASSIGNABLE = ("line_numbers", "instrument_tags")


def _write_reviewed(job_dir: Path) -> None:
    """Stage 6 + text edits + Gate 3 line-number decisions -> out/reviewed_associations.json."""
    stage6 = json.loads((job_dir / "out" / "stage6_trace_associations.json").read_text())
    edits = _read_edits(job_dir)
    gate3 = _read_gates(job_dir).get("3", {})
    decisions = gate3.get("decisions", {})
    reassigned = gate3.get("reassigned", {})
    for name, group in stage6["associations"].items():
        for status in ("accepted", "rejected"):
            for item in group.get(status, []):
                if item.get("id") in edits["edits"]:
                    item["text_original"] = item.get("text", "")
                    item["text"] = edits["edits"][item["id"]]
                    item["normalized_text"] = item["text"]
                    item["text_source"] = "human_edit"
        if name not in REASSIGNABLE:
            continue
        # A human decision moves an item between the lists; the system's verdict is kept.
        items = [(st, it) for st in ("accepted", "rejected") for it in group.get(st, [])]
        group["accepted"], group["rejected"] = [], []
        for st, it in items:
            verdict = decisions.get(it.get("id"))
            target = reassigned.get(it.get("id"))
            if target and target != it.get("trace_id"):
                # Keep what the system chose; drop geometry that described the old trace.
                it["system_assignment"] = {k: it.get(k) for k in (
                    "trace_id", "trace_kind", "segment_index", "projected_xy", "distance_px", "t",
                    "trace_distance_px", "reason")}
                it["trace_id"] = target
                for k in ("trace_kind", "segment_index", "projected_xy", "distance_px", "t", "trace_distance_px", "reason"):
                    it.pop(k, None)
                it["trace_assignment"] = "human"
                verdict = "accepted"
            if verdict:
                it["system_status"] = st
                it["review_state"] = verdict
                it["review_source"] = "human"
                st = "accepted" if verdict == "accepted" else "rejected"
            group[st].append(it)
    # Derived lists: recompute so they agree with the reviewed assignments.
    stage6["unresolved"]["unattached_line_numbers"] = stage6["associations"]["line_numbers"]["rejected"]
    stage6["unresolved"]["unattached_instrument_tags"] = stage6["associations"]["instrument_tags"]["rejected"]
    labelled = {it["trace_id"] for it in stage6["associations"]["line_numbers"]["accepted"] if it.get("trace_id")}
    stage6["unresolved"]["traces_without_line_number"] = sorted(
        {e["trace_id"] for e in stage6["trace_edges"]} - labelled)
    stage6["edits_version"] = edits["version"]
    stage6["gate3_confirmed"] = "3" in _read_gates(job_dir)
    (job_dir / "out" / "reviewed_associations.json").write_text(json.dumps(stage6, indent=2, ensure_ascii=False))
    _write_test_systems(job_dir, stage6)


TEST_SYSTEMS_VERSION = 1
BOUNDARY_TYPES = {"page connection", "utility connection", "equipment"}


def _connectors(objects: dict, edges: dict, systems: dict, text_of) -> list[dict]:
    """Every page / utility connector on the sheet: its text (as reviewed), the traces that start or
    end on it, and the line numbers of those traces. Connectors no trace reached are listed too."""
    lns_on: dict[str, set[str]] = {}
    for key, sysm in systems.items():
        if not sysm["no_text"]:
            for tid in sysm["trace_ids"]:
                lns_on.setdefault(tid, set()).add(key)
    touching: dict[str, list[tuple[str, str]]] = {}
    for tid, e in edges.items():
        for role, oid in (("source", e.get("source_obj_id")), ("terminal", e.get("terminal_obj_id"))):
            if oid:
                touching.setdefault(oid, []).append((tid, role))
    out = []
    for oid, o in sorted(objects.items()):
        if o.get("class_name") not in CONNECTOR_CLASSES:
            continue
        hits = sorted(touching.get(oid, []))
        out.append({"id": oid, "type": o["class_name"], "text": text_of(oid), "bbox": o.get("bbox"),
                    "traces": [{"trace_id": t, "role": r} for t, r in hits],
                    "line_numbers": sorted({ln for t, _ in hits for ln in lns_on.get(t, ())})})
    return out


def _write_test_systems(job_dir: Path, reviewed: dict) -> None:
    """One test system per line number: the traces carrying it and everything on them.

    Membership is DIRECT only: a trace belongs to a system when an accepted line number is
    attached to it. Nothing is propagated across tees to unlabelled traces; those are listed in
    `unassigned_traces` for review. A trace carrying two different line numbers is put in both
    systems with `conflict: true` and listed under `conflicts` — never resolved by guessing.
    Line-number text is used exactly as reviewed; a label with no text gets its own system keyed
    `(no text) <id>` with `no_text: true`.
    """
    out = job_dir / "out"
    eq_path = out / "stage3_equipment_bboxes.json"
    equipment = {e["id"]: e for e in json.loads(eq_path.read_text()).get("equipment", [])} if eq_path.is_file() else {}
    edges = {e["trace_id"]: e for e in reviewed["trace_edges"]}
    assoc = reviewed["associations"]

    def on_trace(group: str) -> dict[str, list[dict]]:
        by: dict[str, list[dict]] = {}
        for it in assoc.get(group, {}).get("accepted", []):
            if it.get("trace_id"):
                by.setdefault(it["trace_id"], []).append(it)
        return by

    inline, instruments, arrows = on_trace("inline_objects"), on_trace("instrument_tags"), on_trace("flow_arrows")
    # Object text as reviewed: stage 4 text, overridden by any saved text edit.
    s4 = out / "stage4_objects.json"
    objects = {o["id"]: o for o in json.loads(s4.read_text())["objects"]} if s4.is_file() else {}
    text_edits = _read_edits(job_dir)["edits"]

    def text_of(oid: str) -> str:
        return text_edits.get(oid, str((objects.get(oid) or {}).get("text") or "").strip())

    systems: dict[str, dict] = {}
    labels_on: dict[str, set[str]] = {}
    for ln in assoc["line_numbers"]["accepted"]:
        text = str(ln.get("text") or "").strip()
        key = text or f"(no text) {ln['id']}"
        sysm = systems.setdefault(key, {"line_number": key, "no_text": not text, "labels": [], "trace_ids": set()})
        sysm["labels"].append({"id": ln["id"], "bbox": ln.get("bbox"), "trace_id": ln.get("trace_id"),
                               "review_source": ln.get("review_source"), "trace_assignment": ln.get("trace_assignment")})
        if ln.get("trace_id") in edges:
            sysm["trace_ids"].add(ln["trace_id"])
            # A textless label cannot contradict a read one, so it never makes a conflict.
            if text:
                labels_on.setdefault(ln["trace_id"], set()).add(key)

    conflicts = {t: sorted(keys) for t, keys in labels_on.items() if len(keys) > 1}

    def obj_row(it: dict) -> dict:
        return {"id": it.get("source_object_id") or it["id"], "class_name": it.get("class_name"),
                "text": it.get("text", ""), "trace_id": it["trace_id"], "bbox": it.get("bbox")}

    def equipment_row(eid: str, trace_id: str, role: str, edge: dict) -> dict:
        e = equipment.get(eid, {})
        port = edge.get("port") if role == "source" else None
        return {"id": eid, "tag": e.get("tag"), "class_name": e.get("class_name"),
                "trace_id": trace_id, "role": role, "port": port}

    result = []
    for key in sorted(systems):
        sysm = systems[key]
        traces, members = [], {"inline_objects": {}, "instruments": {}, "equipment": {}, "boundaries": {}}
        n_arrows = 0
        for tid in sorted(sysm["trace_ids"]):
            edge = edges[tid]
            traces.append({"trace_id": tid, "trace_kind": edge.get("trace_kind"),
                           "source_obj_id": edge.get("source_obj_id"), "source_obj_type": edge.get("source_obj_type"),
                           "terminal_type": edge.get("terminal_type"), "terminal_obj_id": edge.get("terminal_obj_id"),
                           "conflict": tid in conflicts, "conflict_with": [k for k in conflicts.get(tid, []) if k != key]})
            for it in inline.get(tid, []):
                members["inline_objects"].setdefault(obj_row(it)["id"], obj_row(it))
            for it in instruments.get(tid, []):
                members["instruments"].setdefault(obj_row(it)["id"], obj_row(it))
            n_arrows += len(arrows.get(tid, []))
            for role, oid, otype in (("source", edge.get("source_obj_id"), edge.get("source_obj_type")),
                                     ("terminal", edge.get("terminal_obj_id"), edge.get("terminal_type"))):
                if not oid:
                    continue
                if str(oid).startswith("equip_"):
                    members["equipment"].setdefault(oid, equipment_row(oid, tid, role, edge))
                elif otype in BOUNDARY_TYPES:
                    members["boundaries"].setdefault(oid, {"id": oid, "type": otype, "trace_id": tid, "role": role,
                                                           "text": text_of(oid)})
        result.append({
            "line_number": key,
            "no_text": sysm["no_text"],
            "labels": sysm["labels"],
            "traces": traces,
            "has_conflict": any(t["conflict"] for t in traces),
            "equipment": sorted(members["equipment"].values(), key=lambda r: r["id"]),
            "inline_objects": sorted(members["inline_objects"].values(), key=lambda r: r["id"]),
            "instruments": sorted(members["instruments"].values(), key=lambda r: r["id"]),
            "boundaries": sorted(members["boundaries"].values(), key=lambda r: r["id"]),
            "flow_arrow_count": n_arrows,
        })

    payload = {
        "version": TEST_SYSTEMS_VERSION,
        "source": "reviewed_associations.json",
        "edits_version": reviewed.get("edits_version"),
        "gate3_confirmed": reviewed.get("gate3_confirmed"),
        "rule": "direct: a trace belongs to the systems of the accepted line numbers attached to it",
        "systems": result,
        "conflicts": [{"trace_id": t, "line_numbers": k} for t, k in sorted(conflicts.items())],
        "unassigned_traces": sorted(set(edges) - {t for v in systems.values() for t in v["trace_ids"]}),
        "connectors": _connectors(objects, edges, systems, text_of),
    }
    (out / "test_systems.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _check_reassign(job_dir: Path, reassigned: dict[str, str], decisions: dict[str, str]) -> None:
    """Keys must be line numbers or instrument tags Stage 6 saw; targets must be traces Stage 6
    actually used (Gate 2 rejections are gone from trace_edges); reassign + reject contradicts."""
    stage6 = json.loads((job_dir / "out" / "stage6_trace_associations.json").read_text())
    _check_ids(reassigned, {it["id"] for name in REASSIGNABLE
                            for st in ("accepted", "rejected") for it in stage6["associations"][name].get(st, [])})
    _check_ids(reassigned.values(), {e["trace_id"] for e in stage6["trace_edges"]})
    clash = sorted(k for k in reassigned if decisions.get(k) == "rejected")
    if clash:
        raise HTTPException(status_code=400, detail=f"Reassigned and rejected at once: {clash[:10]}")


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


def _check_line_number_edits(job_dir: Path, payload: "GateRequest") -> None:
    """Gate 1 line numbers change only by reject or draw, which go to the line-number file.

    Editing, moving or reclassing one in the objects file would vanish: with a line-number
    file `merge_line_numbers` drops every line-number box there and rebuilds from the file.
    """
    ids = {r["id"] for r in _line_number_rows(job_dir)}
    for path in (_pre_gate1(job_dir, "objects_json"), job_dir / "input" / JSON_INPUTS["objects_json"]):
        if path.is_file():
            ids |= {o["id"] for o in json.loads(path.read_text())["objects"] if _is_ln(o.get("class_name"))}
    hit = sorted(ids & {*payload.edits, *payload.moved, *payload.reclassed})
    if hit or any(_is_ln(c) for c in payload.reclassed.values()):
        raise HTTPException(status_code=400, detail="Line numbers can only be rejected or drawn at Gate 1, "
                                                    f"in the Line numbers list {hit or ''}".rstrip())
    if any(_is_ln(a.class_name) and not a.text.strip() for a in payload.added):
        raise HTTPException(status_code=400, detail="added: a drawn line number needs its text")


def _apply_gate1_line_numbers(job_dir: Path, rejected: list[str], added: list[AddedObject]) -> None:
    """Rewrite the line-number file as the Gate 1 list (minus rejects, plus drawn boxes).

    Runs on the pre-gate inputs, so the base is exactly what the Line numbers tab showed. With
    no upload the first edit creates the file; the pre-gate copy then records `absent`, and
    Reopen deletes the file instead of restoring it. The file is marked `master` so an
    emptied list stays empty instead of falling back to the detector boxes.
    """
    rows = _line_number_rows(job_dir)
    drop = set(rejected) & {r["id"] for r in rows}
    new = [a for a in added if _is_ln(a.class_name)]
    if not drop and not new:
        return
    path, keep = job_dir / "input" / JSON_INPUTS["line_numbers_json"], _pre_gate1(job_dir, "line_numbers_json")
    if not keep.exists():
        keep.write_text(path.read_text() if path.is_file() else json.dumps({"absent": True}))

    def label(b: dict, text: str, score: float) -> dict:
        return {"Left": b["x_min"], "Top": b["y_min"], "Width": b["x_max"] - b["x_min"],
                "Height": b["y_max"] - b["y_min"], "Text": text, "Score": score}
    labels = [label(r["bbox"], r.get("text", ""), r.get("confidence", 1.0)) for r in rows if r["id"] not in drop]
    labels += [{**label(a.bbox.model_dump(), a.text.strip(), 1.0), "source": "human"} for a in new]
    path.write_text(json.dumps({"master": True, "objects": labels}))


def _pre_gate1(job_dir: Path, name: str) -> Path:
    """The untouched copy Gate 1 keeps of a staged input, and Reopen puts back."""
    return job_dir / "input" / JSON_INPUTS[name].replace(".json", ".pre_gate1.json")


def _check_ports(job_dir: Path, ports: dict[str, list[Port]]) -> None:
    """Ports are keyed by equipment Tag (that is how the tracer finds them), on the raster."""
    if not ports:
        return
    path = _pre_gate1(job_dir, "equipment_json")
    if not path.is_file():
        path = job_dir / "input" / JSON_INPUTS["equipment_json"]
    items = json.loads(path.read_text()).get("objects", []) if path.is_file() else []
    tags = {str(e.get("Tag") or "").strip() for e in items} - {""}
    unknown = sorted(set(ports) - tags)
    if unknown:
        raise HTTPException(status_code=400, detail=f"ports: no tagged equipment {unknown}; "
                                                    "an untagged box cannot take ports, tag it first")
    h, w = cv2.imread(str(job_dir / "input" / f"{STEM}.jpg"), cv2.IMREAD_GRAYSCALE).shape[:2]
    for tag, plist in ports.items():
        for p in plist:
            x, y = p.point_px
            if not (0 <= x < w and 0 <= y < h):
                raise HTTPException(status_code=400, detail=f"ports: {tag} point {[x, y]} is off the {w}x{h} raster")


def _apply_gate1_ports(job_dir: Path, ports: dict[str, list[Port]]) -> None:
    """Replace the port lists of the named equipment; the pre-gate file is kept for Reopen."""
    if not ports:
        return
    path, keep = job_dir / "input" / JSON_INPUTS["equipment_json"], _pre_gate1(job_dir, "equipment_json")
    if not keep.exists():
        keep.write_text(path.read_text())
    payload = json.loads(keep.read_text())
    for item in payload.get("objects", []):
        tag = str(item.get("Tag") or "").strip()
        if tag in ports:
            item["Ports_original"] = item.get("Ports") or []
            item["Ports"] = [p.model_dump() for p in ports[tag]]
            item["ports_source"] = "human_edit"
    path.write_text(json.dumps(payload))


TRACE_FILES = ("stage5b_trace_results.json", "stage5c_branch_trace_results.json",
               "stage5b_branch_trace_results.json")


def _run_associations(job_dir: Path) -> str:
    """Stage 6 against out/gate2/: copies of the trace inputs minus Gate 2's rejected traces."""
    out, view = job_dir / "out", job_dir / "out" / "gate2"
    view.mkdir(exist_ok=True)
    gate2 = _read_gates(job_dir).get("2", {})
    rejected = set(gate2.get("rejected", []))
    modified = gate2.get("modified", {})
    rejected |= {k for k, segs in modified.items() if not segs}

    def edit(tid: str, rec: dict) -> dict:
        if tid in modified:
            rec = {**rec, "segments_original": rec.get("segments"), "segments": modified[tid],
                   "geometry_source": "human_edit"}
        return rec

    for name in ("stage4_objects.json", "stage5_connection_ports.json", *TRACE_FILES):
        src = out / name
        if not src.is_file():
            continue
        payload = json.loads(src.read_text())
        if name == "stage5b_trace_results.json":
            payload = {k: edit(k, v) for k, v in payload.items() if k not in rejected}
            mask = _mask(job_dir)
            for n, segs in enumerate(gate2.get("drawn", []), start=1):
                tid = f"manual_{n:06d}"
                last = segs[-1]
                payload[tid] = {"source_obj_id": tid, "segments": segs, "terminal_type": "manual",
                                "terminal_x": last["x2"], "terminal_y": last["y2"], "status": "manual",
                                "source": "human", "mask_coverage": _coverage(mask, segs)}
        elif name in TRACE_FILES:
            payload["branches"] = {k: edit(k, v) for k, v in payload["branches"].items() if k not in rejected}
        (view / name).write_text(json.dumps(payload))
    cmd = [sys.executable, "-m", "garnet.stage5b.run_associations", "--stem", STEM,
           "--indir", str(job_dir / "input"), "--dir", str(view), "--out", str(out)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    log = (proc.stdout + proc.stderr)[-4000:]
    if proc.returncode != 0:
        raise RuntimeError(f"run_associations exited {proc.returncode}\n{log}")
    _write_reviewed(job_dir)
    return (f"{len(rejected)} trace(s) removed, {len(modified)} edited, "
            f"{len(gate2.get('drawn', []))} drawn at Gate 2\n{log}")


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


class Port(BaseModel):
    # Sides `host.load_equipment_ports` maps to a trace direction; anything else it drops.
    side: Literal["top", "bottom", "left", "right"]
    point_px: tuple[int, int]
    mark: str | None = None
    size: str | None = None
    line_number: str | None = None


class Seg(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class SnapRequest(BaseModel):
    # Polyline in image pixels, as the reviewer drew it (already orthogonalised client-side).
    points: list[tuple[int, int]] = Field(min_length=2, max_length=2000)
    snap: bool = True
    radius: int = Field(default=12, ge=0, le=40)


_MASKS: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _mask(job_dir: Path) -> np.ndarray:
    """Stage 5 pipe mask (255 = pipe), same frame as the raster. Small LRU: one per open job."""
    key = str(job_dir)
    if key not in _MASKS:
        m = cv2.imread(str(job_dir / "out" / "stage5_pipe_mask.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(status_code=409, detail="No pipe mask yet: trace has not run")
        _MASKS[key] = m > 0
        while len(_MASKS) > 4:
            _MASKS.popitem(last=False)
    _MASKS.move_to_end(key)
    return _MASKS[key]


def _samples(x1: int, y1: int, x2: int, y2: int, step: int = 2):
    n = max(1, int(max(abs(x2 - x1), abs(y2 - y1)) // step))
    return [(round(x1 + (x2 - x1) * i / n), round(y1 + (y2 - y1) * i / n)) for i in range(n + 1)]


def _on_mask(mask: np.ndarray, x: int, y: int, tol: int = 1) -> bool:
    h, w = mask.shape
    return bool(mask[max(0, y - tol):min(h, y + tol + 1), max(0, x - tol):min(w, x + tol + 1)].any())


def _coverage(mask: np.ndarray, segs: list[dict]) -> float:
    """Fraction of sample points along the segments that sit on a pipe pixel (+-1 px)."""
    pts = [pt for g in segs for pt in _samples(g["x1"], g["y1"], g["x2"], g["y2"])]
    return round(sum(_on_mask(mask, x, y) for x, y in pts) / len(pts), 3) if pts else 0.0


def _snap_points(mask: np.ndarray, pts: list[list[int]], radius: int) -> list[list[int]]:
    """Slide each axis-aligned run sideways (<= radius px) onto the offset with most pipe under it.

    Corners are shared points, so moving a horizontal run's y drags the ends of its vertical
    neighbours with it and the path stays connected. Diagonal runs are left where drawn. A run
    only moves when that strictly improves coverage; ties keep the smaller offset.
    """
    h, w = mask.shape
    pts = [list(p) for p in pts]
    for i in range(len(pts) - 1):
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        horizontal, vertical = ay == by, ax == bx
        if not (horizontal or vertical) or (ax == bx and ay == by):
            continue
        best, best_cov = 0, -1.0
        for d in sorted(range(-radius, radius + 1), key=abs):
            if horizontal:
                y = ay + d
                if not 0 <= y < h:
                    continue
                xs = range(max(0, min(ax, bx)), min(w, max(ax, bx) + 1), 2)
                cov = sum(bool(mask[y, x]) for x in xs) / max(1, len(xs))
            else:
                x = ax + d
                if not 0 <= x < w:
                    continue
                ys = range(max(0, min(ay, by)), min(h, max(ay, by) + 1), 2)
                cov = sum(bool(mask[y, x]) for y in ys) / max(1, len(ys))
            if cov > best_cov + 1e-9:
                best, best_cov = d, cov
        if horizontal:
            pts[i][1] = pts[i + 1][1] = ay + best
        else:
            pts[i][0] = pts[i + 1][0] = ax + best
    return pts


def _check_segments(job_dir: Path, segs: list[Seg]) -> None:
    h, w = cv2.imread(str(job_dir / "input" / f"{STEM}.jpg"), cv2.IMREAD_GRAYSCALE).shape[:2]
    for g in segs:
        if not all(0 <= v <= lim for v, lim in ((g.x1, w), (g.x2, w), (g.y1, h), (g.y2, h))):
            raise HTTPException(status_code=400, detail=f"Segment {g.model_dump()} is off the {w}x{h} raster")


@app.post("/api/garnet/jobs/{job_id}/trace-snap")
def api_garnet_trace_snap(job_id: str, payload: SnapRequest) -> dict:
    """Snap a drawn polyline onto the pipe mask and report how much of it lies on pipe.

    Read-only: nothing is stored. The reviewer sees the coverage before committing the trace.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = JOBS_DIR / job_id
    mask = _mask(job_dir)
    h, w = mask.shape
    pts = [[min(w - 1, max(0, x)), min(h - 1, max(0, y))] for x, y in payload.points]
    if payload.snap:
        pts = _snap_points(mask, pts, payload.radius)
    segs = [{"x1": a[0], "y1": a[1], "x2": b[0], "y2": b[1]} for a, b in zip(pts, pts[1:]) if a != b]
    return {"points": pts, "segments": segs, "coverage": _coverage(mask, segs)}


class GateRequest(BaseModel):
    # Gate 1: object ids to drop before tracing. Gate 2: trace ids to drop before Stage 6.
    rejected: list[str] = []
    # Gate 1 only: boxes a reviewer drew, and new geometry for existing ids.
    added: list[AddedObject] = []
    moved: dict[str, Box] = {}
    # Gate 1 only: object id -> corrected class.
    reclassed: dict[str, str] = {}
    # Gate 1 only: equipment Tag -> its FULL port list (replaces the uploaded one).
    ports: dict[str, list[Port]] = {}
    # Gate 2 only: hand-drawn traces, and replacement geometry for existing traces (what is left
    # after erasing; an empty list removes the trace like a rejection).
    drawn: list[list[Seg]] = []
    modified: dict[str, list[Seg]] = {}
    # Gates 1 and 3: object id -> corrected text.
    edits: dict[str, str] = {}
    # Gate 3: line-number or instrument-tag id -> trace id it belongs on (implies accepted).
    reassigned: dict[str, str] = {}
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
            _check_ids([*payload.rejected, *payload.modified], _trace_ids(job_dir))
            _check_segments(job_dir, [g for poly in payload.drawn for g in poly]
                            + [g for segs in payload.modified.values() for g in segs])
            if any(not poly for poly in payload.drawn):
                raise HTTPException(status_code=400, detail="drawn: every trace needs at least one segment")
        elif payload.drawn or payload.modified:
            raise HTTPException(status_code=400, detail="Traces can be drawn or edited at Gate 2 only")
        else:
            _check_ids([*payload.rejected, *payload.edits, *payload.decisions, *payload.moved, *payload.reclassed],
                       _known_ids(job_dir))
        if gate == 3 and payload.reassigned:
            _check_reassign(job_dir, payload.reassigned, payload.decisions)
        elif payload.reassigned:
            raise HTTPException(status_code=400, detail="Line numbers can be reassigned at Gate 3 only")
        if gate != 1 and (payload.added or payload.moved or payload.reclassed or payload.ports):
            raise HTTPException(status_code=400, detail="Boxes and ports can be edited at Gate 1 only")
        if any(not c.strip() for c in payload.reclassed.values()):
            raise HTTPException(status_code=400, detail="reclassed: class names must be non-empty")
        if gate == 1:
            _check_boxes(job_dir, [a.bbox for a in payload.added] + list(payload.moved.values()))
            _check_ports(job_dir, payload.ports)
            _check_line_number_edits(job_dir, payload)
        bad = {v for v in payload.decisions.values()} - {"accepted", "rejected"}
        if bad:
            raise HTTPException(status_code=400, detail=f"decisions must be accepted/rejected, got {sorted(bad)}")
        if gate == 2 and payload.edits:
            raise HTTPException(status_code=400, detail="Gate 2 takes trace rejections only")

        _record_gate(job_dir, gate, payload.model_dump())
        if gate == 1:
            _apply_gate1_line_numbers(job_dir, payload.rejected, payload.added)
            _apply_gate1(job_dir, payload.rejected, payload.edits,
                         [a for a in payload.added if not _is_ln(a.class_name)], payload.moved, payload.reclassed)
            _apply_gate1_ports(job_dir, payload.ports)
            _write_gate1_line_numbers(job_dir)
        if gate == 3:
            if payload.edits:
                _append_edits(job_dir, {**_read_edits(job_dir)["edits"], **payload.edits})
            _write_reviewed(job_dir)
        job["status"] = "running"
        _persist(job_id)
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


# --- Reopen a gate ----------------------------------------------------------------------------
# Going back is never destructive: everything the later stages produced is MOVED to
# history/<n>_reopen_gate<N>/ and the event is appended to gates_history.jsonl. The job is then
# parked at Gate N exactly as it was before that gate was confirmed.

# What each reopen supersedes. Gate 1 discards the whole trace; Gate 2 everything from Stage 6
# on; Gate 3 only the review overlay (regenerated from Stage 6 as produced).
_REOPEN_FILES = {
    1: ["out"],
    2: ["out/gate2", "out/stage6_trace_associations.json", "out/stage6_trace_association_summary.json",
        "out/stage6_line_number_review.json", "out/stage6_line_number_review_summary.json",
        "out/stage6_trace_association_overlay.png", "out/reviewed_associations.json", "out/test_systems.json"],
    3: ["out/reviewed_associations.json", "out/test_systems.json"],
}


def _read_history(job_dir: Path) -> list[dict]:
    """Gate confirmations and reopens, oldest first, summarised for the UI."""
    p = job_dir / "gates_history.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("event") == "reopen":
            out.append({"event": "reopen", "gate": r["gate"], "at": r["at"], "archive": r.get("archive")})
        else:
            counts = {k: len(v) for k, v in r.items() if isinstance(v, (list, dict)) and v}
            out.append({"event": "confirm", "gate": r["gate"], "at": r.get("confirmed_at"), "changes": counts})
    return out


class ReopenRequest(BaseModel):
    gate: int = Field(ge=1, le=3)


@app.post("/api/garnet/jobs/{job_id}/reopen")
def api_garnet_reopen(job_id: str, payload: ReopenRequest) -> dict:
    gate = payload.gate
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        # A failed job can go back to any gate it got past; the "reached" check below decides.
        if job["status"] not in ("awaiting_gate", "completed", "failed"):
            raise HTTPException(status_code=409, detail=f"Job is {job['status']}; wait for it to rest")
        steps = job["steps"]
        boundary = next(i for i, st in enumerate(steps) if st.get("gate_after") == gate)
        reached = all(st["status"] in ("completed", "skipped") for st in steps[: boundary + 1])
        if not reached:
            raise HTTPException(status_code=409, detail=f"Gate {gate} has not been reached")
        if job["status"] == "awaiting_gate" and job.get("gate") == gate:
            raise HTTPException(status_code=409, detail=f"Already at Gate {gate}")

        job_dir = JOBS_DIR / job_id
        hist = job_dir / "history"
        hist.mkdir(exist_ok=True)
        archive = hist / f"{len(list(hist.iterdir())) + 1:03d}_reopen_gate{gate}"
        archive.mkdir()

        def stash(rel: str) -> None:
            src = job_dir / rel
            if src.exists():
                dst = archive / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

        for rel in _REOPEN_FILES[gate]:
            stash(rel)
        # Edits are text corrections over Stage 6; they belong to the review being discarded.
        stash("edits.json")
        if gate == 1:
            # Undo Gate 1's writes to the staged objects, ports and line numbers: the pre-gate
            # copies become current; a line-number file Gate 1 created (no upload) is removed.
            for name in ("objects_json", "equipment_json", "line_numbers_json"):
                pre = _pre_gate1(job_dir, name)
                if pre.is_file():
                    stash(f"input/{JSON_INPUTS[name]}")
                    if json.loads(pre.read_text()).get("absent"):
                        pre.unlink()
                    else:
                        shutil.move(str(pre), str(job_dir / "input" / JSON_INPUTS[name]))
            _write_gate1_line_numbers(job_dir)
            _MASKS.pop(str(job_dir), None)
            (job_dir / "out").mkdir(exist_ok=True)

        gates = _read_gates(job_dir)
        (archive / "gates.json").write_text(json.dumps(gates, indent=2, ensure_ascii=False))
        kept = {k: v for k, v in gates.items() if int(k) < gate}
        (job_dir / "gates.json").write_text(json.dumps(kept, indent=2, ensure_ascii=False))
        with (job_dir / "gates_history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "reopen", "gate": gate, "at": _now(),
                                 "archive": str(archive.relative_to(job_dir))}) + "\n")

        if gate == 3:
            _write_reviewed(job_dir)  # Stage 6 as produced, minus the discarded Gate 3 review

        for st in steps[boundary + 1:]:
            if st["status"] != "skipped":
                st["status"], st["log"] = "pending", ""
        # Reopening means the reviewer wants the gates back, even on a run that went straight through.
        job.update(status="awaiting_gate", gate=gate, resume_at=boundary + 1, pause_at_gates=True, error=None)
        _persist(job_id)
    logger.info("garnet job %s: reopened Gate %d (archived to %s)", job_id, gate, archive.name)
    return _job_view(job_id)



# --- Save / load a job as one zip ----------------------------------------------------------
# The bundle is the job folder as-is (input, out, history, gate and edit records, job.json), so a
# saved result is the evidence, not a summary of it. Loading rebuilds the job under a NEW id and
# never overwrites another job.

import io  # noqa: E402
import zipfile  # noqa: E402
from fastapi.responses import PlainTextResponse, StreamingResponse  # noqa: E402

MAX_BUNDLE_BYTES = 1024 * 1024 * 1024  # uncompressed ceiling for a loaded bundle


def _safe_stem(name: str | None) -> str:
    """Upload filename -> header-safe download stem (it lands inside a quoted header)."""
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(name or "").stem)
    return stem.strip("._") or "sheet"


@app.get("/api/garnet/jobs/{job_id}/bundle")
def api_garnet_bundle(job_id: str) -> StreamingResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] in ("queued", "running"):
            raise HTTPException(status_code=409, detail="Job is running; save it once it rests")
        _persist(job_id)
        job_dir = JOBS_DIR / job_id
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(job_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(job_dir).as_posix())
    buf.seek(0)
    stem = _safe_stem(job.get("source_name"))
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{stem}_garnet_{job_id[:8]}.zip"'})


def _restore_job(job_dir: Path, state: dict) -> dict:
    """A job-table entry from a saved `job.json`: steps rebuilt from `plan`, saved statuses kept."""
    plan = state["plan"]
    saved = {st["name"]: st for st in state["steps"]}
    weight = Path(plan["weight"])
    steps = _plan(job_dir, set(plan["have"]), plan["review_llm"],
                  weight if weight.is_absolute() else REPO_ROOT / weight, plan["ocr_instruments"])
    for st in steps:
        if st["name"] in saved:
            s = saved[st["name"]]
            st.update(status=s["status"], note=s["note"], log=s["log"])
    job = {k: v for k, v in state.items() if k not in ("steps", "version")}
    job["steps"] = steps
    return job


def _load_saved_jobs() -> None:
    """Rebuild the job table from each job folder's `job.json`, so jobs survive an API restart."""
    for path in sorted(JOBS_DIR.glob("*/job.json")):
        job_id = path.parent.name
        if job_id in JOBS:
            continue
        try:
            job = _restore_job(path.parent, json.loads(path.read_text()))
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("garnet job %s: job.json unreadable, not restored (%s)", job_id, exc)
            continue
        if job["status"] in ("queued", "running"):  # its worker thread died with the old process
            job.update(status="failed", error="interrupted by an API restart; start the sheet again")
        JOBS[job_id] = job


_load_saved_jobs()


@app.post("/api/garnet/jobs/load")
def api_garnet_load(bundle: UploadFile = File(...)) -> dict:
    """Restore a saved bundle as a new job, resting where it was saved (a gate, or completed)."""
    raw = bundle.file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Not a zip file") from exc
    names = zf.namelist()
    # Trust boundary: a bundle may come from anywhere. No absolute paths, no '..', bounded size.
    for n in names:
        parts = Path(n).parts
        if n.startswith(("/", "\\")) or ".." in parts or (parts and ":" in parts[0]):
            raise HTTPException(status_code=400, detail=f"Unsafe path in bundle: {n}")
    if sum(i.file_size for i in zf.infolist()) > MAX_BUNDLE_BYTES:
        raise HTTPException(status_code=413, detail="Bundle too large")
    if "job.json" not in names or f"input/{STEM}.jpg" not in names:
        raise HTTPException(status_code=400, detail="Not a GARNET job bundle (job.json / input raster missing)")
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    try:
        state = json.loads(zf.read("job.json"))
        job = _restore_job(job_dir, state)  # builds paths only; nothing is written yet
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"job.json unreadable: {exc}") from exc
    if state.get("status") not in ("awaiting_gate", "completed", "failed"):
        raise HTTPException(status_code=400, detail=f"Bundle was saved while {state.get('status')}")

    job_dir.mkdir(parents=True)
    zf.extractall(job_dir)
    with JOBS_LOCK:
        JOBS[job_id] = job
        JOBS[job_id]["loaded_from"] = bundle.filename or ""
        _persist(job_id)
    logger.info("garnet job %s: loaded from %s (%s, gate %s)", job_id, bundle.filename, state["status"], state.get("gate"))
    return _job_view(job_id)


# --- Human-readable report -----------------------------------------------------------------

def _report_text(job_id: str) -> str:
    job = JOBS[job_id]
    job_dir = JOBS_DIR / job_id
    ts_path = job_dir / "out" / "test_systems.json"
    if not ts_path.is_file():
        raise HTTPException(status_code=409, detail="No test systems yet: Stage 6 has not run")
    ts = json.loads(ts_path.read_text())
    gates = _read_gates(job_dir)
    W = 78
    out: list[str] = []
    line = out.append

    def rule(ch: str = "-") -> None:
        line(ch * W)

    def row(label: str, value: object) -> None:
        line(f"  {label:<18}{value}")

    rule("=")
    line("GARNET TEST SYSTEMS REPORT")
    rule("=")
    row("Sheet", job.get("source_name") or "(unnamed)")
    row("Job", job_id)
    row("Generated", _now())
    row("Status", job["status"] + (f" (Gate {job.get('gate')})" if job["status"] == "awaiting_gate" else ""))
    row("Gates confirmed", ", ".join(f"Gate {g}" for g in sorted(gates)) or "none")
    row("Gate 3 confirmed", "yes" if ts.get("gate3_confirmed") else "NO - review not final")
    row("Text edits", f"version {ts.get('edits_version', 0)}")
    row("Membership rule", ts.get("rule", ""))
    line("")
    row("Systems", len(ts["systems"]))
    row("Conflicts", len(ts["conflicts"]))
    row("Unassigned traces", len(ts["unassigned_traces"]))
    line("")

    for n, sysm in enumerate(ts["systems"], start=1):
        rule("=")
        flags = [f for f, on in (("CONFLICT", sysm["has_conflict"]), ("NO TEXT", sysm["no_text"])) if on]
        line(f"[{n}] {sysm['line_number']}" + (f"   <{', '.join(flags)}>" if flags else ""))
        rule()
        line(f"  Pipe sections ({len(sysm['traces'])})")
        for t in sysm["traces"]:
            extra = f"  ** also carries: {', '.join(t['conflict_with'])}" if t["conflict"] else ""
            line(f"    - {t['trace_id']:<26} {t.get('trace_kind') or '':<7} -> {t.get('terminal_type') or '-'}{extra}")
        line(f"  Equipment ({len(sysm['equipment'])})")
        for e in sysm["equipment"]:
            line(f"    - {e.get('tag') or e['id']:<14} {e.get('class_name') or '':<18} {e['role']:<9} via {e['trace_id']}")
        line(f"  Instruments ({len(sysm['instruments'])})")
        for o in sysm["instruments"]:
            line(f"    - {o.get('text') or '(no text)':<16} {o.get('class_name') or '':<18} {o['id']}  on {o['trace_id']}")
        counts: dict[str, int] = {}
        for o in sysm["inline_objects"]:
            counts[o.get("class_name") or "?"] = counts.get(o.get("class_name") or "?", 0) + 1
        line(f"  Inline objects ({len(sysm['inline_objects'])})"
             + (": " + ", ".join(f"{c} x{k}" for c, k in sorted(counts.items())) if counts else ""))
        for o in sysm["inline_objects"]:
            line(f"    - {o['id']:<22} {o.get('class_name') or '':<22} {o.get('text') or ''}")
        line(f"  Boundaries ({len(sysm['boundaries'])})")
        for b in sysm["boundaries"]:
            line(f"    - {b['id']:<22} {b['type']:<20} {b['role']}")
        line(f"  Labels: {', '.join(lab['id'] for lab in sysm['labels'])}")
        line(f"  Flow arrows on these sections: {sysm['flow_arrow_count']}")
        line("")

    rule("=")
    line("CONFLICTS (one pipe section carrying more than one line number)")
    rule()
    for c in ts["conflicts"] or []:
        line(f"  - {c['trace_id']}: {' | '.join(c['line_numbers'])}")
    if not ts["conflicts"]:
        line("  none")
    line("")
    rule("=")
    line("PIPE SECTIONS IN NO SYSTEM (no line number attached)")
    rule()
    for i in range(0, len(ts["unassigned_traces"]), 3):
        line("  " + "  ".join(f"{t:<24}" for t in ts["unassigned_traces"][i:i + 3]).rstrip())
    if not ts["unassigned_traces"]:
        line("  none")
    line("")
    rule("=")
    line("Source: out/test_systems.json. Values are reproduced exactly as stored.")
    return "\n".join(x.rstrip() for x in out) + "\n"


@app.get("/api/garnet/jobs/{job_id}/report.txt", response_class=PlainTextResponse)
def api_garnet_report(job_id: str) -> PlainTextResponse:
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Job not found")
        text = _report_text(job_id)
    stem = _safe_stem(JOBS[job_id].get("source_name"))
    return PlainTextResponse(text, headers={"Content-Disposition": f'attachment; filename="{stem}_test_systems.txt"'})


# --- Multi-sheet: link test systems across sheets by line number ---------------------------
# Read-only and recomputed on request from each sheet's own test_systems.json, so a Gate 3 edit on
# any sheet flows through. Linking is EXACT text only: OCR near-misses are flagged, never merged.

MERGE_VERSION = 1


def _norm_line(key: str) -> str:
    """Comparison form for spotting near-duplicates only — never used to link."""
    return "".join(key.split()).casefold().strip("-")


import re  # noqa: E402

# A drawing reference as P&ID connectors write it, e.g. 25-0003 or 25-0002A.
DRAWING_REF = re.compile(r"\b(\d{2}-\d{4}[A-Z]?)\b")


def _sheet_ref(name: str) -> str | None:
    """`14780-8120-25-25-0003.jpg` -> `25-0003`: the form other sheets' connectors use."""
    m = re.search(r"(\d{2}-\d{4}[A-Z]?)$", Path(name).stem.upper())
    return m.group(1) if m else None


def _series_slip(ref: str, loaded: str) -> bool:
    """Same drawing number, different series prefix: `26-0003` vs `25-0003`. That is an OCR slip in
    the two-digit series. A different NUMBER (25-0002 vs 25-0003) is a different drawing, so it is
    never suggested — those are simply sheets that are not loaded."""
    return ref[3:] == loaded[3:] and ref[:2] != loaded[:2]


def _connector_links(sheets: list[dict], conns: dict[str, list[dict]]) -> list[dict]:
    """Pair off-page connectors between loaded sheets.

    A connector points at sheet B when its text contains B's drawing reference. For each pair of
    sheets (A, B), A's connectors pointing at B are paired with B's pointing at A:
      1. by a shared line number on the traces behind them  -> matched_by_line_number
      2. then, if exactly one is left on each side           -> matched_by_elimination, or
                                                                line_number_differs if both carry one
      3. anything left                                       -> ambiguous / no_counterpart
    A reference no loaded sheet has                         -> target_not_loaded (+ possible_target
                                                                when only its series prefix differs)
    Nothing is inferred into the data; this only reports.
    """
    ref_of = {s["job_id"]: s["ref"] for s in sheets if s.get("ref")}
    by_ref = {r: j for j, r in ref_of.items()}
    sheet_name = {s["job_id"]: s["sheet"] for s in sheets}

    def side(jid: str, c: dict) -> dict:
        return {"job_id": jid, "sheet": sheet_name[jid], "connector": c["id"], "type": c["type"], "text": c["text"],
                "line_numbers": c["line_numbers"], "trace_ids": [t["trace_id"] for t in c["traces"]]}

    links: list[dict] = []
    pointing: dict[tuple[str, str], list[dict]] = {}
    for jid, cs in conns.items():
        for c in cs:
            refs = DRAWING_REF.findall(c["text"].upper())
            if not refs:
                continue
            for ref in dict.fromkeys(refs):
                target = by_ref.get(ref)
                if target and target != jid:
                    pointing.setdefault((jid, target), []).append(c)
                elif not target and c["type"] == "page connection":
                    near = sorted(r for r in by_ref if by_ref[r] != jid and _series_slip(ref, r))
                    links.append({"status": "target_not_loaded", "ref": ref, "from": side(jid, c), "to": None,
                                  "possible_target": [sheet_name[by_ref[r]] for r in near]})

    done = set()
    for (a, b) in sorted(pointing):
        if (b, a) in done:
            continue
        done.add((a, b))
        rem_a = sorted(pointing.get((a, b), []), key=lambda c: c["id"])
        rem_b = sorted(pointing.get((b, a), []), key=lambda c: c["id"])
        for ca in list(rem_a):
            cb = next((cb for cb in rem_b if set(ca["line_numbers"]) & set(cb["line_numbers"])), None)
            if cb:
                links.append({"status": "matched_by_line_number", "from": side(a, ca), "to": side(b, cb),
                              "line_numbers": sorted(set(ca["line_numbers"]) & set(cb["line_numbers"]))})
                rem_a.remove(ca)
                rem_b.remove(cb)
        if len(rem_a) == 1 and len(rem_b) == 1:
            ca, cb = rem_a.pop(), rem_b.pop()
            differs = bool(ca["line_numbers"]) and bool(cb["line_numbers"])
            links.append({"status": "line_number_differs" if differs else "matched_by_elimination",
                          "from": side(a, ca), "to": side(b, cb)})
        for jid, other, rem, rem_other in ((a, b, rem_a, rem_b), (b, a, rem_b, rem_a)):
            for c in rem:
                links.append({"status": "ambiguous" if rem_other else "no_counterpart", "from": side(jid, c), "to": None,
                              "candidates": [x["id"] for x in rem_other], "toward": sheet_name[other]})
    order = {"matched_by_line_number": 0, "matched_by_elimination": 1, "line_number_differs": 2,
             "ambiguous": 3, "no_counterpart": 4, "target_not_loaded": 5}
    links.sort(key=lambda l: (order[l["status"]], l["from"]["sheet"], l["from"]["connector"]))
    return links


def _merged(job_ids: list[str]) -> dict:
    sheets, skipped = [], []
    per_line: dict[str, list[dict]] = {}
    conns: dict[str, list[dict]] = {}
    for jid in job_ids:
        job = JOBS.get(jid)
        if job is None:
            skipped.append({"job_id": jid, "reason": "job not found"})
            continue
        ts_path = JOBS_DIR / jid / "out" / "test_systems.json"
        if not ts_path.is_file():
            skipped.append({"job_id": jid, "sheet": job.get("source_name", ""), "reason": "Stage 6 has not run"})
            continue
        ts = json.loads(ts_path.read_text())
        name = job.get("source_name") or jid[:8]
        sheets.append({"job_id": jid, "sheet": name, "status": job["status"], "gate": job.get("gate"),
                       "gate3_confirmed": bool(ts.get("gate3_confirmed")), "ref": _sheet_ref(name)})
        conns[jid] = ts.get("connectors", [])
        for sysm in ts["systems"]:
            # A textless label names nothing that could match another sheet: keep it sheet-local.
            key = f"(no text) {name}:{sysm['labels'][0]['id']}" if sysm["no_text"] else sysm["line_number"]
            counts: dict[str, int] = {}
            for o in sysm["inline_objects"]:
                c = o.get("class_name") or "?"
                counts[c] = counts.get(c, 0) + 1
            bnd: dict[str, int] = {}
            for b in sysm["boundaries"]:
                bnd[b["type"]] = bnd.get(b["type"], 0) + 1
            per_line.setdefault(key, []).append({
                "job_id": jid, "sheet": name,
                "trace_ids": [t["trace_id"] for t in sysm["traces"]],
                "equipment": [e.get("tag") or e["id"] for e in sysm["equipment"]],
                "instruments": [o.get("text") or f"(no text) {o['id']}" for o in sysm["instruments"]],
                "inline_counts": dict(sorted(counts.items())),
                "boundaries": dict(sorted(bnd.items())),
                "conflict": sysm["has_conflict"], "no_text": sysm["no_text"],
            })

    links = _connector_links(sheets, conns)
    # Per line number: the connector links that touch it, and suggestions where the far side of a
    # paired connector carries no line number ("probably continues this line").
    link_by_line: dict[str, list[dict]] = {}
    suggestions: dict[str, list[dict]] = {}
    for lk in links:
        if lk["to"] is None:
            continue
        for near, far in ((lk["from"], lk["to"]), (lk["to"], lk["from"])):
            for ln in near["line_numbers"]:
                link_by_line.setdefault(ln, []).append(lk)
                if not far["line_numbers"] and far["trace_ids"]:
                    suggestions.setdefault(ln, []).append({"sheet": far["sheet"], "job_id": far["job_id"],
                                                           "connector": far["connector"], "trace_ids": far["trace_ids"],
                                                           "via": lk["status"]})

    groups: dict[str, set[str]] = {}
    for key in per_line:
        groups.setdefault(_norm_line(key), set()).add(key)

    lines = []
    for key in sorted(per_line):
        entries = sorted(per_line[key], key=lambda e: (e["sheet"], e["job_id"]))
        offpage = any("page connection" in e["boundaries"] for e in entries)
        eq = sorted({t for e in entries for t in e["equipment"]})
        lines.append({
            "line_number": key,
            "no_text": entries[0]["no_text"],
            "sheets": entries,
            "sheet_count": len({e["job_id"] for e in entries}),
            "totals": {"pipe_sections": sum(len(e["trace_ids"]) for e in entries),
                       "equipment": eq,
                       "instruments": sum(len(e["instruments"]) for e in entries),
                       "inline_objects": sum(sum(e["inline_counts"].values()) for e in entries)},
            "offpage": offpage,
            # Leaves the sheet through a connector, yet no other loaded sheet carries it.
            "continuation_missing": offpage and len({e["job_id"] for e in entries}) == 1 and not entries[0]["no_text"],
            "conflict": any(e["conflict"] for e in entries),
            "similar_to": [] if entries[0]["no_text"] else sorted(groups[_norm_line(key)] - {key}),
            "connector_links": [{"status": lk["status"], "from": f"{lk['from']['sheet']}:{lk['from']['connector']}",
                                 "to": f"{lk['to']['sheet']}:{lk['to']['connector']}"}
                                for lk in {id(l): l for l in link_by_line.get(key, [])}.values()],
            # Unlabelled sections on another sheet joined to this line through a paired connector.
            "suggested_continuations": suggestions.get(key, []),
        })
        # Any paired connector means the line does continue on a loaded sheet — even when the far
        # side's line number differs; that is its own flag (connector_links status), not "missing".
        if link_by_line.get(key):
            lines[-1]["continuation_missing"] = False
    return {
        "version": MERGE_VERSION,
        "generated_at": _now(),
        "rule": "exact line-number text match across sheets; near-matches are flagged in similar_to, never merged",
        "sheets": sheets,
        "skipped_sheets": skipped,
        "lines": lines,
        "connector_links": links,
        "summary": {"lines": len(lines),
                    "connector_links": {st: sum(1 for l in links if l["status"] == st) for st in (
                        "matched_by_line_number", "matched_by_elimination", "line_number_differs",
                        "ambiguous", "no_counterpart", "target_not_loaded")},
                    "multi_sheet": sum(1 for x in lines if x["sheet_count"] > 1),
                    "continuation_missing": sum(1 for x in lines if x["continuation_missing"]),
                    "similar": sum(1 for x in lines if x["similar_to"]),
                    "conflict": sum(1 for x in lines if x["conflict"])},
    }


def _merged_text(m: dict) -> str:
    W = 78
    out: list[str] = []
    line = out.append
    line("=" * W)
    line("GARNET MULTI-SHEET TEST SYSTEMS REPORT")
    line("=" * W)
    line(f"  Generated         {m['generated_at']}")
    line(f"  Linking rule      {m['rule']}")
    line("")
    line("  Sheets")
    for s in m["sheets"]:
        state = "Gate 3 confirmed" if s["gate3_confirmed"] else f"NOT final ({s['status']}{' Gate ' + str(s['gate']) if s['gate'] else ''})"
        line(f"    - {s['sheet']:<32} {state}")
    for s in m["skipped_sheets"]:
        line(f"    - {s.get('sheet') or s['job_id']:<32} SKIPPED: {s['reason']}")
    sm = m["summary"]
    line("")
    line(f"  Line numbers      {sm['lines']}  (on more than one sheet: {sm['multi_sheet']})")
    line(f"  Off-sheet, continuation not in loaded sheets: {sm['continuation_missing']}")
    line(f"  Possible duplicates (similar text): {sm['similar']}")
    line(f"  With a conflict on some sheet:      {sm['conflict']}")
    line("")
    for n, x in enumerate(m["lines"], start=1):
        flags = [f for f, on in (("MULTI-SHEET", x["sheet_count"] > 1), ("CONTINUATION NOT FOUND", x["continuation_missing"]),
                                 ("CONFLICT", x["conflict"]), ("NO TEXT", x["no_text"])) if on]
        line("=" * W)
        line(f"[{n}] {x['line_number']}" + (f"   <{', '.join(flags)}>" if flags else ""))
        line("-" * W)
        if x["similar_to"]:
            line(f"  ** similar text, NOT linked: {' | '.join(x['similar_to'])}")
        for lk in x["connector_links"]:
            line(f"  via connector: {lk['from']} <-> {lk['to']}  ({lk['status']})")
        for sg in x["suggested_continuations"]:
            line(f"  ?? probably continues on {sg['sheet']}: {', '.join(sg['trace_ids'])} (via {sg['connector']}, no line number there)")
        t = x["totals"]
        line(f"  Sheets {x['sheet_count']} | sections {t['pipe_sections']} | instruments {t['instruments']} | inline {t['inline_objects']}")
        line(f"  Equipment: {', '.join(t['equipment']) or '-'}")
        for e in x["sheets"]:
            line(f"  > {e['sheet']}" + ("   (conflict on this sheet)" if e["conflict"] else ""))
            line(f"      sections    {', '.join(e['trace_ids']) or '-'}")
            line(f"      equipment   {', '.join(e['equipment']) or '-'}")
            line(f"      instruments {', '.join(e['instruments']) or '-'}")
            line(f"      inline      {', '.join(f'{c} x{k}' for c, k in e['inline_counts'].items()) or '-'}")
            if e["boundaries"]:
                line(f"      boundaries  {', '.join(f'{c} x{k}' for c, k in e['boundaries'].items())}")
        line("")
    line("=" * W)
    line("OFF-PAGE CONNECTOR LINKS")
    line("-" * W)
    for lk in m["connector_links"]:
        f = lk["from"]
        a = f"{f['sheet']}:{f['connector']} \"{f['text']}\" [{', '.join(f['line_numbers']) or 'no line no.'}]"
        if lk["to"]:
            t = lk["to"]
            b = f"{t['sheet']}:{t['connector']} \"{t['text']}\" [{', '.join(t['line_numbers']) or 'no line no.'}]"
            line(f"  {lk['status'].upper():<24} {a}")
            line(f"  {'':<24} <-> {b}")
        else:
            extra = (f" (ref {lk['ref']}" + (f", possibly {', '.join(lk['possible_target'])}" if lk.get("possible_target") else "") + ")") \
                if lk["status"] == "target_not_loaded" else f" (toward {lk.get('toward')})"
            line(f"  {lk['status'].upper():<24} {a}{extra}")
    if not m["connector_links"]:
        line("  none")
    line("")
    line("=" * W)
    line("Source: each sheet's out/test_systems.json. Values are reproduced exactly as stored.")
    return "\n".join(x.rstrip() for x in out) + "\n"


def _merge_jobs(jobs: str) -> list[str]:
    ids = [j for j in (jobs or "").split(",") if j]
    if not ids:
        raise HTTPException(status_code=400, detail="jobs: give one or more job ids, comma-separated")
    if len(ids) > 200:
        raise HTTPException(status_code=400, detail="jobs: at most 200 sheets")
    return list(dict.fromkeys(ids))  # de-duplicate, keep order


@app.get("/api/garnet/merge/test_systems.json")
def api_garnet_merge_json(jobs: str = "") -> dict:
    with JOBS_LOCK:
        return _merged(_merge_jobs(jobs))


@app.get("/api/garnet/merge/test_systems.txt", response_class=PlainTextResponse)
def api_garnet_merge_txt(jobs: str = "") -> PlainTextResponse:
    with JOBS_LOCK:
        text = _merged_text(_merged(_merge_jobs(jobs)))
    return PlainTextResponse(text, headers={"Content-Disposition": 'attachment; filename="garnet_multi_sheet_test_systems.txt"'})
