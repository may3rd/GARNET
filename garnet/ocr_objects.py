#!/usr/bin/env python
"""OCR for the text-bearing detected objects — instrument tags, line numbers, connections.

Reads the text INSIDE an already-detected box, one crop at a time. This is not sheet-wide text
discovery: for that, `garnet.stage5b.stage2_ocr` runs macOS Vision over the whole raster to find
text the detector missed. Here the boxes are known and only their contents are unknown.

The engine is EasyOCR with the preprocessing `backend/api.py` calibrated for these crops
(invert, 2x2 dilate, invert back — which thickens thin drafting strokes), and the same
allowlist. Measured on 14780-8120-25-25-0002A instrument tags, EasyOCR with those settings read
`PI 0203` and `TE 0201` where macOS Vision returned `P` and digits only, so accuracy decides the
engine here even though Vision is ~3x faster on small crops.

    from garnet.ocr_objects import read_object_text, has_text
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("garnet.api")

# Classes whose box contains text worth reading, and those drawn rotated when tall.
# Mirrored from backend/garnet/Settings.py; the weight spells them with spaces, but compare
# underscore-normalized so an `instrument_tag` from another weight is not silently skipped.
SYMBOL_WITH_TEXT = {
    "page connection",
    "instrument dcs",
    "instrument logic",
    "instrument tag",
    "line number",
    "utility connection",
}
VERTICAL_TEXT = {"page connection", "line number"}
ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"½'

_READER_SLOT: dict = {"reader": None}
_READER_LOCK = threading.Lock()


def _normalize(class_name: str) -> str:
    return class_name.replace("_", " ").strip().lower()


def has_text(class_name: str) -> bool:
    """True when this detection class is one whose box contains readable text."""
    return _normalize(class_name) in SYMBOL_WITH_TEXT


def _get_reader():
    """The EasyOCR reader, built once. Caller MUST hold `_READER_LOCK`.

    Reader construction downloads/loads the detection + recognition models and costs seconds;
    `easyocr.Reader` is not documented thread-safe, so it is serialised the same way the YOLO
    model is in `detect_objects.load_model`.
    """
    if _READER_SLOT["reader"] is None:
        import easyocr

        logger.info("ocr: initialising EasyOCR reader")
        _READER_SLOT["reader"] = easyocr.Reader(["en"])
    return _READER_SLOT["reader"]


def read_object_text(image_bgr, rows: list[dict]) -> int:
    """OCR every text-bearing row in place, overwriting its `Text`. Returns rows read.

    `rows` are frontend rows (`Object`, `Left`, `Top`, `Width`, `Height`, `Text`). A row whose
    crop yields nothing keeps its default label rather than being blanked — an empty tag would
    look like a read tag with no value. A row that fails to OCR is logged and skipped, but a
    reader that cannot be built raises: the whole request asked for OCR, so returning boxes
    without it silently is the failure being avoided.
    """
    import cv2

    targets = [row for row in rows if has_text(row["Object"])]
    if not targets:
        logger.info("ocr: no text-bearing objects among %d rows", len(rows))
        return 0

    height, width = image_bgr.shape[:2]
    read = 0
    with _READER_LOCK:
        reader = _get_reader()
        logger.info("ocr: reading %d of %d objects", len(targets), len(rows))
        for row in targets:
            # Clamp: a box may extend past the raster edge, and a numpy slice would silently
            # return a smaller or empty crop instead of failing.
            x0, y0 = max(0, row["Left"]), max(0, row["Top"])
            x1 = min(width, row["Left"] + row["Width"])
            y1 = min(height, row["Top"] + row["Height"])
            if x1 <= x0 or y1 <= y0:
                logger.warning("ocr: object %s has no on-image area", row["Index"])
                continue
            crop = image_bgr[y0:y1, x0:x1]

            if _normalize(row["Object"]) in VERTICAL_TEXT and crop.shape[0] > crop.shape[1]:
                # Tall box for a class drawn rotated: CLOCKWISE, matching the backend's
                # rotate_image(crop, 270) — getRotationMatrix2D takes a counter-clockwise
                # angle, so 270 CCW is 90 CW. The other direction reads the line number
                # mirrored and reversed ('3"-CUL-25-002010-B2A2-' comes back
                # 'IN-ZV28-OLOZOO-SZ-I3-'), which still looks like a plausible tag.
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

            # Thicken thin drafting strokes: invert, dilate, invert back.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            crop = cv2.bitwise_not(cv2.dilate(cv2.bitwise_not(crop), kernel, iterations=1))

            try:
                found = reader.readtext(
                    crop,
                    decoder="wordbeamsearch",
                    batch_size=4,
                    paragraph=True,
                    detail=0,
                    mag_ratio=1.0,
                    text_threshold=0.7,
                    low_text=0.2,
                    allowlist=ALLOWLIST,
                )
            except Exception as exc:
                logger.warning("ocr: object %s failed: %s", row["Index"], exc)
                continue

            text = " ".join(str(part) for part in found).strip()
            if text:
                row["Text"] = text
                read += 1

    logger.info("ocr: read %d of %d text-bearing objects", read, len(targets))
    return read
