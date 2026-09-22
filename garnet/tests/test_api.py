"""Checks for garnet/api.py — the detect path wired for the frontend.

`detect()` is stubbed throughout: this asserts the API contract (row shape, numbering,
validation, CRUD), not YOLO. Nothing here loads a weight.

Run: .venv/bin/python -m unittest discover -s garnet/tests -p "test_api.py" -v
"""

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import numpy

from garnet import api
from garnet import ocr_objects


def _stub_detection(objects):
    return {"image_width": 1000, "image_height": 800, "objects": objects}


def _obj(class_id, class_name, x, y, conf=0.9):
    return {
        "class_name": class_name,
        "class_id": class_id,
        "confidence": conf,
        "bbox": {"x_min": x, "y_min": y, "x_max": x + 10, "y_max": y + 20},
    }


class DetectTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self._real_detect = api.detect
        api.RESULTS.clear()
        # A real weight must exist for the path check; use whatever the repo ships.
        weights = api._weight_files()
        self.weight = weights[0] if weights else None

    def tearDown(self):
        api.detect = self._real_detect
        ocr_objects._READER_SLOT["reader"] = None
        for path in api.DETECT_DIR.glob("prediction_*"):
            path.unlink(missing_ok=True)
        api.RESULTS.clear()

    def _post(self, **form):
        payload = {"selected_model": "ultralytics", "conf_th": "0.5"}
        payload.update({k: str(v) for k, v in form.items()})
        return self.client.post(
            "/api/detect",
            files={"file_input": ("sheet.png", b"fake-raster", "image/png")},
            data=payload,
        )

    def _stub(self, objects):
        # A real array: OCR crops it, so None would only work while OCR is off.
        canvas = numpy.zeros((800, 1000, 3), dtype=numpy.uint8)
        api.detect = lambda image_path, **kw: (_stub_detection(objects), canvas)

    def test_rows_number_per_class_and_sort(self):
        """ObjectID restarts per class and rows come back sorted by (CategoryID, ObjectID)."""
        self._stub([_obj(3, "valve", 0, 0), _obj(1, "pump", 5, 5), _obj(3, "valve", 9, 9)])
        resp = self._post()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["count"], 3)
        self.assertEqual(
            [(o["Index"], o["CategoryID"], o["ObjectID"], o["Text"]) for o in body["objects"]],
            [(1, 1, 1, "pump - no. 1"), (2, 3, 1, "valve - no. 1"), (3, 3, 2, "valve - no. 2")],
        )
        first = body["objects"][1]
        self.assertEqual((first["Left"], first["Top"], first["Width"], first["Height"]), (0, 0, 10, 20))
        self.assertTrue(body["image_url"].startswith("/runs/detect/"))
        self.assertTrue((api.DETECT_DIR / Path(body["image_url"]).name).is_file())

    def test_ocr_reads_only_text_bearing_classes(self):
        """Tag/connection boxes get their Text replaced; a valve is never cropped at all."""
        self._stub([_obj(0, "instrument tag", 0, 0), _obj(5, "gate valve", 40, 40)])
        cropped = []

        class FakeReader:
            def readtext(self, crop, **kw):
                cropped.append(crop.shape)
                return ["PI 0203"]

        ocr_objects._READER_SLOT["reader"] = FakeReader()
        resp = self._post(text_OCR="true")
        self.assertEqual(resp.status_code, 200, resp.text)
        by_class = {o["Object"]: o["Text"] for o in resp.json()["objects"]}
        self.assertEqual(by_class["instrument tag"], "PI 0203")
        self.assertEqual(by_class["gate valve"], "gate valve - no. 1")
        self.assertEqual(len(cropped), 1)

    def test_empty_ocr_keeps_the_default_label(self):
        """A blank read must not blank the row — that reads as a tag with no value."""
        self._stub([_obj(0, "instrument tag", 0, 0)])

        class FakeReader:
            def readtext(self, crop, **kw):
                return []

        ocr_objects._READER_SLOT["reader"] = FakeReader()
        resp = self._post(text_OCR="true")
        self.assertEqual(resp.json()["objects"][0]["Text"], "instrument tag - no. 1")

    def test_one_failing_crop_does_not_fail_the_request(self):
        self._stub([_obj(0, "instrument tag", 0, 0), _obj(0, "instrument tag", 30, 30)])
        calls = []

        class FlakyReader:
            def readtext(self, crop, **kw):
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("boom")
                return ["FT 0206"]

        ocr_objects._READER_SLOT["reader"] = FlakyReader()
        resp = self._post(text_OCR="true")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(
            [o["Text"] for o in resp.json()["objects"]],
            ["instrument tag - no. 1", "FT 0206"],
        )

    def test_unbuildable_reader_is_a_500_not_silent_boxes(self):
        self._stub([_obj(0, "instrument tag", 0, 0)])
        ocr_objects._READER_SLOT["reader"] = None
        real = ocr_objects._get_reader
        ocr_objects._get_reader = lambda: (_ for _ in ()).throw(RuntimeError("no model"))
        try:
            resp = self._post(text_OCR="true")
        finally:
            ocr_objects._get_reader = real
        self.assertEqual(resp.status_code, 500)
        self.assertIn("OCR failed", resp.json()["detail"])

    def test_tall_vertical_class_is_rotated_clockwise(self):
        """Direction, not just shape: anti-clockwise reads a line number reversed.

        A mark at the crop's top-left must land top-RIGHT after a clockwise rotation. Asserting
        the rotated shape alone passes either way, which is how the wrong direction ships.
        """
        seen = []

        class FakeReader:
            def readtext(self, crop, **kw):
                seen.append(crop)
                return ["A"]

        ocr_objects._READER_SLOT["reader"] = FakeReader()
        tall = _obj(0, "page connection", 0, 0)
        tall["bbox"] = {"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 60}
        self._stub([tall])
        # White mark in the crop's top-left corner, on the otherwise black canvas.
        api_detect = api.detect
        canvas = numpy.zeros((800, 1000, 3), dtype=numpy.uint8)
        canvas[0:5, 0:5] = 255
        api.detect = lambda image_path, **kw: (_stub_detection([tall]), canvas)
        try:
            self.assertEqual(self._post(text_OCR="true").status_code, 200)
        finally:
            api.detect = api_detect
        crop = seen[0]
        self.assertEqual(crop.shape[:2], (20, 60))          # 60x20 rotated to 20x60
        self.assertTrue(crop[0:5, -5:].any())               # mark now top-right
        self.assertFalse(crop[-5:, 0:5].any())              # not bottom-left (anti-clockwise)

    def test_wide_vertical_class_is_not_rotated(self):
        """A wide `page connection` is drawn horizontally; rotating it would read worse."""
        shapes = []

        class FakeReader:
            def readtext(self, crop, **kw):
                shapes.append(crop.shape[:2])
                return ["A"]

        ocr_objects._READER_SLOT["reader"] = FakeReader()
        wide = _obj(0, "page connection", 0, 0)
        wide["bbox"] = {"x_min": 100, "y_min": 100, "x_max": 160, "y_max": 120}
        self._stub([wide])
        self.assertEqual(self._post(text_OCR="true").status_code, 200)
        self.assertEqual(shapes, [(20, 60)])

    def test_weight_outside_weights_dir_is_refused(self):
        """weight_file is client-supplied; it must not name an arbitrary file for torch."""
        self._stub([])
        resp = self._post(weight_file="../../etc/passwd")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("yolo_weights", resp.json()["detail"])

    def test_bad_params_and_types_are_refused(self):
        self._stub([])
        self.assertEqual(self._post(conf_th="1.5").status_code, 400)
        self.assertEqual(self._post(overlap_ratio="1.0").status_code, 400)
        self.assertEqual(self._post(postprocess_type="LSNMS").status_code, 400)
        self.assertEqual(self._post(selected_model="gemini").status_code, 400)
        resp = self.client.post(
            "/api/detect",
            files={"file_input": ("sheet.pdf", b"x", "application/pdf")},
            data={"selected_model": "ultralytics"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_undecodable_image_does_not_leave_a_file(self):
        def boom(image_path, **kw):
            raise FileNotFoundError("Cannot read image")

        api.detect = boom
        resp = self._post()
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(list(api.DETECT_DIR.glob("prediction_*")), [])


class ResultCrudTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self._real_detect = api.detect
        api.RESULTS.clear()
        api.detect = lambda image_path, **kw: (
            _stub_detection([_obj(2, "valve", 0, 0)]),
            numpy.zeros((800, 1000, 3), dtype=numpy.uint8),
        )
        self.result_id = self.client.post(
            "/api/detect",
            files={"file_input": ("sheet.png", b"fake", "image/png")},
            data={"selected_model": "ultralytics"},
        ).json()["id"]

    def tearDown(self):
        api.detect = self._real_detect
        for path in api.DETECT_DIR.glob("prediction_*"):
            path.unlink(missing_ok=True)
        api.RESULTS.clear()

    def test_create_patch_delete_round_trip(self):
        created = self.client.post(
            f"/api/results/{self.result_id}/objects",
            json={"Object": "pump", "Left": 1.7, "Top": 2.2, "Width": 5.1, "Height": 6.9, "Text": ""},
        ).json()
        # Left/Top floor, Width/Height ceil — a box must never shrink below what was drawn.
        self.assertEqual((created["Left"], created["Top"], created["Width"], created["Height"]), (1, 2, 6, 7))
        self.assertEqual(created["Text"], "pump")
        self.assertEqual(self.client.get(f"/api/results/{self.result_id}").json()["count"], 2)

        patched = self.client.patch(
            f"/api/results/{self.result_id}/objects/{created['Index']}",
            json={"Text": "P-101", "ReviewStatus": "accepted"},
        ).json()
        self.assertEqual((patched["Text"], patched["ReviewStatus"]), ("P-101", "accepted"))

        self.assertEqual(
            self.client.delete(f"/api/results/{self.result_id}/objects/{created['Index']}").status_code, 200
        )
        self.assertEqual(self.client.get(f"/api/results/{self.result_id}").json()["count"], 1)

    def test_missing_result_and_object_are_404(self):
        self.assertEqual(self.client.get("/api/results/nope").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/results/{self.result_id}/objects/999").status_code, 404)
        self.assertEqual(
            self.client.patch(f"/api/results/{self.result_id}/objects/1", json={}).status_code, 400
        )


class DiscoveryTests(unittest.TestCase):
    def test_health_and_listings(self):
        client = TestClient(api.app)
        health = client.get("/api/health").json()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["models_available"], len(api._weight_files()))
        self.assertEqual(client.get("/api/models").json(), ["ultralytics"])
        self.assertEqual(client.get("/api/config-files").json(), [])
        for name in client.get("/api/weight-files").json():
            self.assertTrue((api.REPO_ROOT / name).is_file())


if __name__ == "__main__":
    unittest.main()
