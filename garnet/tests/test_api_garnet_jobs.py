"""garnet/ job route: all three JSONs uploaded -> ML steps skipped, chain runs to Stage 6."""
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from garnet.api import app

INPUT = Path(__file__).parent / "input"
STEM = "14780-8120-25-25-0002A"


class TestGarnetJobs(unittest.TestCase):
    def test_uploaded_json_skips_ml_and_completes(self):
        client = TestClient(app)
        files = {
            "file_input": ("s.jpg", (INPUT / f"{STEM}.jpg").read_bytes(), "image/jpeg"),
            "objects_json": ("o.json", (INPUT / f"{STEM}_objects.json").read_bytes()),
            "equipment_json": ("e.json", (INPUT / f"{STEM}_equipment_bboxes.json").read_bytes()),
            "line_numbers_json": ("l.json", (INPUT / f"{STEM}_line_numbers.json").read_bytes()),
        }
        r = client.post("/api/garnet/jobs", files=files)
        self.assertEqual(r.status_code, 200, r.text)
        job = r.json()
        status = {s["name"]: s["status"] for s in job["steps"]}
        self.assertEqual(status["detect_objects"], "skipped")
        self.assertEqual(status["detect_equipment"], "skipped")

        deadline = time.time() + 600
        while job["status"] in ("queued", "running") and time.time() < deadline:
            time.sleep(2)
            job = client.get(f"/api/garnet/jobs/{job['id']}").json()
        self.assertEqual(job["status"], "completed", job)
        names = {a["name"] for a in job["artifacts"]}
        self.assertIn("out/stage6_trace_associations.json", names)
        # Artifact URLs are served by the /runs static mount.
        overlay = next(a for a in job["artifacts"] if a["name"] == "out/stage5b_trace_overlay.png")
        self.assertEqual(client.get(overlay["url"]).status_code, 200)

        # OCR filled instrument text in the staged copy; the arriving file is kept untouched.
        names = {a["name"] for a in job["artifacts"]}
        self.assertIn("input/sheet_objects.source.json", names)
        status = {s["name"]: s["status"] for s in job["steps"]}
        self.assertEqual(status["ocr_instruments"], "completed")

        # Edits: an overlay on Stage 6, versioned, never a rewrite of the Stage 6 file.
        jid = job["id"]
        stage6_before = client.get(f"/runs/jobs/{jid}/out/stage6_trace_associations.json").json()
        tag = stage6_before["associations"]["instrument_tags"]["accepted"][0]
        r = client.put(f"/api/garnet/jobs/{jid}/edits", json={"edits": {tag["id"]: "TI 9999"}})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["edits"], {tag["id"]: "TI 9999"})
        reviewed = client.get(f"/runs/jobs/{jid}/out/reviewed_associations.json").json()
        item = next(i for i in reviewed["associations"]["instrument_tags"]["accepted"] if i["id"] == tag["id"])
        self.assertEqual((item["text"], item["text_original"], item["text_source"]), ("TI 9999", tag["text"], "human_edit"))
        self.assertEqual(client.get(f"/runs/jobs/{jid}/out/stage6_trace_associations.json").json(), stage6_before)
        self.assertEqual(client.put(f"/api/garnet/jobs/{jid}/edits", json={"edits": {"nope": "x"}}).status_code, 400)
        client.put(f"/api/garnet/jobs/{jid}/edits", json={"edits": {}})
        history = client.get(f"/runs/jobs/{jid}/edits_history.jsonl").text.strip().splitlines()
        self.assertEqual(len(history), 2)

    def test_bad_json_rejected_before_staging(self):
        client = TestClient(app)
        files = {
            "file_input": ("s.jpg", (INPUT / f"{STEM}.jpg").read_bytes(), "image/jpeg"),
            "objects_json": ("o.json", b"{not json"),
        }
        self.assertEqual(client.post("/api/garnet/jobs", files=files).status_code, 400)


if __name__ == "__main__":
    unittest.main()
