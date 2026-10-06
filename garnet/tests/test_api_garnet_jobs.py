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
        r = client.post("/api/garnet/jobs", files=files, data={"pause_at_gates": "false"})
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

    def test_gates_park_apply_and_resume(self):
        client = TestClient(app)
        files = {
            "file_input": ("s.jpg", (INPUT / f"{STEM}.jpg").read_bytes(), "image/jpeg"),
            "objects_json": ("o.json", (INPUT / f"{STEM}_objects.json").read_bytes()),
            "equipment_json": ("e.json", (INPUT / f"{STEM}_equipment_bboxes.json").read_bytes()),
            "line_numbers_json": ("l.json", (INPUT / f"{STEM}_line_numbers.json").read_bytes()),
        }
        # OCR off: this test is about gate mechanics, and it keeps the run fast.
        job = client.post("/api/garnet/jobs", files=files, data={"ocr_instruments": "false"}).json()
        jid = job["id"]

        def wait():
            j = client.get(f"/api/garnet/jobs/{jid}").json()
            deadline = time.time() + 600
            while j["status"] in ("queued", "running") and time.time() < deadline:
                time.sleep(1)
                j = client.get(f"/api/garnet/jobs/{jid}").json()
            return j

        def get(rel):
            return client.get(f"/runs/jobs/{jid}/{rel}").json()

        job = wait()
        self.assertEqual((job["status"], job["gate"]), ("awaiting_gate", 1))
        self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/gates/2", json={}).status_code, 409)
        objs = get("input/sheet_objects.json")["objects"]
        drop = next(o["id"] for o in objs if o["class_name"] == "gate valve")
        tag = next(o["id"] for o in objs if o["class_name"] == "instrument tag")
        moved = next(o for o in objs if o["class_name"] == "check valve")
        new_box = {k: moved["bbox"][k] + 2 for k in ("x_min", "y_min", "x_max", "y_max")}
        added = {"class_name": "instrument tag", "text": "PI 0999",
                 "bbox": {"x_min": 10, "y_min": 10, "x_max": 60, "y_max": 60}}
        off = {"class_name": "gate valve", "bbox": {"x_min": 10, "y_min": 10, "x_max": 10, "y_max": 99999}}
        self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/gates/1", json={"added": [off]}).status_code, 400)
        r = client.post(f"/api/garnet/jobs/{jid}/gates/1", json={
            "rejected": [drop], "edits": {tag: "TI 0001"}, "added": [added], "moved": {moved["id"]: new_box},
            "reclassed": {moved["id"]: "globe valve"}})
        self.assertEqual(r.status_code, 200, r.text)

        job = wait()
        self.assertEqual((job["status"], job["gate"]), ("awaiting_gate", 2))
        staged = {o["id"]: o for o in get("input/sheet_objects.json")["objects"]}
        self.assertNotIn(drop, staged)
        self.assertEqual(staged[tag]["text"], "TI 0001")
        self.assertEqual((staged[moved["id"]]["bbox"], staged[moved["id"]]["bbox_original"]), (new_box, moved["bbox"]))
        self.assertEqual((staged[moved["id"]]["class_name"], staged[moved["id"]]["class_name_original"]),
                         ("globe valve", "check valve"))
        human = staged["human_000001"]
        self.assertEqual((human["class_name"], human["text"], human["source"]), ("instrument tag", "PI 0999", "human"))
        self.assertIn("human_000001", {o["id"] for o in get("out/stage4_objects.json")["objects"]})
        self.assertIn(drop, {o["id"] for o in get("input/sheet_objects.pre_gate1.json")["objects"]})
        traces = get("out/stage5b_trace_results.json")
        dropped_trace = next(iter(traces))
        self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/gates/2", json={"rejected": ["nope"]}).status_code, 400)
        self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/gates/2", json={"rejected": [dropped_trace]}).status_code, 200)

        job = wait()
        self.assertEqual((job["status"], job["gate"]), ("awaiting_gate", 3))
        stage6 = get("out/stage6_trace_associations.json")
        self.assertNotIn(dropped_trace, {e["trace_id"] for e in stage6["trace_edges"]})
        self.assertIn(dropped_trace, get("out/stage5b_trace_results.json"))  # original untouched
        ln = stage6["associations"]["line_numbers"]["accepted"][0]["id"]
        moved_ln = stage6["associations"]["line_numbers"]["accepted"][1]
        target = next(e["trace_id"] for e in stage6["trace_edges"] if e["trace_id"] != moved_ln["trace_id"])
        for bad in ({"reassigned": {moved_ln["id"]: dropped_trace}},             # rejected at Gate 2
                    {"reassigned": {moved_ln["id"]: target}, "decisions": {moved_ln["id"]: "rejected"}}):
            self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/gates/3", json=bad).status_code, 400)
        r = client.post(f"/api/garnet/jobs/{jid}/gates/3", json={
            "decisions": {ln: "rejected"}, "edits": {ln: "X-1"}, "reassigned": {moved_ln["id"]: target}})
        self.assertEqual(r.status_code, 200, r.text)

        job = wait()
        self.assertEqual(job["status"], "completed")
        reviewed = get("out/reviewed_associations.json")["associations"]["line_numbers"]
        item = next(i for i in reviewed["rejected"] if i["id"] == ln)
        self.assertEqual((item["review_source"], item["system_status"], item["text"]), ("human", "accepted", "X-1"))
        re_item = next(i for i in reviewed["accepted"] if i["id"] == moved_ln["id"])
        self.assertEqual((re_item["trace_id"], re_item["trace_assignment"], re_item["system_assignment"]["trace_id"]),
                         (target, "human", moved_ln["trace_id"]))
        self.assertNotIn("projected_xy", re_item)
        unresolved = get("out/reviewed_associations.json")["unresolved"]["traces_without_line_number"]
        self.assertNotIn(target, unresolved)
        self.assertEqual(set(get("gates.json")), {"1", "2", "3"})

    def test_bad_json_rejected_before_staging(self):
        client = TestClient(app)
        files = {
            "file_input": ("s.jpg", (INPUT / f"{STEM}.jpg").read_bytes(), "image/jpeg"),
            "objects_json": ("o.json", b"{not json"),
        }
        self.assertEqual(client.post("/api/garnet/jobs", files=files).status_code, 400)


if __name__ == "__main__":
    unittest.main()
