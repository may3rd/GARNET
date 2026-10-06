"""garnet/ job route: all three JSONs uploaded -> ML steps skipped, chain runs to Stage 6."""
import json
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

        # Test systems: every accepted line number lands in a system; traces with two different
        # read line numbers are flagged, not silently given to one.
        ts = client.get(f"/runs/jobs/{job['id']}/out/test_systems.json").json()
        reviewed = client.get(f"/runs/jobs/{job['id']}/out/reviewed_associations.json").json()
        accepted = reviewed["associations"]["line_numbers"]["accepted"]
        in_systems = {lab["id"] for sysm in ts["systems"] for lab in sysm["labels"]}
        self.assertEqual(in_systems, {ln["id"] for ln in accepted})
        conflict_traces = {c["trace_id"] for c in ts["conflicts"]}
        self.assertIn("obj_000197", conflict_traces)  # 3"-CUL-25-002008 and 2"-NAS-25-003004 on one trace
        for c in ts["conflicts"]:
            self.assertGreaterEqual(len(c["line_numbers"]), 2)
            self.assertFalse(any(k.startswith("(no text)") for k in c["line_numbers"]))
        assigned = {t["trace_id"] for sysm in ts["systems"] for t in sysm["traces"]}
        self.assertEqual(set(ts["unassigned_traces"]) & assigned, set())
        self.assertEqual(set(ts["unassigned_traces"]) | assigned, {e["trace_id"] for e in reviewed["trace_edges"]})
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

    def test_reopen_gate1_restores_inputs_after_restart(self):
        from garnet import api
        client = TestClient(app)
        files = {
            "file_input": ("s.jpg", (INPUT / f"{STEM}.jpg").read_bytes(), "image/jpeg"),
            "objects_json": ("o.json", (INPUT / f"{STEM}_objects.json").read_bytes()),
            "equipment_json": ("e.json", (INPUT / f"{STEM}_equipment_bboxes.json").read_bytes()),
            "line_numbers_json": ("l.json", (INPUT / f"{STEM}_line_numbers.json").read_bytes()),
        }
        jid = client.post("/api/garnet/jobs", files=files, data={"ocr_instruments": "false"}).json()["id"]

        def wait():
            j = client.get(f"/api/garnet/jobs/{jid}").json()
            deadline = time.time() + 600
            while j["status"] in ("queued", "running") and time.time() < deadline:
                time.sleep(1)
                j = client.get(f"/api/garnet/jobs/{jid}").json()
            return j

        def get(rel):
            return client.get(f"/runs/jobs/{jid}/{rel}").json()

        self.assertEqual(wait()["gate"], 1)
        objs0, ln0, eq0 = (get("input/sheet_objects.json"), get("input/gate1_line_numbers.json"),
                           get("input/sheet_equipment_bboxes.json"))
        v = next(e for e in eq0["objects"] if e["Tag"] == "V-2501")
        ports = v["Ports"][1:] + [{"side": "left", "point_px": [v["Left"], v["Top"] + 300]}]

        def gate1(body):
            return client.post(f"/api/garnet/jobs/{jid}/gates/1", json=body)
        self.assertEqual(gate1({"ports": {"NOPE-1": ports}}).status_code, 400)
        self.assertEqual(gate1({"ports": {"V-2501": [{"side": "diagonal", "point_px": [1, 1]}]}}).status_code, 422)
        self.assertEqual(gate1({"ports": {"V-2501": [{"side": "top", "point_px": [-5, 1]}]}}).status_code, 400)
        # Line numbers change only in their own list: a drawn one needs text, no object-style edits.
        box = {"x_min": 1, "y_min": 1, "x_max": 9, "y_max": 9}
        self.assertEqual(gate1({"added": [{"class_name": "Line Number", "bbox": box}]}).status_code, 400)
        self.assertEqual(gate1({"edits": {ln0["objects"][0]["id"]: "X"}}).status_code, 400)
        self.assertEqual(gate1({"reclassed": {objs0["objects"][0]["id"]: "line_number"}}).status_code, 400)
        self.assertEqual(gate1({"ports": {"V-2501": ports}}).status_code, 200)
        self.assertEqual(wait()["gate"], 2)
        edited = next(e for e in get("input/sheet_equipment_bboxes.json")["objects"] if e["Tag"] == "V-2501")
        self.assertEqual(([p["side"] for p in edited["Ports"]], edited["ports_source"]),
                         ([p["side"] for p in ports], "human_edit"))

        # The API restarts mid-run: the job comes back from its job.json as failed, and can
        # still go back to a gate it got past.
        saved = api.JOBS_DIR / jid / "job.json"
        saved.write_text(saved.read_text().replace('"status": "awaiting_gate"', '"status": "running"', 1))
        api.JOBS.pop(jid)
        api._load_saved_jobs()
        self.assertEqual(client.get(f"/api/garnet/jobs/{jid}").json()["status"], "failed")
        r = client.post(f"/api/garnet/jobs/{jid}/reopen", json={"gate": 1})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((r.json()["status"], r.json()["gate"]), ("awaiting_gate", 1))
        self.assertEqual(get("input/sheet_objects.json"), objs0)
        self.assertEqual(get("input/sheet_equipment_bboxes.json"), eq0)
        self.assertEqual(get("input/gate1_line_numbers.json"), ln0)

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
        is_ln = lambda o: o["class_name"].strip().lower() in ("line number", "line_number")  # noqa: E731
        gate1_ln = get("input/gate1_line_numbers.json")["objects"]
        self.assertTrue(gate1_ln and all(map(is_ln, gate1_ln)))
        # The upload is the master list: only its labels, each with text, no detector boxes.
        upload = json.loads((INPUT / f"{STEM}_line_numbers.json").read_text())["objects"]
        self.assertEqual(sorted(o["text"] for o in gate1_ln),
                         sorted(str(u["Text"]).strip() for u in upload if str(u.get("Text") or "").strip()))
        objs = get("input/sheet_objects.json")["objects"]
        drop = next(o["id"] for o in objs if o["class_name"] == "gate valve")
        tag = next(o["id"] for o in objs if o["class_name"] == "instrument tag")
        moved = next(o for o in objs if o["class_name"] == "check valve")
        new_box = {k: moved["bbox"][k] + 2 for k in ("x_min", "y_min", "x_max", "y_max")}
        added = {"class_name": "instrument tag", "text": "PI 0999",
                 "bbox": {"x_min": 10, "y_min": 10, "x_max": 60, "y_max": 60}}
        off = {"class_name": "gate valve", "bbox": {"x_min": 10, "y_min": 10, "x_max": 10, "y_max": 99999}}
        self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/gates/1", json={"added": [off]}).status_code, 400)
        # Line numbers: reject one and draw one; both go to the uploaded line-number file.
        ln_drop = gate1_ln[0]["id"]
        ln_new = {"class_name": "Line Number", "text": '9"-P-99-999',
                  "bbox": {"x_min": 100, "y_min": 100, "x_max": 300, "y_max": 130}}
        ln_expected = sorted([o["text"] for o in gate1_ln if o["id"] != ln_drop] + [ln_new["text"]])
        ln_texts = lambda: sorted(o["text"] for o in get("input/gate1_line_numbers.json")["objects"])  # noqa: E731
        r = client.post(f"/api/garnet/jobs/{jid}/gates/1", json={
            "rejected": [drop, ln_drop], "edits": {tag: "TI 0001"}, "added": [added, ln_new],
            "moved": {moved["id"]: new_box}, "reclassed": {moved["id"]: "globe valve"}})
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
        # Gate 1's line-number view is exactly the set the tracer got.
        self.assertEqual(sorted((o["id"], o.get("text")) for o in get("input/gate1_line_numbers.json")["objects"]),
                         sorted((o["id"], o.get("text")) for o in get("out/stage4_objects.json")["objects"] if is_ln(o)))
        self.assertIn(drop, {o["id"] for o in get("input/sheet_objects.pre_gate1.json")["objects"]})
        self.assertEqual(ln_texts(), ln_expected)  # and, above, exactly what Stage 4 got
        self.assertEqual(get("input/sheet_line_numbers.pre_gate1.json"), json.loads((INPUT / f"{STEM}_line_numbers.json").read_text()))
        traces = get("out/stage5b_trace_results.json")
        dropped_trace = next(iter(traces))
        self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/gates/2", json={"rejected": ["nope"]}).status_code, 400)

        # Snap: a stroke drawn 5 px below the pipe at y=988 (x 400-1000, obj_000195) lands on it.
        snap = client.post(f"/api/garnet/jobs/{jid}/trace-snap", json={"points": [[400, 993], [1000, 993]]}).json()
        self.assertIn(snap["points"][0][1], (987, 988, 989))
        self.assertGreater(snap["coverage"], 0.6)
        raw = client.post(f"/api/garnet/jobs/{jid}/trace-snap", json={"points": [[400, 993], [1000, 993]], "snap": False}).json()
        self.assertLess(raw["coverage"], 0.1)

        # Edit: keep only the first segment of another trace; draw one trace.
        edited_id, edited = next((k, v) for k, v in traces.items() if k != dropped_trace and len(v["segments"]) > 1)
        keep = [{k: edited["segments"][0][k] for k in ("x1", "y1", "x2", "y2")}]
        bad = {"drawn": [[{"x1": 0, "y1": 0, "x2": 99999, "y2": 0}]]}
        self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/gates/2", json=bad).status_code, 400)
        r = client.post(f"/api/garnet/jobs/{jid}/gates/2", json={
            "rejected": [dropped_trace], "modified": {edited_id: keep}, "drawn": [snap["segments"]]})
        self.assertEqual(r.status_code, 200, r.text)

        job = wait()
        self.assertEqual((job["status"], job["gate"]), ("awaiting_gate", 3))
        stage6 = get("out/stage6_trace_associations.json")
        self.assertNotIn(dropped_trace, {e["trace_id"] for e in stage6["trace_edges"]})
        self.assertIn(dropped_trace, get("out/stage5b_trace_results.json"))  # original untouched
        edges = {e["trace_id"]: e for e in stage6["trace_edges"]}
        self.assertEqual(len(edges[edited_id]["segments"]), 1)
        self.assertEqual(len(get("out/stage5b_trace_results.json")[edited_id]["segments"]), len(edited["segments"]))
        self.assertIn("manual_000001", edges)
        view = get("out/gate2/stage5b_trace_results.json")["manual_000001"]
        self.assertEqual((view["source"], view["terminal_type"]), ("human", "manual"))
        self.assertGreater(view["mask_coverage"], 0.6)
        ln = stage6["associations"]["line_numbers"]["accepted"][0]["id"]
        moved_ln = stage6["associations"]["line_numbers"]["accepted"][1]
        target = next(e["trace_id"] for e in stage6["trace_edges"] if e["trace_id"] != moved_ln["trace_id"])
        for bad in ({"reassigned": {moved_ln["id"]: dropped_trace}},             # rejected at Gate 2
                    {"reassigned": {moved_ln["id"]: target}, "decisions": {moved_ln["id"]: "rejected"}}):
            self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/gates/3", json=bad).status_code, 400)
        tag3 = stage6["associations"]["instrument_tags"]["accepted"][0]
        tag_target = next(e["trace_id"] for e in stage6["trace_edges"] if e["trace_id"] != tag3["trace_id"])
        r = client.post(f"/api/garnet/jobs/{jid}/gates/3", json={
            "decisions": {ln: "rejected"}, "edits": {ln: "X-1"},
            "reassigned": {moved_ln["id"]: target, tag3["id"]: tag_target}})
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
        tag_item = next(i for i in get("out/reviewed_associations.json")["associations"]["instrument_tags"]["accepted"]
                        if i["id"] == tag3["id"])
        self.assertEqual((tag_item["trace_id"], tag_item["system_assignment"]["trace_id"]), (tag_target, tag3["trace_id"]))
        unresolved = get("out/reviewed_associations.json")["unresolved"]["traces_without_line_number"]
        self.assertNotIn(target, unresolved)
        self.assertEqual(set(get("gates.json")), {"1", "2", "3"})

        # Reopen Gate 2 from a completed job: Stage 6 onward is archived, not deleted.
        r = client.post(f"/api/garnet/jobs/{jid}/reopen", json={"gate": 2})
        self.assertEqual(r.status_code, 200, r.text)
        job = r.json()
        self.assertEqual((job["status"], job["gate"], job["gates_confirmed"]), ("awaiting_gate", 2, [1]))
        names = {a["name"] for a in job["artifacts"]}
        self.assertNotIn("out/stage6_trace_associations.json", names)
        self.assertIn("history/001_reopen_gate2/out/stage6_trace_associations.json", names)
        self.assertIn("out/stage5b_trace_results.json", names)  # the trace itself is kept
        self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/reopen", json={"gate": 2}).status_code, 409)
        self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/gates/2", json={}).status_code, 200)
        job = wait()
        self.assertEqual((job["status"], job["gate"]), ("awaiting_gate", 3))
        stage6 = get("out/stage6_trace_associations.json")
        self.assertIn(dropped_trace, {e["trace_id"] for e in stage6["trace_edges"]})  # Gate 2 redone with no rejections
        self.assertEqual(ln_texts(), ln_expected)  # Gate 1's line-number edits survive a later Reopen

        # Reset to start: Gate 1 undone in the staged objects, whole trace archived.
        job = client.post(f"/api/garnet/jobs/{jid}/reopen", json={"gate": 1}).json()
        self.assertEqual((job["status"], job["gate"], job["gates_confirmed"]), ("awaiting_gate", 1, []))
        staged = {o["id"]: o for o in get("input/sheet_objects.json")["objects"]}
        self.assertIn(drop, staged)
        self.assertNotIn("human_000001", staged)
        self.assertEqual(get("input/sheet_line_numbers.json"), json.loads((INPUT / f"{STEM}_line_numbers.json").read_text()))
        self.assertEqual(ln_texts(), sorted(o["text"] for o in gate1_ln))
        self.assertFalse(any(a["name"].startswith("out/") for a in job["artifacts"]))
        self.assertEqual([h["event"] for h in job["history"]], ["confirm", "confirm", "confirm", "reopen", "confirm", "reopen"])
        # Save at Gate 1, load it back as a new job, and drive the LOADED job on: its steps must
        # be rebuilt (callables included), not just its status.
        bundle = client.get(f"/api/garnet/jobs/{jid}/bundle")
        self.assertEqual(bundle.status_code, 200)
        loaded = client.post("/api/garnet/jobs/load", files={"bundle": ("b.zip", bundle.content, "application/zip")}).json()
        self.assertNotEqual(loaded["id"], jid)
        self.assertEqual((loaded["status"], loaded["gate"], loaded["history"]), (job["status"], job["gate"], job["history"]))
        jid = loaded["id"]
        self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/gates/1", json={}).status_code, 200)
        job = wait()
        self.assertEqual((job["status"], job["gate"]), ("awaiting_gate", 2))

    def test_gate1_line_numbers_without_upload(self):
        """No upload: the first line-number edit creates the file from the tab; Reopen deletes it."""
        client = TestClient(app)
        files = {
            "file_input": ("s.jpg", (INPUT / f"{STEM}.jpg").read_bytes(), "image/jpeg"),
            "objects_json": ("o.json", (INPUT / f"{STEM}_objects.json").read_bytes()),
            "equipment_json": ("e.json", (INPUT / f"{STEM}_equipment_bboxes.json").read_bytes()),
        }
        jid = client.post("/api/garnet/jobs", files=files, data={"ocr_instruments": "false"}).json()["id"]

        def wait():
            j = client.get(f"/api/garnet/jobs/{jid}").json()
            deadline = time.time() + 600
            while j["status"] in ("queued", "running") and time.time() < deadline:
                time.sleep(1)
                j = client.get(f"/api/garnet/jobs/{jid}").json()
            return j

        def get(rel):
            return client.get(f"/runs/jobs/{jid}/{rel}").json()

        def boxes(rows):
            return sorted(tuple(o["bbox"].values()) for o in rows)
        self.assertEqual(wait()["gate"], 1)
        ln0 = get("input/gate1_line_numbers.json")["objects"]
        drop = ln0[0]
        self.assertEqual(client.post(f"/api/garnet/jobs/{jid}/gates/1", json={"rejected": [drop["id"]]}).status_code, 200)
        self.assertEqual(wait()["gate"], 2)
        ln1 = get("input/gate1_line_numbers.json")["objects"]
        self.assertEqual(boxes(ln1), boxes(ln0[1:]))  # text-less detector boxes are kept, too
        stage4 = [o for o in get("out/stage4_objects.json")["objects"] if o["class_name"] == "line number"]
        self.assertEqual(boxes(stage4), boxes(ln1))
        self.assertTrue(get("input/sheet_line_numbers.pre_gate1.json")["absent"])

        r = client.post(f"/api/garnet/jobs/{jid}/reopen", json={"gate": 1})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(client.get(f"/runs/jobs/{jid}/input/sheet_line_numbers.json").status_code, 404)
        self.assertEqual(client.get(f"/runs/jobs/{jid}/input/sheet_line_numbers.pre_gate1.json").status_code, 404)
        self.assertEqual(get("input/gate1_line_numbers.json")["objects"], ln0)

    def test_report_and_bundle_round_trip(self):
        import io, zipfile
        client = TestClient(app)
        files = {
            "file_input": ("my sheet\"x.jpg", (INPUT / f"{STEM}.jpg").read_bytes(), "image/jpeg"),
            "objects_json": ("o.json", (INPUT / f"{STEM}_objects.json").read_bytes()),
            "equipment_json": ("e.json", (INPUT / f"{STEM}_equipment_bboxes.json").read_bytes()),
            "line_numbers_json": ("l.json", (INPUT / f"{STEM}_line_numbers.json").read_bytes()),
        }
        job = client.post("/api/garnet/jobs", files=files,
                          data={"pause_at_gates": "false", "ocr_instruments": "false"}).json()
        deadline = time.time() + 600
        while job["status"] in ("queued", "running") and time.time() < deadline:
            time.sleep(1)
            job = client.get(f"/api/garnet/jobs/{job['id']}").json()
        self.assertEqual(job["status"], "completed")

        rep = client.get(f"/api/garnet/jobs/{job['id']}/report.txt")
        self.assertEqual(rep.status_code, 200)
        # Whatever the client did to the upload name, only header-safe characters come back.
        import re
        self.assertRegex(rep.headers["content-disposition"], r'^attachment; filename="my_sheet[\w.-]*_test_systems\.txt"$')
        ts = client.get(f"/runs/jobs/{job['id']}/out/test_systems.json").json()
        self.assertTrue(rep.text.startswith("=" * 78 + "\nGARNET TEST SYSTEMS REPORT"))
        for sysm in ts["systems"]:
            self.assertIn(sysm["line_number"], rep.text)  # every system, text exactly as stored

        zipped = client.get(f"/api/garnet/jobs/{job['id']}/bundle").content
        loaded = client.post("/api/garnet/jobs/load", files={"bundle": ("b.zip", zipped)}).json()
        self.assertEqual(loaded["status"], "completed")
        self.assertEqual(client.get(f"/runs/jobs/{loaded['id']}/out/test_systems.json").json(), ts)

        evil = io.BytesIO()
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("job.json", "{}")
            zf.writestr("../escape.txt", "x")
        self.assertEqual(client.post("/api/garnet/jobs/load", files={"bundle": ("e.zip", evil.getvalue())}).status_code, 400)
        self.assertEqual(client.post("/api/garnet/jobs/load", files={"bundle": ("n.zip", b"not a zip")}).status_code, 400)

    def test_multi_sheet_links_line_numbers(self):
        """4"-CUL-25-002007-B1A2-NI is drawn on both 0002A and 0003: one cross-sheet line."""
        client = TestClient(app)

        def run(stem: str, eq: str) -> str:
            files = {
                "file_input": (f"{stem}.jpg", (INPUT / f"{stem}.jpg").read_bytes(), "image/jpeg"),
                "objects_json": ("o.json", (INPUT / f"{stem}_objects.json").read_bytes()),
                "equipment_json": ("e.json", (INPUT / f"{stem}_{eq}.json").read_bytes()),
                "line_numbers_json": ("l.json", (INPUT / f"{stem}_line_numbers.json").read_bytes()),
            }
            # OCR on: connector text (the drawing each one continues on) is what pairs them.
            return client.post("/api/garnet/jobs", files=files,
                               data={"pause_at_gates": "false", "ocr_instruments": "true"}).json()["id"]

        ids = [run(STEM, "equipment_bboxes"), run("14780-8120-25-25-0003", "equipment_bboxes")]
        deadline = time.time() + 900
        while time.time() < deadline and any(
                client.get(f"/api/garnet/jobs/{j}").json()["status"] in ("queued", "running") for j in ids):
            time.sleep(1)

        m = client.get("/api/garnet/merge/test_systems.json", params={"jobs": ",".join(ids + ["missing"])}).json()
        self.assertEqual(len(m["sheets"]), 2)
        self.assertEqual(m["skipped_sheets"][0]["reason"], "job not found")
        by_key = {x["line_number"]: x for x in m["lines"]}
        shared = by_key['4"-CUL-25-002007-B1A2-NI']
        self.assertEqual(shared["sheet_count"], 2)
        self.assertEqual({e["sheet"] for e in shared["sheets"]}, {f"{STEM}.jpg", "14780-8120-25-25-0003.jpg"})
        # Linking is exact: every multi-sheet line is one text present on each of its sheets.
        for x in m["lines"]:
            if x["no_text"]:
                self.assertEqual(x["sheet_count"], 1)  # a textless label never links
                self.assertIn(":", x["line_number"])
        # Every per-sheet system shows up exactly once in the merge.
        total = sum(len(client.get(f"/runs/jobs/{j}/out/test_systems.json").json()["systems"]) for j in ids)
        self.assertEqual(sum(len(x["sheets"]) for x in m["lines"]), total)

        txt = client.get("/api/garnet/merge/test_systems.txt", params={"jobs": ",".join(ids)}).text
        self.assertIn('4"-CUL-25-002007-B1A2-NI   <MULTI-SHEET', txt)
        self.assertEqual(client.get("/api/garnet/merge/test_systems.json").status_code, 400)

        # Off-page connectors: 0002A's obj_000200 reads 25-0003, 0003's obj_000142 reads 25-0002A, and
        # both sit on 4"-CUL-25-002007 -> a pair confirmed by line number.
        a, b = f"{STEM}.jpg", "14780-8120-25-25-0003.jpg"
        pairs = {(lk["from"]["sheet"], lk["from"]["connector"], lk["to"]["sheet"], lk["to"]["connector"]): lk
                 for lk in m["connector_links"] if lk["to"]}
        lk = pairs.get((a, "obj_000200", b, "obj_000142")) or pairs.get((b, "obj_000142", a, "obj_000200"))
        self.assertIsNotNone(lk, m["connector_links"])
        self.assertEqual(lk["status"], "matched_by_line_number")
        self.assertEqual(lk["line_numbers"], ['4"-CUL-25-002007-B1A2-NI'])
        self.assertIn({"status": "matched_by_line_number", "from": f"{lk['from']['sheet']}:{lk['from']['connector']}",
                       "to": f"{lk['to']['sheet']}:{lk['to']['connector']}"}, shared["connector_links"])
        self.assertFalse(shared["continuation_missing"])
        # 26-0003 is one character off 25-0003: reported, pointed at, never linked.
        stray = [lk for lk in m["connector_links"] if lk["status"] == "target_not_loaded" and lk["ref"] == "26-0003"]
        self.assertTrue(stray and stray[0]["possible_target"] == [b], m["connector_links"])
        self.assertIn("OFF-PAGE CONNECTOR LINKS", txt)

    def test_bad_json_rejected_before_staging(self):
        client = TestClient(app)
        files = {
            "file_input": ("s.jpg", (INPUT / f"{STEM}.jpg").read_bytes(), "image/jpeg"),
            "objects_json": ("o.json", b"{not json"),
        }
        self.assertEqual(client.post("/api/garnet/jobs", files=files).status_code, 400)


if __name__ == "__main__":
    unittest.main()
