"""Fast gate tests for garnet.pipeline: no tracer, no model."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from garnet import pipeline

IN = Path(__file__).parent / "input"
STEM = "14780-8120-25-25-0002A"


def stage(d: Path, with_b=True):
    for n in (f"{STEM}.jpg", f"{STEM}_objects.json", f"{STEM}_line_numbers.json") + ((f"{STEM}_equipment_bboxes.json",) if with_b else ()):
        shutil.copy(IN / n, d / n)


class Gates(unittest.TestCase):
    def test_g1_stops_driver_without_tracer(self):
        with tempfile.TemporaryDirectory() as fx, tempfile.TemporaryDirectory() as root:
            stage(Path(fx), with_b=False)
            with mock.patch.object(pipeline, "run", side_effect=AssertionError("tracer must not run")):
                rc = pipeline.main(["--job", "j", "--sheets", STEM, "--from-fixtures", fx, "--root", root, "--no-agents"])
            self.assertEqual(rc, 1)
            rep = json.loads((Path(root) / "j" / "run_report.json").read_text())
            g = rep["sheets"][STEM]["gates"]
            self.assertEqual(g["G1"]["status"], "fail")
            self.assertIn("no equipment file", g["G1"]["reason"])
            self.assertNotIn("G3", g)

    def test_g1_names_dropped_items(self):
        with tempfile.TemporaryDirectory() as d:
            stage(Path(d))
            p = Path(d) / f"{STEM}_equipment_bboxes.json"
            j = json.loads(p.read_text())
            for o in j["objects"]:
                o["Equipment_type"] = "Flux Capacitor"
            p.write_text(json.dumps(j))
            ok, why, info = pipeline.check_equipment(Path(d), STEM)
            self.assertFalse(ok)
            self.assertIn("V-2501", why)

    def test_g0_frame_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            stage(Path(d))
            self.assertTrue(pipeline.check_frame(Path(d), STEM)[0])
            p = Path(d) / f"{STEM}_line_numbers.json"
            j = json.loads(p.read_text())
            j["image_width"] = 100
            p.write_text(json.dumps(j))
            ok, why = pipeline.check_frame(Path(d), STEM)
            self.assertFalse(ok)
            self.assertIn("line_numbers.json declares 100x", why)

    def test_dispatch_attempt_cap(self):
        rep = {"attempts": {}}
        item = {"id": "s:equipment", "max_attempts": 2, "attempt": 0}
        self.assertTrue(pipeline.claim_attempt(rep, item))
        self.assertTrue(pipeline.claim_attempt(rep, item))
        self.assertEqual(item["attempt"], 2)
        self.assertFalse(pipeline.claim_attempt(rep, item))

    def test_apply_decisions(self):
        g = {"nodes": [{"id": "n1"}], "line_list": [{"edges": ["e1", "e2"]}],
             "edges": [{"id": "e1", "u": "n1", "v": "n1", "evidence": {}}, {"id": "e2", "u": "n1", "v": "n1", "evidence": {}}]}
        done = pipeline.apply_decisions({"s": g}, [{"target": "e1", "decision": "reject"},
                                                    {"target": "e2", "decision": "confirm"},
                                                    {"target": "n1", "decision": "fix", "fix": {"tag": "X"}}])
        self.assertEqual(len(done), 3)
        self.assertEqual([e["id"] for e in g["edges"]], ["e2"])
        self.assertEqual(g["edges"][0]["evidence"]["review"], "confirmed")
        self.assertEqual(g["nodes"][0]["tag"], "X")
        self.assertEqual(g["line_list"][0]["edges"], ["e2"])


if __name__ == "__main__":
    unittest.main()
