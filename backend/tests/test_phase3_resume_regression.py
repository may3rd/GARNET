"""Regression coverage for review driven pipeline resume boundaries."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet import pid_extractor


class ResumePipeline(pid_extractor.PIDPipeline):
    """Deterministic runner that exercises the real manifest implementation."""

    def __init__(self, out_dir: str | Path, *, fail_stage: int | None = None) -> None:
        self._input_path = Path(out_dir) / "image.png"
        self._input_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._input_path.exists():
            self._input_path.write_bytes(b"phase3 image")
        decisions_path = Path(out_dir) / "stage8_review_decisions.json"
        if not decisions_path.exists():
            decisions_path.write_text('{"decisions": []}', encoding="utf-8")
        super().__init__(str(self._input_path), output_dir=out_dir)
        self.called: list[str] = []
        self.fail_stage = fail_stage

    def _record(self, name: str, *artifacts: str) -> None:
        self.called.append(name)
        for artifact in artifacts or (f"{name}_artifact",):
            self._save_json(Path(artifact).stem, {"stage": name})
        if self.fail_stage is not None and name == f"stage{self.fail_stage}":
            raise RuntimeError(f"{name} failed")

    def stage1_input_normalization(self): self._record("stage1")
    def stage2_ocr_discovery(self): self._record("stage2")
    def stage4_object_detection(self): self._record("stage4")
    def stage4_line_number_fusion(self): self._record("stage4b")
    def stage4_instrument_tag_fusion(self): self._record("stage4c")
    def stage5_pipe_mask(self): self._record("stage5")
    def stage5b_pipe_trace(self): self._record("stage5b")
    def stage6_trace_associations(self): self._record("stage6", "stage6_line_number_review.json")
    def stage7_geometric_graph_assembly(self): self._record("stage7", "stage7_graph.json")
    def stage7c_page_connector_labeling(self): self._record("stage7c")
    def stage7b_graph_export(self): self._record("stage7b", "stage7b_graph_v1.json")
    def stage8_graph_qa(self): self._record("stage8", "stage8_review_items.json")
    def stage9_apply_review_decisions(self):
        self._record(
            "stage9",
            "stage9_corrected_graph.json",
            "stage9_review_resolutions.json",
            "stage9_correction_audit.json",
            "stage9_correction_summary.json",
        )
    def stage10_process_exports(self):
        self._record(
            "stage10",
            "stage10_line_list.json",
            "stage10_equipment_connectivity.json",
            "stage10_inline_mto.json",
            "stage10_instrument_index.json",
        )
    def stage11_connection_overlay(self): self._record("stage11")


def _manifest(path: Path) -> dict:
    return json.loads((path / "stage_manifest.json").read_text(encoding="utf-8"))


class Phase3ResumeRegressionTests(unittest.TestCase):
    def test_stage6_review_edit_invalidates_stage7_and_resume_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            ResumePipeline(tmp).run(stop_after=11)
            review = Path(tmp) / "stage6_line_number_review.json"
            review.write_text('{"edited": true}', encoding="utf-8")

            first = ResumePipeline(tmp)
            first.run(stop_after=7, resume=True)
            self.assertEqual(first.called, ["stage7", "stage7c", "stage7b"])
            self.assertFalse((Path(tmp) / "stage9_corrected_graph.json").exists())
            self.assertFalse((Path(tmp) / "stage10_line_list.json").exists())

            second = ResumePipeline(tmp)
            second.run(stop_after=11, resume=True)
            self.assertEqual(second.called, ["stage8", "stage9", "stage10", "stage11"])

    def test_stage8_decision_edit_reruns_stage9_and_downstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            ResumePipeline(tmp).run(stop_after=11)
            decisions = Path(tmp) / "stage8_review_decisions.json"
            decisions.write_text('{"edited": true}', encoding="utf-8")

            ResumePipeline(tmp).run(stop_after=7, resume=True)
            self.assertFalse((Path(tmp) / "stage9_corrected_graph.json").exists())
            self.assertFalse((Path(tmp) / "stage10_line_list.json").exists())
            resumed = ResumePipeline(tmp)
            resumed.run(stop_after=11, resume=True)
            self.assertEqual(resumed.called, ["stage8", "stage9", "stage10", "stage11"])

    def test_failed_stage9_does_not_leave_completed_downstream_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            ResumePipeline(tmp).run(stop_after=11)
            (Path(tmp) / "stage8_review_decisions.json").write_text('{"edited": true}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "stage9 failed"):
                ResumePipeline(tmp, fail_stage=9).run(stop_after=11, resume=True)

            resumed = ResumePipeline(tmp)
            resumed.run(stop_after=11, resume=True)
            self.assertEqual(resumed.called, ["stage9", "stage10", "stage11"])

    def test_legacy_missing_stage7_input_fingerprint_reruns_stage7_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            ResumePipeline(tmp).run(stop_after=11)
            manifest_path = Path(tmp) / "stage_manifest.json"
            manifest = _manifest(Path(tmp))
            next(item for item in manifest["stages"] if item["name"] == "stage7_geometric_graph_assembly").pop(
                "input_fingerprints", None
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            resumed = ResumePipeline(tmp)
            resumed.run(stop_after=11, resume=True)
            self.assertEqual(resumed.called, ["stage7", "stage7c", "stage7b", "stage8", "stage9", "stage10", "stage11"])

    def test_input_and_review_signature_mismatch_preserves_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ResumePipeline(tmp).run(stop_after=11)
            manifest_path = Path(tmp) / "stage_manifest.json"
            original = manifest_path.read_bytes()
            (Path(tmp) / "image.png").write_bytes(b"changed input")
            (Path(tmp) / "stage6_line_number_review.json").write_text('{"edited": true}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "input.sha256"):
                ResumePipeline(tmp).run(stop_after=11, resume=True)
            self.assertEqual(manifest_path.read_bytes(), original)

    def test_api_shared_graph_v1_invalidation_allows_resume(self):
        from api import _mark_pipeline_stale_from

        with tempfile.TemporaryDirectory() as tmp:
            ResumePipeline(tmp).run(stop_after=11)
            _mark_pipeline_stale_from(tmp, "stage9_apply_review_decisions", "stage8_review_decisions.json")
            self.assertFalse((Path(tmp) / "stage7b_graph_v1.json").exists())

            resumed = ResumePipeline(tmp)
            resumed.run(stop_after=11, resume=True)
            self.assertEqual(resumed.called, ["stage7b", "stage8", "stage9", "stage10", "stage11"])


if __name__ == "__main__":
    unittest.main()
