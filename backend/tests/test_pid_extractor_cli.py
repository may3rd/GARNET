import json
import tempfile
import unittest
from pathlib import Path

from garnet import pid_extractor


class FakePipeline(pid_extractor.PIDPipeline):
    def __init__(self, out_dir: str | Path, fail_stage: int | None = None) -> None:
        super().__init__("image.png", out_dir=out_dir)
        self.called: list[str] = []
        self.fail_stage = fail_stage

    def _record(self, name: str) -> None:
        self.called.append(name)
        self._save_json(f"{name}_artifact", {"stage": name})
        if self.fail_stage is not None and name == f"stage{self.fail_stage}":
            raise RuntimeError(f"{name} failed")

    def stage1_input_normalization(self) -> None:
        self._record("stage1")

class PIDPipelineRunnerTests(unittest.TestCase):
    def test_stage_definitions_follow_master_plan_order(self) -> None:
        pipe = FakePipeline(tempfile.mkdtemp())

        stage_names = [name for _, name, _ in pipe._stage_definitions()]

        self.assertEqual(
            stage_names,
            [
                "stage1_input_normalization",
            ],
        )

    def test_run_stops_after_requested_stage_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)

            pipe.run(stop_after=1)

            self.assertEqual(pipe.called, ["stage1"])
            manifest = json.loads((Path(tmp) / "stage_manifest.json").read_text())
            self.assertEqual(manifest["stop_after"], 1)
            self.assertEqual(
                [item["name"] for item in manifest["stages"]],
                [
                    "stage1_input_normalization",
                ],
            )
            self.assertTrue(all(item["status"] == "completed" for item in manifest["stages"]))

    def test_run_writes_failed_stage_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp, fail_stage=2)

            with self.assertRaisesRegex(ValueError, "stop_after must be between 1 and 1"):
                pipe.run(stop_after=2)

    def test_run_writes_failed_stage_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp, fail_stage=1)

            with self.assertRaisesRegex(RuntimeError, "stage1 failed"):
                pipe.run(stop_after=1)

            manifest = json.loads((Path(tmp) / "stage_manifest.json").read_text())
            self.assertEqual(manifest["stages"][0]["status"], "failed")
            self.assertIn("stage1 failed", manifest["stages"][0]["error"])


if __name__ == "__main__":
    unittest.main()
