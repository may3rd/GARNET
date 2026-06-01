"""Guards that the production path tracer stays CV-only."""

import unittest
from dataclasses import fields
from pathlib import Path

from garnet.pid_extractor import PipelineConfig


class TestCvOnlyPathTracerPipeline(unittest.TestCase):
    def test_pipeline_config_has_no_vlm_port_detection_fields(self):
        names = {field.name for field in fields(PipelineConfig)}
        self.assertNotIn("port_detection_mode", names)
        self.assertNotIn("port_detection_model", names)

    def test_pid_extractor_does_not_import_agent2_hybrid(self):
        source = Path("garnet/pid_extractor.py").read_text()
        self.assertNotIn("agent2_hybrid", source)
        self.assertNotIn("compute_port_vlm", source)


if __name__ == "__main__":
    unittest.main()

class TestGeometricOnlyPipelineCleanup(unittest.TestCase):
    def test_pid_extractor_has_no_legacy_stage_methods_or_imports(self):
        source = Path("garnet/pid_extractor.py").read_text()
        forbidden = [
            "stage6_morphological_sealing",
            "stage7_skeleton_generation",
            "stage8_skeleton_node_detection",
            "stage9_node_clustering",
            "stage10_edge_tracing",
            "stage11_junction_review",
            "stage9_continuity_check",
            "stage10_recovery_loop",
            "stage5_geometric_line_detection",
            "stage4_equipment_tag_fusion",
            "run_pipe_seal_stage",
            "run_pipe_skeleton_stage",
            "run_pipe_node_stage",
            "run_pipe_crossing_stage",
            "run_pipe_edge_stage",
            "run_polyline_simplification_stage",
            "split_edges_at_inline_elements",
            "run_pipe_junction_stage",
            "run_continuity_checker_stage",
            "run_line_detection_inpaint",
            "run_equipment_tag_fusion_stage",
        ]
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_geometric_pipeline_uses_compact_stage_numbers(self):
        from garnet.pid_extractor import PIDPipeline

        pipeline = PIDPipeline.__new__(PIDPipeline)
        names_by_number = [(num, name) for num, name, _ in pipeline._stage_definitions()]

        self.assertIn((6, "stage6_trace_associations"), names_by_number)
        self.assertIn((7, "stage7_geometric_graph_assembly"), names_by_number)
        self.assertIn((8, "stage8_graph_qa"), names_by_number)
        self.assertIn((9, "stage9_apply_review_decisions"), names_by_number)
        self.assertIn((10, "stage10_process_exports"), names_by_number)
        self.assertNotIn((15, "stage10_process_exports"), names_by_number)
