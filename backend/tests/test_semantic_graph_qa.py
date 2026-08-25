"""Regression tests for the advisory semantic-graph QA stage.

This stage is intentionally *not wired into the pipeline* yet (see the module
docstring in `garnet/semantic_graph_qa.py`). These tests lock in its contract:
it is advisory and fails soft (never raises, never blocks a pipeline run) when
no API key is configured or a model call errors.
"""

import unittest
from unittest.mock import patch

from garnet import semantic_graph_qa
from garnet.semantic_graph_qa import SemanticQaConfig, run_semantic_graph_qa


class SemanticGraphQaContractTests(unittest.TestCase):
    def _payload(self) -> dict:
        return {
            "schema_version": "graph_v1",
            "document": {"doc_id": "DWG-100"},
            "nodes": [
                {"id": "eq::pump-1", "type": "pump", "x": 10.0, "y": 20.0},
                {"id": "jn::n1", "type": "junction", "x": 50.0, "y": 60.0},
            ],
            "edges": [
                {
                    "id": "ed::1",
                    "src": "eq::pump-1",
                    "dst": "jn::n1",
                    "type": "solid",
                }
            ],
        }

    def test_skips_without_api_key_and_never_calls_model(self) -> None:
        """Without an API key the stage must return skipped with empty issues."""
        with patch.object(semantic_graph_qa, "_resolve_api_key", return_value=""):
            with patch.object(semantic_graph_qa, "_call_model") as call_model:
                result = run_semantic_graph_qa(
                    image_id="DWG-100",
                    graph_payload=self._payload(),
                    cfg=SemanticQaConfig(),
                )

        call_model.assert_not_called()
        self.assertTrue(result["skipped"])
        self.assertEqual(result["issues"], [])
        self.assertIn("OPENROUTER_API_KEY", result.get("skip_reason", ""))

    def test_builds_summary_without_error(self) -> None:
        """A summary is still produced even when the stage is skipped."""
        with patch.object(semantic_graph_qa, "_resolve_api_key", return_value=""):
            result = run_semantic_graph_qa(
                image_id="DWG-100",
                graph_payload=self._payload(),
                cfg=SemanticQaConfig(),
            )
        self.assertIn("summary", result)
        self.assertIsNotNone(result["summary"])

    def test_fails_soft_when_model_call_errors(self) -> None:
        """A raising model call must not propagate — it degrades to skipped."""
        with patch.object(semantic_graph_qa, "_resolve_api_key", return_value="test-key"):
            with patch.object(semantic_graph_qa, "_call_model", side_effect=RuntimeError("boom")):
                result = run_semantic_graph_qa(
                    image_id="DWG-100",
                    graph_payload=self._payload(),
                    cfg=SemanticQaConfig(),
                )
        self.assertTrue(result["skipped"])
        self.assertEqual(result["issues"], [])
        self.assertIn("boom", result.get("skip_reason", ""))


if __name__ == "__main__":
    unittest.main()
