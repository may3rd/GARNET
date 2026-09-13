import time
import threading
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import numpy as np

try:
    from api import PIPELINE_LAST_STAGE, _resolve_pipeline_stage, app
except ModuleNotFoundError as exc:
    if exc.name == "pdf2image":
        app = None
        PIPELINE_LAST_STAGE = None
        _resolve_pipeline_stage = None
    else:
        raise


@unittest.skipIf(app is None, "pdf2image is not installed in this test environment")
class PipelineApiTests(unittest.TestCase):
    def test_pipeline_system_runs_only_one_child_at_a_time(self) -> None:
        api_module = __import__("api")
        active = 0
        maximum = 0
        counter_lock = threading.Lock()

        def fake_execute(*_args, **_kwargs):
            nonlocal active, maximum
            with counter_lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with counter_lock:
                active -= 1

        jobs = {
            "job-a": {"system_id": "system-one"},
            "job-b": {"system_id": "system-one"},
        }
        args = ("unused.png", "unused-dir", 4, "ocrmac", 0.1, "weights.pt")
        with patch.dict("api.PIPELINE_JOBS", jobs, clear=False), patch("api._execute_pipeline_job", side_effect=fake_execute):
            threads = [
                threading.Thread(target=api_module._run_pipeline_job, args=(job_id, *args))
                for job_id in jobs
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(maximum, 1)

    def test_pipeline_system_create_persists_confirmed_sheet_ids(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp, patch("api.PIPELINE_SYSTEMS_DIR", str(Path(tmp) / "systems")), patch(
            "api.PIPELINE_JOBS_DIR", str(Path(tmp) / "jobs")
        ), patch("api.resolve_pipeline_weight_file", return_value="weights.pt"), patch(
            "api._start_pipeline_system", create=True
        ) as start_system:
            response = client.post(
                "/api/pipeline/systems",
                files=[
                    ("files", ("first.png", b"first", "image/png")),
                    ("files", ("second.png", b"second", "image/png")),
                ],
                data={"sheet_ids": [" DWG-100 ", "DWG-200"], "ocr_route": "ocrmac"},
            )

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual([page["sheet_id"] for page in payload["pages"]], [" DWG-100 ", "DWG-200"])
            manifest_path = Path(tmp) / "systems" / payload["system_id"] / "system_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_version"], 1)
            self.assertEqual(manifest["status"], "processing")
            self.assertEqual(manifest["audit_history"][0]["event"], "system_created")
            self.assertEqual([page["source_filename"] for page in manifest["pages"]], ["first.png", "second.png"])
            for page in manifest["pages"]:
                job = __import__("api").PIPELINE_JOBS[page["job_id"]]
                self.assertEqual(job["document_id"], page["sheet_id"])
                self.assertEqual(job["system_id"], payload["system_id"])
            start_system.assert_called_once_with(payload["system_id"])

    def test_pipeline_system_create_rejects_duplicate_sheet_ids_case_insensitively(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/api/pipeline/systems",
            files=[
                ("files", ("first.png", b"first", "image/png")),
                ("files", ("second.png", b"second", "image/png")),
            ],
            data={"sheet_ids": ["DWG-100", " dwg-100 "], "ocr_route": "ocrmac"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("unique", response.text.lower())

    def test_pipeline_system_connector_review_regenerates_self_contained_graph(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            systems_root = Path(tmp) / "systems"
            jobs_root = Path(tmp) / "jobs"
            system_id = "system-review"
            pages = []
            jobs = {}
            for sheet_id, target, edge_id in (
                ("DWG-100", "DWG-200", "edge-a"),
                ("DWG-200", "DWG-100", "edge-b"),
            ):
                job_id = f"job-{sheet_id}"
                job_dir = jobs_root / job_id
                job_dir.mkdir(parents=True)
                graph = {
                    "schema_version": "graph_v1",
                    "document": {"doc_id": sheet_id},
                    "nodes": [],
                    "edges": [{
                        "id": edge_id,
                        "off_page_connector": {
                            "reference_type": "drawing",
                            "reference_value": target,
                            "target_sheet_reference": target,
                            "connector_key": "10-P-100-A",
                            "direction": "bidirectional",
                            "exit_terminal": "source",
                        },
                    }],
                }
                (job_dir / "stage7b_graph_v1.json").write_text(json.dumps(graph), encoding="utf-8")
                (job_dir / "stage_manifest.json").write_text(
                    json.dumps({"stages": [{"num": 11, "name": "stage11_connection_overlay", "status": "completed"}]}),
                    encoding="utf-8",
                )
                pages.append({"sheet_id": sheet_id, "source_filename": f"{sheet_id}.png", "job_id": job_id})
                jobs[job_id] = {
                    "job_id": job_id,
                    "status": "completed",
                    "current_stage": "stage11_connection_overlay",
                    "job_dir": str(job_dir),
                    "created_at": time.time(),
                    "stop_after": 11,
                    "system_id": system_id,
                    "document_id": sheet_id,
                }
            system_dir = systems_root / system_id
            system_dir.mkdir(parents=True)
            (system_dir / "system_manifest.json").write_text(
                json.dumps({
                    "manifest_version": 1,
                    "system_id": system_id,
                    "created_at": time.time(),
                    "config": {},
                    "pages": pages,
                    "connector_review": {"revision": 0, "connector_overrides": [], "manual_pairs": []},
                    "connector_review_audit": [],
                    "merge": {"status": "stale"},
                }),
                encoding="utf-8",
            )

            with patch("api.PIPELINE_SYSTEMS_DIR", str(systems_root)), patch("api.PIPELINE_JOBS_DIR", str(jobs_root)), patch.dict(
                "api.PIPELINE_JOBS", jobs, clear=False
            ):
                response = client.put(
                    f"/api/pipeline/systems/{system_id}/connector-review",
                    json={
                        "connector_overrides": [],
                        "manual_pairs": [],
                        "reviewer": "qa-user",
                    },
                )
                graph_response = client.get(f"/api/pipeline/systems/{system_id}/graph")

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(graph_response.status_code, 200, graph_response.text)
            graph = graph_response.json()
            self.assertEqual(graph["schema_version"], "graph_v2")
            self.assertEqual(len(graph["sheets"]), 2)
            self.assertEqual(graph["sheets"][0]["graph_v1"]["schema_version"], "graph_v1")
            self.assertEqual(len(graph["cross_sheet_edges"]), 1)
            combined = graph["combined_graph"]
            self.assertEqual(len(combined["connectors"]), 2)
            self.assertEqual(len(combined["relationships"]), 1)
            self.assertEqual(combined["relationships"][0]["type"], "cross_sheet_continues")
            connector_ids = {connector["id"] for connector in combined["connectors"]}
            relationship = combined["relationships"][0]
            self.assertTrue(set(relationship["connector_ids"]) <= connector_ids)
            self.assertTrue(relationship["source"] in connector_ids)
            self.assertTrue(relationship["target"] in connector_ids)
            self.assertEqual(relationship["review_state"], "accepted")
            self.assertEqual(relationship["semantic_state"], "reviewed")
            self.assertEqual(relationship["provenance"]["review"]["revision"], 1)
            self.assertEqual(graph["connector_review"]["revision"], 1)
            self.assertTrue(graph["release_gate"]["release_ready"])
            review = response.json()["connector_review"]
            self.assertEqual(review["connector_overrides"], [])
            self.assertEqual(review["raw_connectors"][0]["connector_key"], "10-P-100-A")
            self.assertEqual(review["effective_connectors"][0]["connector_key"], "10-P-100-A")

    def test_pipeline_system_release_aggregates_phase8_views_with_qualified_ids(self) -> None:
        api_module = __import__("api")
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            systems_root = Path(tmp) / "systems"
            jobs_root = Path(tmp) / "jobs"
            system_id = "phase8-system"
            pages = []
            jobs = {}
            for sheet_id in ("SHEET-A", "SHEET-B"):
                job_id = f"job-{sheet_id}"
                job_dir = jobs_root / job_id
                job_dir.mkdir(parents=True)
                graph = {
                    "schema_version": "graph_v1",
                    "document": {"doc_id": sheet_id},
                    "nodes": [
                        {"id": "equip-1", "type": "equipment"},
                        {"id": "junction-1", "type": "crossing"},
                    ],
                    # Deliberately collide an equipment node and an edge ID.
                    "edges": [{"id": "equip-1", "source": "equip-1", "target": "junction-1"}],
                    "relationships": [{"id": "rel-1", "type": "edge_to_equipment", "source": "equip-1", "target": "equip-1"}],
                }
                (job_dir / "stage7b_graph_v1.json").write_text(json.dumps(graph), encoding="utf-8")
                (job_dir / "stage9_corrected_graph.json").write_text(json.dumps(graph), encoding="utf-8")
                (job_dir / "stage_manifest.json").write_text(
                    json.dumps({"stages": [{"num": 11, "name": "stage11_connection_overlay", "status": "completed"}]}),
                    encoding="utf-8",
                )
                (job_dir / "stage10_process_boundaries.json").write_text(json.dumps({
                    "schema_version": "phase8_boundary_candidates_v1", "release_ready": True,
                    "source_graph_content_sha256": api_module._phase8_graph_content_sha256(graph),
                    "candidates": [{"id": "b-1", "member_edge_ids": ["equip-1"], "node_ids": ["equip-1"]}],
                }), encoding="utf-8")
                (job_dir / "stage10_test_package_candidates.json").write_text(json.dumps({
                    "schema_version": "phase8_test_package_candidates_v1", "release_ready": True,
                    "source_graph_content_sha256": api_module._phase8_graph_content_sha256(graph),
                    "candidates": [{"id": "pkg-1", "edge_ids": ["equip-1"], "line_ids": ["L-1"], "equipment_ids": ["equip-1"]}],
                }), encoding="utf-8")
                (job_dir / "stage10_engineering_view_summary.json").write_text(json.dumps({
                    "release_ready": True,
                    "graph_revision": f"{sheet_id}-r1",
                    "graph_content_sha256": api_module._phase8_graph_content_sha256(graph),
                    "source_graph_content_sha256": api_module._phase8_graph_content_sha256(graph),
                    "graph_counts": {"node_count": 2, "edge_count": 1, "relationship_count": 1},
                }), encoding="utf-8")
                (job_dir / "stage10_llm_projections.json").write_text(json.dumps({
                    "schema_version": "llm_projections_v1",
                    "source_graph_content_sha256": api_module._phase8_graph_content_sha256(graph),
                    "process_description": {"graph": {
                        "entities": [
                            {"id": "equip-1", "type": "equipment"},
                            {"id": "junction-1", "type": "topology"},
                            {"id": "equip-1", "type": "route"},
                        ],
                        "routes": [{
                            "id": "equip-1", "source": "equip-1", "target": "junction-1",
                            "flow": {"source": "equip-1", "target": "junction-1"},
                        }],
                        "relationships": [{"id": "rel-1", "type": "edge_to_equipment", "source": "equip-1", "target": "equip-1"}],
                    }},
                }), encoding="utf-8")
                pages.append({"sheet_id": sheet_id, "source_filename": f"{sheet_id}.png", "job_id": job_id})
                jobs[job_id] = {
                    "job_id": job_id, "status": "completed", "current_stage": "stage11_connection_overlay",
                    "job_dir": str(job_dir), "created_at": time.time(), "stop_after": 11,
                    "system_id": system_id, "document_id": sheet_id,
                }
            system_dir = systems_root / system_id
            system_dir.mkdir(parents=True)
            (system_dir / "system_manifest.json").write_text(json.dumps({
                "manifest_version": 1, "system_id": system_id, "pages": pages,
                "connector_review": {"revision": 1, "connector_overrides": [], "manual_pairs": []},
                "connector_review_audit": [], "merge": {"status": "stale"},
            }), encoding="utf-8")

            with patch("api.PIPELINE_SYSTEMS_DIR", str(systems_root)), patch("api.PIPELINE_JOBS_DIR", str(jobs_root)), patch.dict(
                "api.PIPELINE_JOBS", jobs, clear=False
            ):
                graph = api_module._regenerate_pipeline_system_graph(system_id)

            self.assertEqual(graph["release_gate"]["phase8_views_status"], "ready")
            combined = graph["combined_graph"]["phase8_views"]
            self.assertEqual(combined["summary"]["page_count"], 2)
            self.assertEqual(combined["summary"]["graph_counts"], {"node_count": 4, "edge_count": 2, "relationship_count": 2})
            self.assertEqual(combined["summary"]["graph_revision_values"], ["SHEET-A-r1", "SHEET-B-r1"])
            self.assertEqual(
                len(combined["summary"]["graph_content_sha256_values"]),
                2,
            )
            self.assertEqual({item["id"] for item in combined["process_boundaries"]}, {"boundary::SHEET-A::b-1", "boundary::SHEET-B::b-1"})
            self.assertEqual({item["id"] for item in combined["test_package_candidates"]}, {"test_package::SHEET-A::pkg-1", "test_package::SHEET-B::pkg-1"})
            self.assertEqual(len(combined["llm_projections"]), 2)
            for page_projection in combined["llm_projections"]:
                projection_graph = page_projection["projection"]["process_description"]["graph"]
                entities = {entity["id"] for entity in projection_graph["entities"]}
                route = projection_graph["routes"][0]
                relationship = projection_graph["relationships"][0]
                self.assertEqual(route["source"], f"equipment::{page_projection['sheet_id']}::equip-1")
                self.assertEqual(route["target"], f"topology::{page_projection['sheet_id']}::junction-1")
                self.assertEqual(route["flow"]["source"], route["source"])
                self.assertEqual(route["flow"]["target"], route["target"])
                self.assertEqual(relationship["source"], f"edge::{page_projection['sheet_id']}::equip-1")
                self.assertEqual(relationship["target"], route["source"])
                self.assertTrue({route["source"], route["target"], relationship["source"], relationship["target"]} <= entities | {route["source"], route["target"]})

            for artifact_name in api_module.PHASE8_ARTIFACT_NAMES:
                artifact_path = Path(jobs["job-SHEET-A"]["job_dir"]) / artifact_name
                original_artifact = artifact_path.read_text(encoding="utf-8")
                tampered_artifact = json.loads(original_artifact)
                tampered_artifact["source_graph_content_sha256"] = "0" * 64
                artifact_path.write_text(json.dumps(tampered_artifact), encoding="utf-8")
                with self.subTest(tampered_artifact=artifact_name):
                    with patch("api.PIPELINE_SYSTEMS_DIR", str(systems_root)), patch("api.PIPELINE_JOBS_DIR", str(jobs_root)), patch.dict(
                        "api.PIPELINE_JOBS", jobs, clear=False
                    ):
                        with self.assertRaises(HTTPException) as raised:
                            api_module._regenerate_pipeline_system_graph(system_id)
                    self.assertEqual(raised.exception.status_code, 409)
                artifact_path.write_text(original_artifact, encoding="utf-8")

            summary_path = Path(jobs["job-SHEET-A"]["job_dir"]) / "stage10_engineering_view_summary.json"
            mismatched_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            mismatched_summary["graph_content_sha256"] = "0" * 64
            summary_path.write_text(json.dumps(mismatched_summary), encoding="utf-8")
            with patch("api.PIPELINE_SYSTEMS_DIR", str(systems_root)), patch("api.PIPELINE_JOBS_DIR", str(jobs_root)), patch.dict(
                "api.PIPELINE_JOBS", jobs, clear=False
            ):
                with self.assertRaises(HTTPException) as raised:
                    api_module._regenerate_pipeline_system_graph(system_id)
                response = client.put(
                    f"/api/pipeline/systems/{system_id}/connector-review",
                    json={"connector_overrides": [], "manual_pairs": [], "reviewer": "mismatch-test"},
                )
            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("do not match", raised.exception.detail)
            self.assertEqual(response.status_code, 409, response.text)

    def test_pipeline_system_phase8_bundle_compatibility(self) -> None:
        api_module = __import__("api")

        def build_fixture(root: Path, state: str) -> tuple[Path, dict[str, dict]]:
            systems_root = root / "systems"
            jobs_root = root / "jobs"
            system_id = f"phase8-{state}"
            pages = []
            jobs = {}
            for index, sheet_id in enumerate(("SHEET-A", "SHEET-B")):
                job_id = f"job-{state}-{index}"
                job_dir = jobs_root / job_id
                job_dir.mkdir(parents=True)
                (job_dir / "stage7b_graph_v1.json").write_text(
                    json.dumps({"schema_version": "graph_v1", "document": {"doc_id": sheet_id}, "nodes": [], "edges": []}),
                    encoding="utf-8",
                )
                (job_dir / "stage_manifest.json").write_text(
                    json.dumps({"stages": [{"num": 11, "name": "stage11_connection_overlay", "status": "completed"}]}),
                    encoding="utf-8",
                )
                has_bundle = state == "mixed" and index == 1
                if state == "incomplete" and index == 1:
                    has_bundle = True
                artifact_names = list(api_module.PHASE8_ARTIFACT_NAMES)
                if state == "incomplete" and index == 0:
                    artifact_names = artifact_names[:2]
                if has_bundle:
                    for artifact_name in artifact_names:
                        payload = {"release_ready": True, "candidates": []}
                        if artifact_name == "stage10_llm_projections.json":
                            payload = {"schema_version": "llm_projections_v1"}
                        (job_dir / artifact_name).write_text(json.dumps(payload), encoding="utf-8")
                pages.append({"sheet_id": sheet_id, "source_filename": f"{sheet_id}.png", "job_id": job_id})
                jobs[job_id] = {
                    "job_id": job_id,
                    "status": "completed",
                    "current_stage": "stage11_connection_overlay",
                    "job_dir": str(job_dir),
                    "created_at": time.time(),
                    "stop_after": 11,
                    "system_id": system_id,
                    "document_id": sheet_id,
                }
            system_dir = systems_root / system_id
            system_dir.mkdir(parents=True)
            (system_dir / "system_manifest.json").write_text(
                json.dumps({
                    "manifest_version": 1,
                    "system_id": system_id,
                    "pages": pages,
                    "connector_review": {"revision": 1, "connector_overrides": [], "manual_pairs": []},
                    "connector_review_audit": [],
                }),
                encoding="utf-8",
            )
            return systems_root, jobs

        for state in ("legacy", "mixed", "incomplete"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                systems_root, jobs = build_fixture(Path(tmp), state)
                system_id = f"phase8-{state}"
                with patch("api.PIPELINE_SYSTEMS_DIR", str(systems_root)), patch(
                    "api.PIPELINE_JOBS_DIR", str(Path(tmp) / "jobs")
                ), patch.dict("api.PIPELINE_JOBS", jobs, clear=False):
                    if state == "legacy":
                        graph = api_module._regenerate_pipeline_system_graph(system_id)
                        self.assertEqual(graph["release_gate"]["phase8_views_status"], "legacy_unavailable")
                        self.assertNotIn("phase8_views", graph)
                    else:
                        with self.assertRaises(HTTPException) as raised:
                            api_module._regenerate_pipeline_system_graph(system_id)
                        self.assertEqual(raised.exception.status_code, 409)

    def test_phase8_qualification_is_collision_safe_and_marks_ambiguous_relationships(self) -> None:
        api_module = __import__("api")
        graph = {
            "nodes": [{"id": "shared", "type": "equipment"}],
            "edges": [{"id": "shared", "source": "shared", "target": "terminal"}],
        }
        llm = {
            "process_description": {"graph": {
                "entities": [
                    {"id": "shared", "type": "equipment"},
                    {"id": "terminal", "type": "topology"},
                    {"id": "shared", "type": "route"},
                ],
                "routes": [{"id": "shared", "source": "shared", "target": "terminal",
                            "flow": {"source": "shared", "target": "terminal"}}],
                "relationships": [{"id": "ambiguous", "source": "shared", "target": "shared"}],
            }},
        }
        id_kinds = api_module._phase8_entity_kind_map(graph, llm)
        qualified = api_module._qualify_phase8_record(llm, sheet_id="SHEET-X", id_kinds=id_kinds)
        projection_graph = qualified["process_description"]["graph"]
        entities = {item["id"] for item in projection_graph["entities"]}
        route = projection_graph["routes"][0]
        relationship = projection_graph["relationships"][0]
        self.assertIn("equipment::SHEET-X::shared", entities)
        self.assertIn("edge::SHEET-X::shared", entities)
        self.assertEqual(route["id"], "edge::SHEET-X::shared")
        self.assertEqual(route["source"], "equipment::SHEET-X::shared")
        self.assertEqual(route["flow"]["source"], route["source"])
        self.assertEqual(relationship["source"], "shared")
        self.assertEqual(relationship["target"], "shared")
        self.assertEqual(relationship["reference_uncertainty"][0]["reason"], "ambiguous_id_kind")

    def test_pipeline_merge_reads_current_stage7b_artifact(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            jobs = {}
            job_ids = []
            for sheet_id, target, edge_id in (
                ("DWG-100", "DWG-200", "edge-a"),
                ("DWG-200", "DWG-100", "edge-b"),
            ):
                job_id = f"merge-{sheet_id}"
                job_ids.append(job_id)
                job_dir = Path(tmp) / job_id
                job_dir.mkdir()
                (job_dir / "stage7b_graph_v1.json").write_text(
                    json.dumps({
                        "schema_version": "graph_v1",
                        "document": {"doc_id": sheet_id},
                        "nodes": [{"id": f"equipment::{sheet_id}", "type": "equipment"}],
                        "edges": [{"id": f"route-{sheet_id}", "source": f"equipment::{sheet_id}", "target": f"equipment::{sheet_id}", "flow_direction_state": "unknown",
                                   "off_page_connector": {
                                       "local_edge_id": edge_id,
                                       "reference_type": "drawing",
                                       "reference_value": target,
                                       "target_sheet_reference": target,
                                       "connector_key": "10-P-100-A",
                                       "direction": "bidirectional",
                                       "exit_terminal": "source",
                                   }}],
                    }),
                    encoding="utf-8",
                )
                jobs[job_id] = {
                    "job_id": job_id,
                    "status": "completed",
                    "current_stage": "stage11_connection_overlay",
                    "job_dir": str(job_dir),
                    "created_at": time.time(),
                    "stop_after": 11,
                }
            with patch.dict("api.PIPELINE_JOBS", jobs, clear=False):
                response = client.post("/api/pipeline/merge", json={"job_ids": job_ids})

            self.assertEqual(response.status_code, 200, response.text)
            merged = response.json()
            self.assertEqual(len(merged["cross_sheet_edges"]), 1)
            combined = merged["combined_graph"]
            self.assertEqual({node["id"] for node in combined["nodes"]}, {"node::DWG-100::equipment::DWG-100", "node::DWG-200::equipment::DWG-200"})
            self.assertEqual(len(combined["connectors"]), 2)
            self.assertEqual(len(combined["relationships"]), 1)
            self.assertTrue(all("::" in relation["source"] for relation in combined["relationships"]))

    def test_pipeline_merge_rejects_duplicate_normalized_document_ids(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            jobs = {}
            for index, doc_id in enumerate(("Sheet A", " sheet   a ")):
                job_id = f"dup-{index}"
                job_dir = Path(tmp) / job_id
                job_dir.mkdir()
                (job_dir / "stage7b_graph_v1.json").write_text(json.dumps({"schema_version": "graph_v1", "document": {"doc_id": doc_id}, "nodes": [], "edges": []}), encoding="utf-8")
                jobs[job_id] = {"job_id": job_id, "status": "completed", "current_stage": "stage11_connection_overlay", "job_dir": str(job_dir), "created_at": time.time(), "stop_after": 11}
            with patch.dict("api.PIPELINE_JOBS", jobs, clear=False):
                response = client.post("/api/pipeline/merge", json={"job_ids": list(jobs)})
            self.assertEqual(response.status_code, 400)
            self.assertIn("Duplicate normalized document.doc_id", response.json()["detail"])

    def test_system_graph_writer_preserves_additive_merge_fields_and_rejects_nan(self) -> None:
        api_module = __import__("api")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "system_graph_v2.json"
            additive = {
                "schema_version": "graph_v2",
                "cross_sheet_edges": [{"id": "x", "line_number_id": "L-1", "route_direction_state": "unknown"}],
                "canonical_lines": [{"canonical_line_id": "line::L-1", "sheet_ids": ["A", "B"]}],
                "connector_relations": [{"type": "same_physical_connector", "source": "A::c", "target": "B::c"}],
            }
            api_module._write_json_atomic(str(path), additive)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), additive)
            with self.assertRaises(ValueError):
                api_module._write_json_atomic(str(path), {"bad": float("nan")})

    def test_release_gate_controls_public_graph_with_legacy_compatibility(self) -> None:
        api_module = __import__("api")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"stages": [{"name": "stage9_apply_review_decisions", "status": "completed"}]}
            (root / "stage_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(api_module._pipeline_graph_is_fresh(str(root)))
            manifest["stages"][0]["artifacts"] = ["stage9_corrected_graph.json", "stage9_release_gate.json"]
            (root / "stage_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "stage9_release_gate.json").write_text(json.dumps({"release_ready": False}), encoding="utf-8")
            self.assertFalse(api_module._pipeline_graph_is_fresh(str(root)))
            (root / "stage9_release_gate.json").write_text(json.dumps({"release_ready": True}), encoding="utf-8")
            self.assertTrue(api_module._pipeline_graph_is_fresh(str(root)))
            (root / "stage9_release_gate.json").write_text("{corrupt", encoding="utf-8")
            self.assertFalse(api_module._pipeline_graph_is_fresh(str(root)))

    def test_current_completed_stage9_with_deleted_gate_is_stale(self) -> None:
        api_module = __import__("api")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stage_manifest.json").write_text(json.dumps({"stages": [{
                "name": "stage9_apply_review_decisions",
                "status": "completed",
                "artifacts": ["stage9_corrected_graph.json", "stage9_release_gate.json"],
            }]}), encoding="utf-8")
            self.assertFalse(api_module._pipeline_graph_is_fresh(str(root)))

    def test_phase8_artifacts_follow_single_sheet_release_gate(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_id = "phase8-gate-job"
            phase8_names = [
                "stage10_process_boundaries.json",
                "stage10_test_package_candidates.json",
                "stage10_engineering_view_summary.json",
                "stage10_llm_projections.json",
            ]
            for name in phase8_names:
                (root / name).write_text("{}", encoding="utf-8")
            (root / "stage_manifest.json").write_text(json.dumps({"stages": [{
                "name": "stage9_apply_review_decisions",
                "status": "completed",
                "artifacts": ["stage9_release_gate.json", *phase8_names],
            }]}), encoding="utf-8")
            (root / "stage9_release_gate.json").write_text(json.dumps({"release_ready": False}), encoding="utf-8")
            job = {"job_id": job_id, "job_dir": str(root), "status": "completed"}
            with patch.dict("api.PIPELINE_JOBS", {job_id: job}, clear=False):
                blocked = [
                    client.get(f"/api/pipeline/jobs/{job_id}/artifacts/{name}")
                    for name in phase8_names
                ]
                self.assertTrue(all(response.status_code == 409 for response in blocked))
                (root / "stage9_release_gate.json").write_text(json.dumps({"release_ready": True}), encoding="utf-8")
                released = [
                    client.get(f"/api/pipeline/jobs/{job_id}/artifacts/{name}")
                    for name in phase8_names
                ]
            self.assertTrue(all(response.status_code == 200 for response in released))

    def test_phase8_derived_artifacts_cannot_be_replaced_via_put(self) -> None:
        api_module = __import__("api")
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_id = "phase8-derived-write-job"
            phase8_names = list(api_module.PHASE8_ARTIFACT_NAMES)
            originals = {}
            for index, name in enumerate(phase8_names):
                payload = {"release_ready": True, "marker": index}
                originals[name] = json.dumps(payload, sort_keys=True)
                (root / name).write_text(originals[name], encoding="utf-8")
            (root / "stage_manifest.json").write_text(json.dumps({"stages": [{
                "name": "stage9_apply_review_decisions",
                "status": "completed",
                "artifacts": ["stage9_release_gate.json", *phase8_names],
            }]}), encoding="utf-8")
            (root / "stage9_release_gate.json").write_text(json.dumps({"release_ready": True}), encoding="utf-8")
            job = {"job_id": job_id, "job_dir": str(root), "status": "completed"}
            with patch.dict("api.PIPELINE_JOBS", {job_id: job}, clear=False):
                for name in phase8_names:
                    with self.subTest(artifact=name):
                        response = client.put(
                            f"/api/pipeline/jobs/{job_id}/artifacts/{name}",
                            json={"tampered": True},
                        )
                        self.assertEqual(response.status_code, 409, response.text)
                        self.assertEqual((root / name).read_text(encoding="utf-8"), originals[name])
                        served = client.get(f"/api/pipeline/jobs/{job_id}/artifacts/{name}")
                        self.assertEqual(served.status_code, 200, served.text)
                        self.assertEqual(served.json(), json.loads(originals[name]))

    def test_artifact_put_rejects_non_finite_json_without_changing_input(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_id = "finite-artifact-job"
            input_name = "stage8_review_decisions.json"
            original = '{"decisions": []}'
            (root / input_name).write_text(original, encoding="utf-8")
            (root / "stage_manifest.json").write_text(json.dumps({"stages": []}), encoding="utf-8")
            job = {"job_id": job_id, "job_dir": str(root), "status": "completed"}
            with patch.dict("api.PIPELINE_JOBS", {job_id: job}, clear=False):
                response = client.put(
                    f"/api/pipeline/jobs/{job_id}/artifacts/{input_name}",
                    content=b'{"value": NaN}',
                    headers={"content-type": "application/json"},
                )
            self.assertEqual(response.status_code, 400, response.text)
            self.assertEqual((root / input_name).read_text(encoding="utf-8"), original)

    def test_system_graph_release_gate_blocks_pending_and_issue_states(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp, patch("api.PIPELINE_SYSTEMS_DIR", tmp):
            system_id = "gate-system"
            system_dir = Path(tmp) / system_id
            system_dir.mkdir()
            (system_dir / "system_graph_v2.json").write_text(json.dumps({"schema_version": "graph_v2"}), encoding="utf-8")
            base = {"manifest_version": 1, "system_id": system_id, "pages": [], "merge": {"graph_artifact": "system_graph_v2.json"}}
            for merge, expected in (({"status": "awaiting_connector_review"}, 409), ({"status": "completed", "release_ready": False}, 409), ({"status": "completed", "release_ready": True}, 200)):
                (system_dir / "system_manifest.json").write_text(json.dumps({**base, "merge": {**base["merge"], **merge}}), encoding="utf-8")
                response = client.get(f"/api/pipeline/systems/{system_id}/graph")
                self.assertEqual(response.status_code, expected)
    def test_ambiguous_numeric_resume_stages_are_rejected(self) -> None:
        expected_names = {
            "4": "stage4_object_detection",
            "5": "stage5_pipe_mask",
            "7": "stage7_geometric_graph_assembly",
        }
        for stage, expected_name in expected_names.items():
            with self.subTest(stage=stage):
                with self.assertRaises(HTTPException) as caught:
                    _resolve_pipeline_stage(stage)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertIn(expected_name, caught.exception.detail)

    def test_named_and_unique_numeric_resume_stages_are_valid(self) -> None:
        self.assertEqual(_resolve_pipeline_stage("6"), (6, "stage6_trace_associations"))
        self.assertEqual(_resolve_pipeline_stage("stage5b_pipe_trace"), (5, "stage5b_pipe_trace"))

    def test_pipeline_stage_status_reports_stale_after_stage4_objects_update(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "stage_state_job_stage4_objects"
            stale_artifact = Path(tmp) / "stage5_pipe_mask.png"
            stale_artifact.write_bytes(b"old")
            manifest = {
                "stages": [
                    {"num": 1, "name": "stage1_input_normalization", "status": "completed"},
                    {"num": 2, "name": "stage2_ocr_discovery", "status": "completed"},
                    {"num": 4, "name": "stage4_object_detection", "status": "completed"},
                    {"num": 4, "name": "stage4_line_number_fusion", "status": "completed"},
                    {"num": 4, "name": "stage4_instrument_tag_fusion", "status": "completed"},
                    {"num": 5, "name": "stage5_pipe_mask", "status": "completed"},
                    {"num": 5, "name": "stage5b_pipe_trace", "status": "completed"},
                    {"num": 6, "name": "stage6_trace_associations", "status": "completed"},
                ]
            }
            with open(Path(tmp) / "stage_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage6_trace_associations",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 6,
                "ocr_route": "ocrmac",
                "gemini_postprocess_match_threshold": 0.1,
                "weight_file": "yolo_weights/model.pt",
            }}, clear=False):
                response = client.put(
                    f"/api/pipeline/jobs/{job_id}/artifacts/stage4_objects.json",
                    json={"objects": [{"id": "obj_001", "class_name": "gate_valve", "bbox": {"x_min": 1, "y_min": 2, "x_max": 10, "y_max": 20}}]},
                )
                self.assertEqual(response.status_code, 200)

                status_response = client.get(f"/api/pipeline/jobs/{job_id}/stage-status")

            self.assertEqual(status_response.status_code, 200)
            stages = {item["name"]: item for item in status_response.json()["stages"]}
            self.assertEqual(stages["stage4_object_detection"]["status"], "completed")
            self.assertEqual(stages["stage4_line_number_fusion"]["status"], "stale")
            self.assertEqual(stages["stage4_instrument_tag_fusion"]["status"], "stale")
            self.assertEqual(stages["stage5_pipe_mask"]["status"], "stale")
            self.assertEqual(stages["stage5b_pipe_trace"]["status"], "stale")
            self.assertEqual(stages["stage6_trace_associations"]["status"], "stale")
            self.assertTrue((Path(tmp) / "stage4_objects.json").exists())
            self.assertFalse(stale_artifact.exists())

    def test_pipeline_stage_status_reports_stale_after_stage3_artifact_update(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "stage_state_job_1"
            manifest = {
                "stages": [
                    {"num": 1, "name": "stage1_input_normalization", "status": "completed"},
                    {"num": 2, "name": "stage2_ocr_discovery", "status": "completed"},
                    {"num": 4, "name": "stage4_object_detection", "status": "completed"},
                    {"num": 5, "name": "stage5_pipe_mask", "status": "completed"},
                    {"num": 5, "name": "stage5b_pipe_trace", "status": "completed"},
                    {"num": 6, "name": "stage6_trace_associations", "status": "completed"},
                ]
            }
            with open(Path(tmp) / "stage_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage6_trace_associations",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 6,
                "ocr_route": "ocrmac",
                "gemini_postprocess_match_threshold": 0.1,
                "weight_file": "yolo_weights/model.pt",
            }}, clear=False):
                response = client.put(
                    f"/api/pipeline/jobs/{job_id}/artifacts/stage3_equipment_bboxes.json",
                    json={"equipment": [{"id": "equip_001", "class_name": "vessel", "bbox": {"x_min": 1, "y_min": 2, "x_max": 10, "y_max": 20}}]},
                )
                self.assertEqual(response.status_code, 200)

                status_response = client.get(f"/api/pipeline/jobs/{job_id}/stage-status")

            self.assertEqual(status_response.status_code, 200)
            stages = {item["name"]: item for item in status_response.json()["stages"]}
            self.assertEqual(stages["stage5b_pipe_trace"]["status"], "stale")
            self.assertEqual(stages["stage6_trace_associations"]["status"], "stale")
            self.assertEqual(stages["stage4_object_detection"]["status"], "completed")
            self.assertTrue((Path(tmp) / "stage3_equipment_bboxes.json").exists())
            self.assertFalse((Path(tmp) / "stage5_connection_ports.json").exists())

    def test_pipeline_resume_from_stage_reruns_from_requested_stage(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            image_path.write_bytes(b"placeholder")
            manifest = {
                "image_path": str(image_path),
                "stages": [
                    {"num": 1, "name": "stage1_input_normalization", "status": "completed"},
                    {"num": 2, "name": "stage2_ocr_discovery", "status": "completed"},
                    {"num": 5, "name": "stage5_pipe_mask", "status": "completed"},
                    {"num": 5, "name": "stage5b_pipe_trace", "status": "stale"},
                ],
            }
            with open(Path(tmp) / "stage_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            run_calls: list[tuple[int, bool]] = []

            class FakeResumePipeline:
                def __init__(self, image_path: str, output_dir: str, stage_callback=None, cfg=None) -> None:
                    self.stage_manifest = {"stages": [{"name": "stage5b_pipe_trace"}]}

                def run(self, stop_after: int, resume: bool = False) -> None:
                    run_calls.append((stop_after, resume))

            job_id = "stage_state_job_2"
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage5b_pipe_trace",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 5,
                "ocr_route": "ocrmac",
                "gemini_postprocess_match_threshold": 0.1,
                "weight_file": "yolo_weights/model.pt",
            }}, clear=False), patch("api.PIDPipeline", FakeResumePipeline):
                response = client.post(f"/api/pipeline/jobs/{job_id}/resume-from/stage5b_pipe_trace")
                self.assertEqual(response.status_code, 200)
                deadline = time.time() + 5
                while time.time() < deadline and not run_calls:
                    time.sleep(0.05)

            self.assertEqual(run_calls, [(PIPELINE_LAST_STAGE, True)])

    def test_pipeline_resume_from_stage_honors_explicit_stop_after(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            image_path.write_bytes(b"placeholder")
            manifest = {
                "image_path": str(image_path),
                "stages": [
                    {"num": 1, "name": "stage1_input_normalization", "status": "completed"},
                    {"num": 2, "name": "stage2_ocr_discovery", "status": "completed"},
                    {"num": 5, "name": "stage5_pipe_mask", "status": "completed"},
                    {"num": 5, "name": "stage5b_pipe_trace", "status": "stale"},
                ],
            }
            with open(Path(tmp) / "stage_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            run_calls: list[tuple[int, bool]] = []

            class FakeResumePipeline:
                def __init__(self, image_path: str, output_dir: str, stage_callback=None, cfg=None) -> None:
                    self.stage_manifest = {"stages": [{"name": "stage5b_pipe_trace"}]}

                def run(self, stop_after: int, resume: bool = False) -> None:
                    run_calls.append((stop_after, resume))

            job_id = "stage_state_job_explicit_stop_after"
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage5b_pipe_trace",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 5,
                "ocr_route": "ocrmac",
                "gemini_postprocess_match_threshold": 0.1,
                "weight_file": "yolo_weights/model.pt",
            }}, clear=False), patch("api.PIDPipeline", FakeResumePipeline):
                response = client.post(f"/api/pipeline/jobs/{job_id}/resume-from/stage5b_pipe_trace?stop_after=5")
                self.assertEqual(response.status_code, 200)
                deadline = time.time() + 5
                while time.time() < deadline and not run_calls:
                    time.sleep(0.05)

            self.assertEqual(run_calls, [(5, True)])

    def test_pipeline_job_passes_debug_artifacts_to_pipeline_config(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"
        captured_debug_artifacts: list[bool] = []

        class FakeDebugPipeline:
            def __init__(self, image_path: str, output_dir: str, stage_callback=None, cfg=None) -> None:
                captured_debug_artifacts.append(bool(getattr(cfg, "debug_artifacts", False)))
                self.stage_manifest = {"stages": [{"name": "stage1_input_normalization"}]}

            def run(self, stop_after: int, resume: bool = False) -> None:
                return None

        with patch("api.PIDPipeline", FakeDebugPipeline):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={
                        "stop_after": "1",
                        "ocr_route": "ocrmac",
                        "debug_artifacts": "true",
                    },
                )
            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]
            deadline = time.time() + 5
            while time.time() < deadline and not captured_debug_artifacts:
                time.sleep(0.05)
            poll = client.get(f"/api/pipeline/jobs/{job_id}")

        self.assertEqual(poll.status_code, 200)
        self.assertEqual(captured_debug_artifacts, [True])
        self.assertTrue(poll.json()["debug_artifacts"])

    def test_pipeline_review_state_get_returns_empty_default(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "review_job_1"
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage13_graph_qa",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 13,
                "ocr_route": "ocrmac",
            }}, clear=False):
                response = client.get(f"/api/pipeline/jobs/{job_id}/review-state")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["job_id"], Path(tmp).name)
            self.assertEqual(payload["items"], [])

    def test_pipeline_review_workspace_get_initializes_from_artifacts(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "review_workspace_job_1"
            (Path(tmp) / "stage4_objects.json").write_text(
                json.dumps(
                    {
                        "image_id": "sample.png",
                        "objects": [
                            {
                                "id": "obj_001",
                                "class_name": "gate_valve",
                                "bbox": {"x_min": 1, "y_min": 2, "x_max": 10, "y_max": 20},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage6_trace_associations",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 6,
                "ocr_route": "ocrmac",
            }}, clear=False):
                response = client.get(f"/api/pipeline/jobs/{job_id}/review-workspace")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["job_id"], job_id)
            self.assertEqual(payload["artifact"]["name"], "review_workspace_state.json")
            self.assertEqual(payload["workspace"]["objects"][0]["id"], "obj_001")

    def test_pipeline_review_workspace_put_persists_payload(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "review_workspace_job_2"
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage6_trace_associations",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 6,
                "ocr_route": "ocrmac",
            }}, clear=False):
                response = client.put(
                    f"/api/pipeline/jobs/{job_id}/review-workspace",
                    json={"objects": [{"id": "obj_002", "class_name": "pump"}]},
                )
                get_response = client.get(f"/api/pipeline/jobs/{job_id}/review-workspace")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(get_response.status_code, 200)
            self.assertEqual(get_response.json()["workspace"]["objects"][0]["id"], "obj_002")
            self.assertTrue((Path(tmp) / "review_workspace_state.json").exists())

    def test_pipeline_review_workspace_recompute_writes_reviewed_inputs_and_layers(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            image_path.write_bytes(b"placeholder")
            with open(Path(tmp) / "stage_manifest.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "image_path": str(image_path),
                        "stages": [
                            {"num": 1, "name": "stage1_input_normalization", "status": "completed"},
                            {"num": 2, "name": "stage2_ocr_discovery", "status": "completed"},
                            {"num": 4, "name": "stage4_object_detection", "status": "completed"},
                            {"num": 5, "name": "stage5_pipe_mask", "status": "completed"},
                            {"num": 5, "name": "stage5b_pipe_trace", "status": "completed"},
                            {"num": 6, "name": "stage6_trace_associations", "status": "completed"},
                        ],
                    },
                    f,
                )

            class FakeRecomputePipeline:
                stale_ports_were_removed = False

                def __init__(self, image_path: str, output_dir: str, stage_callback=None, cfg=None) -> None:
                    self.output_dir = Path(output_dir)
                    self.stage_manifest = {
                        "stages": [
                            {"num": 5, "name": "stage5_pipe_mask", "status": "completed"},
                            {"num": 5, "name": "stage5b_pipe_trace", "status": "completed"},
                            {"num": 6, "name": "stage6_trace_associations", "status": "completed"},
                        ]
                    }

                def run(self, stop_after: int, resume: bool = False) -> None:
                    FakeRecomputePipeline.stale_ports_were_removed = not self.output_dir.joinpath("stage5_connection_ports.json").exists()
                    self.output_dir.joinpath("stage5_connection_ports.json").write_text(json.dumps({"ports": [{"id": "p01"}]}), encoding="utf-8")
                    self.output_dir.joinpath("stage5b_trace_results.json").write_text(json.dumps({"results": {"equip_001": {}}}), encoding="utf-8")
                    self.output_dir.joinpath("stage5b_branch_trace_results.json").write_text(json.dumps({"branches": {}}), encoding="utf-8")
                    self.output_dir.joinpath("stage6_trace_associations.json").write_text(json.dumps({"trace_edges": [{"trace_id": "equip_001"}]}), encoding="utf-8")
                    self.output_dir.joinpath("stage6_line_number_review.json").write_text(json.dumps({"accepted": []}), encoding="utf-8")

            job_id = "review_workspace_job_3"
            (Path(tmp) / "stage5_connection_ports.json").write_text(json.dumps({"old_obj": [[1, 2, "RIGHT"]]}), encoding="utf-8")
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage6_trace_associations",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 6,
                "ocr_route": "ocrmac",
                "gemini_postprocess_match_threshold": 0.1,
                "weight_file": "yolo_weights/model.pt",
            }}, clear=False), patch("api.PIDPipeline", FakeRecomputePipeline):
                response = client.post(
                    f"/api/pipeline/jobs/{job_id}/review-workspace/recompute",
                    json={
                        "scope": "stage5_to_6",
                        "workspace": {
                            "image_id": "sample.png",
                            "objects": [
                                {
                                    "id": "obj_001",
                                    "class_name": "gate_valve",
                                    "bbox": {"x_min": 1, "y_min": 2, "x_max": 10, "y_max": 20},
                                }
                            ],
                            "equipment": [
                                {
                                    "id": "equip_001",
                                    "class_name": "vessel",
                                    "bbox": {"x_min": 30, "y_min": 40, "x_max": 130, "y_max": 240},
                                }
                            ],
                        },
                    },
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["job_id"], job_id)
            self.assertIn("stage5_connection_ports", payload["layers"])
            self.assertEqual(payload["layers"]["stage6_trace_associations"]["trace_edges"][0]["trace_id"], "equip_001")
            self.assertTrue(FakeRecomputePipeline.stale_ports_were_removed)
            self.assertTrue((Path(tmp) / "review_workspace_state.json").exists())
            self.assertTrue((Path(tmp) / "stage3_equipment_bboxes.json").exists())
            self.assertTrue((Path(tmp) / "stage4_objects.json").exists())

    def test_pipeline_review_workspace_commit_marks_downstream_stale(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "review_workspace_job_commit"
            stale_graph = Path(tmp) / "stage7_graph.json"
            stale_graph.write_text("{}", encoding="utf-8")
            with open(Path(tmp) / "stage_manifest.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "image_path": str(Path(tmp) / "input.png"),
                        "stages": [
                            {"num": 6, "name": "stage6_trace_associations", "status": "completed"},
                            {"num": 7, "name": "stage7_geometric_graph_assembly", "status": "completed"},
                            {"num": 10, "name": "stage10_process_exports", "status": "completed"},
                        ],
                    },
                    f,
                )

            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage10_process_exports",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 10,
                "ocr_route": "ocrmac",
            }}, clear=False):
                response = client.post(
                    f"/api/pipeline/jobs/{job_id}/review-workspace/commit",
                    json={
                        "workspace": {
                            "image_id": "sample.png",
                            "objects": [
                                {
                                    "id": "obj_001",
                                    "class_name": "gate_valve",
                                    "bbox": {"x_min": 1, "y_min": 2, "x_max": 10, "y_max": 20},
                                }
                            ],
                            "equipment": [],
                        }
                    },
                )

            self.assertEqual(response.status_code, 200)
            stages = {item["name"]: item for item in response.json()["stages"]}
            self.assertEqual(stages["stage6_trace_associations"]["status"], "completed")
            self.assertEqual(stages["stage7_geometric_graph_assembly"]["status"], "stale")
            self.assertEqual(stages["stage10_process_exports"]["status"], "stale")
            self.assertFalse(stale_graph.exists())
            self.assertTrue((Path(tmp) / "stage4_objects.json").exists())

    def test_pipeline_review_state_put_persists_payload(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "review_job_2"
            with open(Path(tmp) / "stage_manifest.json", "w", encoding="utf-8") as f:
                json.dump({"image_path": "sample.png"}, f)
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage13_graph_qa",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 13,
                "ocr_route": "ocrmac",
            }}, clear=False):
                response = client.put(
                    f"/api/pipeline/jobs/{job_id}/review-state",
                    json={
                        "items": [
                            {
                                "item_id": "stage4_line_number:line_number_000001",
                                "bucket": "stage4_line_number",
                                "source_stage": "stage4_line_number_fusion",
                                "source_artifact": "stage4_line_numbers.json",
                                "entity_id": "line_number_000001",
                                "decision": "accepted",
                            }
                        ],
                        "workspace_objects": {"stage4_line_number": [{"Object": "line_number"}]},
                    },
                )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(len(payload["items"]), 1)
            self.assertEqual(payload["items"][0]["decision"], "accepted")
            self.assertTrue((Path(tmp) / "stage_review_state.json").exists())

    def test_pipeline_review_state_put_materializes_stage4_line_number_artifact(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "review_job_line_numbers"
            job_dir = Path(tmp)
            (job_dir / "stage_manifest.json").write_text(
                json.dumps(
                    {
                        "image_path": "sample.png",
                        "stages": [
                            {"num": 4, "name": "stage4_line_number_fusion", "status": "completed"},
                            {"num": 6, "name": "stage6_trace_associations", "status": "completed"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "stage4_line_numbers.json").write_text(
                json.dumps({"image_id": "sample.png", "line_numbers": [{"id": "old_line"}], "rejected": []}),
                encoding="utf-8",
            )
            (job_dir / "stage6_trace_associations.json").write_text(json.dumps({"trace_edges": []}), encoding="utf-8")

            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage6_trace_associations",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 6,
                "ocr_route": "ocrmac",
            }}, clear=False):
                response = client.put(
                    f"/api/pipeline/jobs/{job_id}/review-state",
                    json={
                        "items": [],
                        "workspace_objects": {
                            "stage4_line_number": [
                                {
                                    "Object": "line_number",
                                    "SourceItemId": "line_number_000123",
                                    "Text": "3-CUL-25-001",
                                    "Left": 10,
                                    "Top": 20,
                                    "Width": 100,
                                    "Height": 12,
                                    "Score": 0.91,
                                    "ReviewStatus": "accepted",
                                },
                                {
                                    "Object": "line_number",
                                    "SourceItemId": "line_number_000124",
                                    "Text": "bad",
                                    "Left": 30,
                                    "Top": 40,
                                    "Width": 50,
                                    "Height": 10,
                                    "Score": 0.5,
                                    "ReviewStatus": "rejected",
                                },
                            ]
                        },
                    },
                )

            self.assertEqual(response.status_code, 200)
            materialized = json.loads((job_dir / "stage4_line_numbers.json").read_text(encoding="utf-8"))
            self.assertEqual([item["id"] for item in materialized["line_numbers"]], ["line_number_000123"])
            self.assertEqual(materialized["line_numbers"][0]["normalized_text"], "3-CUL-25-001")
            self.assertEqual([item["id"] for item in materialized["rejected"]], ["line_number_000124"])
            self.assertFalse((job_dir / "stage6_trace_associations.json").exists())
            stages = {item["name"]: item for item in response.json()["stages"]}
            self.assertEqual(stages["stage6_trace_associations"]["status"], "stale")

    def test_pipeline_review_state_put_recovers_disk_job_after_reload(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "review_job_line_numbers_recovered"
            jobs_root = Path(tmp) / "pipeline_jobs"
            job_dir = jobs_root / job_id
            job_dir.mkdir(parents=True)
            (job_dir / "input.png").write_bytes(b"image")
            (job_dir / "stage_manifest.json").write_text(
                json.dumps(
                    {
                        "image_path": str(job_dir / "input.png"),
                        "stages": [
                            {"num": 4, "name": "stage4_line_number_fusion", "status": "completed"},
                            {"num": 6, "name": "stage6_trace_associations", "status": "completed"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "stage4_line_numbers.json").write_text(
                json.dumps({"image_id": "input.png", "line_numbers": [], "rejected": []}),
                encoding="utf-8",
            )

            with patch("api.PIPELINE_JOBS_DIR", str(jobs_root)), patch.dict("api.PIPELINE_JOBS", {}, clear=True):
                response = client.put(
                    f"/api/pipeline/jobs/{job_id}/review-state",
                    json={
                        "items": [],
                        "workspace_objects": {
                            "stage4_line_number": [
                                {
                                    "Object": "line_number",
                                    "SourceItemId": "line_number_000125",
                                    "Text": "4-F-25-001",
                                    "Left": 11,
                                    "Top": 22,
                                    "Width": 120,
                                    "Height": 14,
                                    "ReviewStatus": "accepted",
                                }
                            ]
                        },
                    },
                )

            self.assertEqual(response.status_code, 200)
            materialized = json.loads((job_dir / "stage4_line_numbers.json").read_text(encoding="utf-8"))
            self.assertEqual([item["id"] for item in materialized["line_numbers"]], ["line_number_000125"])
            self.assertEqual(materialized["line_numbers"][0]["bbox"], {"x_min": 11, "y_min": 22, "x_max": 131, "y_max": 36})

    def test_pipeline_review_state_put_rejects_invalid_bucket(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "review_job_3"
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage13_graph_qa",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 13,
                "ocr_route": "ocrmac",
            }}, clear=False):
                response = client.put(
                    f"/api/pipeline/jobs/{job_id}/review-state",
                    json={
                        "items": [
                            {
                                "item_id": "bad",
                                "bucket": "bad_bucket",
                                "decision": "accepted",
                            }
                        ],
                        "workspace_objects": {},
                    },
                )
            self.assertEqual(response.status_code, 400)

    def test_pipeline_job_runs_stage2_and_reports_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "text_regions": [
                    {
                        "id": "ocr_000001",
                        "text": "P-1001",
                        "normalized_text": "P-1001",
                        "class": "line_number",
                        "confidence": 0.91,
                        "bbox": {"x_min": 10, "y_min": 20, "x_max": 90, "y_max": 40},
                        "rotation": 0,
                        "reading_direction": "horizontal",
                        "legibility": "clear",
                    }
                ],
            },
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "tile_count": 1,
                "raw_detection_count": 1,
                "merged_region_count": 1,
                "exception_candidate_count": 0,
                "slice_height": 1600,
                "slice_width": 1600,
                "overlap_height_ratio": 0.2,
                "overlap_width_ratio": 0.2,
            },
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "2", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertIn("job_id", payload)
            job_id = payload["job_id"]

            deadline = time.time() + 10
            job_payload = None
            while time.time() < deadline:
                poll = client.get(f"/api/pipeline/jobs/{job_id}")
                self.assertEqual(poll.status_code, 200)
                job_payload = poll.json()
                if job_payload["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.1)

            self.assertIsNotNone(job_payload)
            assert job_payload is not None
            self.assertEqual(job_payload["status"], "completed")
            self.assertEqual(job_payload["current_stage"], "stage2_ocr_discovery")
            self.assertEqual(job_payload["ocr_route"], "easyocr")
            self.assertEqual(len(job_payload["manifest"]["stages"]), 2)
            self.assertEqual(job_payload["manifest"]["ocr_route"], "easyocr")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage1_gray.png", artifact_names)
            self.assertIn("stage1_normalization_summary.json", artifact_names)
            self.assertIn("stage2_ocr_regions.json", artifact_names)
            self.assertIn("stage2_ocr_summary.json", artifact_names)
            self.assertIn("stage2_ocr_exception_candidates.json", artifact_names)

    def test_pipeline_job_accepts_ocrmac_route(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "route": "ocrmac"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }

        with patch("garnet.pid_extractor.run_ocrmac_sahi", return_value=fake_ocr_result):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "2", "ocr_route": "ocrmac"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

            deadline = time.time() + 10
            job_payload = None
            while time.time() < deadline:
                poll = client.get(f"/api/pipeline/jobs/{job_id}")
                self.assertEqual(poll.status_code, 200)
                job_payload = poll.json()
                if job_payload["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.1)

            self.assertIsNotNone(job_payload)
            assert job_payload is not None
            self.assertEqual(job_payload["status"], "completed")
            self.assertEqual(job_payload["ocr_route"], "ocrmac")

    def test_pipeline_job_runs_stage4_and_reports_object_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "tile_count": 1,
                "raw_detection_count": 0,
                "merged_region_count": 0,
                "exception_candidate_count": 0,
                "slice_height": 1600,
                "slice_width": 1600,
                "overlap_height_ratio": 0.2,
                "overlap_width_ratio": 0.2,
            },
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "objects": [
                    {
                        "id": "obj_000001",
                        "class_name": "valve",
                        "confidence": 0.88,
                        "bbox": {"x_min": 5, "y_min": 6, "x_max": 20, "y_max": 30},
                        "source_model": "ultralytics",
                        "source_weight": "yolo_weights/yolo11n_PPCL_640_20250204.pt",
                    }
                ],
            },
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 1,
                "source_weight": "yolo_weights/yolo11n_PPCL_640_20250204.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_line_number_fusion_result = {
            "line_numbers_payload": {"line_numbers": [], "rejected": []},
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"matched_line_number_count": 0},
        }
        fake_instrument_tag_fusion_result = {
            "instrument_tags_payload": {"instrument_tags": [], "rejected": []},
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"matched_instrument_tag_count": 0},
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch(
            "garnet.pid_extractor.run_line_number_fusion_stage", return_value=fake_line_number_fusion_result
        ), patch(
            "garnet.pid_extractor.run_instrument_tag_fusion_stage", return_value=fake_instrument_tag_fusion_result
        ):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "4", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

            deadline = time.time() + 10
            job_payload = None
            while time.time() < deadline:
                poll = client.get(f"/api/pipeline/jobs/{job_id}")
                self.assertEqual(poll.status_code, 200)
                job_payload = poll.json()
                if job_payload["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.1)

            self.assertIsNotNone(job_payload)
            assert job_payload is not None
            self.assertEqual(job_payload["status"], "completed")
            self.assertEqual(job_payload["current_stage"], "stage4_instrument_tag_fusion")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage4_objects.json", artifact_names)
            self.assertIn("stage4_objects_summary.json", artifact_names)
            self.assertIn("stage4_objects_overlay.png", artifact_names)
            self.assertIn("stage4_line_numbers.json", artifact_names)
            self.assertIn("stage4_line_number_summary.json", artifact_names)
            self.assertIn("stage4_line_number_overlay.png", artifact_names)
            self.assertIn("stage4_instrument_tags.json", artifact_names)
            self.assertIn("stage4_instrument_tag_summary.json", artifact_names)
            self.assertIn("stage4_instrument_tag_overlay.png", artifact_names)

    def test_pipeline_job_uses_selected_weight_file(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"
        selected_weight = "yolo_weights/custom-selected.pt"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": selected_weight,
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_line_number_fusion_result = {
            "line_numbers_payload": {"line_numbers": [], "rejected": []},
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"matched_line_number_count": 0},
        }
        fake_instrument_tag_fusion_result = {
            "instrument_tags_payload": {"instrument_tags": [], "rejected": []},
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"matched_instrument_tag_count": 0},
        }

        with patch("api.resolve_pipeline_weight_file", return_value=selected_weight), patch(
            "garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result
        ), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch(
            "garnet.pid_extractor.run_line_number_fusion_stage", return_value=fake_line_number_fusion_result
        ), patch(
            "garnet.pid_extractor.run_instrument_tag_fusion_stage", return_value=fake_instrument_tag_fusion_result
        ):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "4", "ocr_route": "easyocr", "weight_file": selected_weight},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

            deadline = time.time() + 10
            job_payload = None
            while time.time() < deadline:
                poll = client.get(f"/api/pipeline/jobs/{job_id}")
                self.assertEqual(poll.status_code, 200)
                job_payload = poll.json()
                if job_payload["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.1)

            self.assertIsNotNone(job_payload)
            assert job_payload is not None
            self.assertEqual(job_payload["status"], "completed")
            self.assertEqual(job_payload["weight_file"], selected_weight)
            self.assertEqual(job_payload["manifest"]["detection_weight_path"], selected_weight)

    def test_pipeline_job_runs_stage5_and_reports_pipe_mask_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": "yolo_weights/yolo11n_PPCL_640_20250204.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_pipe_mask_result = {
            "mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "mask_pixel_count": 42,
                "source_artifacts": [
                    "stage1_gray.png",
                    "stage2_ocr_regions.json",
                    "stage4_objects.json",
                ],
            },
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch("garnet.pid_extractor.run_pipe_mask_stage", return_value=fake_pipe_mask_result):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "5", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

            deadline = time.time() + 10
            job_payload = None
            while time.time() < deadline:
                poll = client.get(f"/api/pipeline/jobs/{job_id}")
                self.assertEqual(poll.status_code, 200)
                job_payload = poll.json()
                if job_payload["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.1)

            self.assertIsNotNone(job_payload)
            assert job_payload is not None
            self.assertEqual(job_payload["status"], "completed")
            self.assertEqual(job_payload["current_stage"], "stage5b_pipe_trace")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage5_pipe_mask.png", artifact_names)
            self.assertIn("stage5_pipe_mask_overlay.png", artifact_names)
            self.assertIn("stage5_pipe_mask_summary.json", artifact_names)

    def test_review_decision_update_invalidates_shared_graph_and_parent_system(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_dir = root / "job"
            system_dir = root / "system-test"
            job_dir.mkdir()
            system_dir.mkdir()
            phase8_artifacts = (
                "stage10_process_boundaries.json",
                "stage10_test_package_candidates.json",
                "stage10_engineering_view_summary.json",
                "stage10_llm_projections.json",
            )
            for artifact_name in phase8_artifacts:
                (job_dir / artifact_name).write_text("{}", encoding="utf-8")
            (job_dir / "stage_manifest.json").write_text(json.dumps({"stages": [
                {"name": "stage7b_graph_export", "status": "completed", "artifacts": ["stage7b_graph_v1.json"]},
                {"name": "stage8_graph_qa", "status": "completed", "artifacts": ["stage8_review_items.json"]},
                {"name": "stage9_apply_review_decisions", "status": "completed", "artifacts": ["stage9_corrected_graph.json"]},
            ]}))
            (job_dir / "stage7b_graph_v1.json").write_text("{}")
            system_id = "system-test"
            (system_dir / "system_manifest.json").write_text(json.dumps({
                "manifest_version": 1,
                "system_id": system_id,
                "status": "completed",
                "pages": [],
                "merge": {"status": "completed", "graph_artifact": "system_graph_v2.json"},
            }))
            (system_dir / "system_graph_v2.json").write_text("{}")
            job_id = "job-review-invalidation"
            job = {"job_id": job_id, "job_dir": str(job_dir), "system_id": system_id, "status": "completed"}
            with patch("api.PIPELINE_SYSTEMS_DIR", str(root)), patch.dict("api.PIPELINE_JOBS", {job_id: job}, clear=False):
                response = client.put(
                    f"/api/pipeline/jobs/{job_id}/artifacts/stage8_review_decisions.json",
                    json={"decisions": []},
                )
            self.assertEqual(response.status_code, 200, response.text)
            manifest = json.loads((job_dir / "stage_manifest.json").read_text())
            statuses = {item["name"]: item["status"] for item in manifest["stages"]}
            self.assertEqual(statuses["stage7b_graph_export"], "stale")
            self.assertFalse((job_dir / "stage7b_graph_v1.json").exists())
            self.assertTrue(all(not (job_dir / name).exists() for name in phase8_artifacts))
            system_manifest = json.loads((system_dir / "system_manifest.json").read_text())
            self.assertEqual(system_manifest["merge"]["status"], "stale")
            self.assertFalse((system_dir / "system_graph_v2.json").exists())

    def test_stale_stage9_withheld_graph_serialization_artifact_and_merge(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            (job_dir / "stage_manifest.json").write_text(json.dumps({"stages": [
                {"name": "stage9_apply_review_decisions", "status": "stale"},
            ]}))
            (job_dir / "stage7b_graph_v1.json").write_text(json.dumps({"schema_version": "graph_v1"}))
            (job_dir / "stage10_line_list.json").write_text(json.dumps({"lines": []}))
            (job_dir / "stage10_process_boundaries.json").write_text(json.dumps({"release_ready": True}))
            (job_dir / "stage9_release_gate.json").write_text(json.dumps({"release_ready": False}))
            (job_dir / "stage9_correction_audit.json").write_text(json.dumps({"warnings": []}))
            job_id = "job-stale-graph"
            job = {"job_id": job_id, "job_dir": str(job_dir), "status": "completed"}
            with patch.dict("api.PIPELINE_JOBS", {job_id: job}, clear=False):
                serialized = client.get(f"/api/pipeline/jobs/{job_id}")
                artifact = client.get(f"/api/pipeline/jobs/{job_id}/artifacts/stage7b_graph_v1.json")
                process_artifact = client.get(f"/api/pipeline/jobs/{job_id}/artifacts/stage10_line_list.json")
                phase8_artifact = client.get(f"/api/pipeline/jobs/{job_id}/artifacts/stage10_process_boundaries.json")
                gate_artifact = client.get(f"/api/pipeline/jobs/{job_id}/artifacts/stage9_release_gate.json")
                audit_artifact = client.get(f"/api/pipeline/jobs/{job_id}/artifacts/stage9_correction_audit.json")
                merged = client.post("/api/pipeline/merge", json={"job_ids": [job_id]})
            self.assertEqual(serialized.status_code, 200)
            self.assertNotIn("graph_v1", serialized.json())
            self.assertEqual(artifact.status_code, 409)
            self.assertEqual(process_artifact.status_code, 409)
            self.assertEqual(phase8_artifact.status_code, 409)
            self.assertEqual(gate_artifact.status_code, 200)
            self.assertEqual(audit_artifact.status_code, 200)
            self.assertEqual(merged.status_code, 409)

    def test_completed_or_unreviewed_graph_remains_readable(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            (job_dir / "stage_manifest.json").write_text(json.dumps({"stages": []}))
            (job_dir / "stage7b_graph_v1.json").write_text(json.dumps({"source_graph_artifact": "stage7_graph.json"}))
            job_id = "job-draft-graph"
            job = {"job_id": job_id, "job_dir": str(job_dir), "status": "completed"}
            with patch.dict("api.PIPELINE_JOBS", {job_id: job}, clear=False):
                draft = client.get(f"/api/pipeline/jobs/{job_id}")
            self.assertEqual(draft.status_code, 200)
            self.assertEqual(draft.json()["graph_v1"]["source_graph_artifact"], "stage7_graph.json")



if __name__ == "__main__":
    unittest.main()
