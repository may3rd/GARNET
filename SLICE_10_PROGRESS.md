# Slice 10 Progress Log

## Purpose
- Working log for the Stage 10 edge-tracing slice.
- Track provisional pipe-edge extraction from the skeleton and clustered node candidates.

## Entry format
- Timestamp
- Task
- Action
- Evidence
- Verification
- Next step / blocker

## Entries

### 2026-03-09 08:09 ICT
- Task: `Slice 10 / Implementation + real sample`
- Action: Added provisional edge tracing from the Stage 7 skeleton using Stage 9 clustered nodes, emitted edge JSON plus overlay artifacts, and ran the sample through Stage 10.
- Evidence:
  - [`backend/garnet/pipe_edges.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/garnet/pipe_edges.py)
  - [`backend/output/pid_extractor_stage10_start/stage10_pipe_edges.json`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/output/pid_extractor_stage10_start/stage10_pipe_edges.json)
  - [`backend/output/pid_extractor_stage10_start/stage10_pipe_edge_summary.json`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/output/pid_extractor_stage10_start/stage10_pipe_edge_summary.json)
- Verification:
  - `cd backend && python -m unittest discover -s tests -p 'test*.py' -v` -> pass with API tests skipped when `pdf2image` is unavailable
  - `cd backend && python -m garnet.pid_extractor --image sample.png --out output/pid_extractor_stage10_start --ocr-route easyocr --stop-after 10` -> pass
  - Stage 10 summary reports:
    - `edge_count = 1698`
    - `min_edge_length_px = 2`
- Next step / blocker:
  - Review clustered junction candidates explicitly instead of treating all of them as accepted graph truth.

### 2026-03-11 14:47 ICT
- Task: `Slice 10 / Crossing-vs-junction disambiguation + PPCL tuning`
- Action: Added explicit crossing resolution inside the current Stage 10 flow while keeping public stage numbering stable. The new Stage 10 now classifies clustered junction candidates as `confirmed_junction`, `non_connecting_crossing`, or `unresolved`, emits a dedicated crossing-resolution artifact bundle, and routes edge tracing through non-connecting crossings instead of always promoting them to graph nodes. Tuned the first accepted baseline on the 9-image PPCL set by merging branch exits by angle and widening the center-blob window.
- Evidence:
  - [`backend/garnet/pipe_crossings.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/garnet/pipe_crossings.py)
  - [`backend/garnet/pipe_edges.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/garnet/pipe_edges.py)
  - [`backend/garnet/pid_extractor.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
  - [`backend/garnet/pipe_junctions.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/garnet/pipe_junctions.py)
  - [`backend/output/ppcl_crossing_tuned_final_20260311`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/output/ppcl_crossing_tuned_final_20260311)
- Verification:
  - `cd backend && python -m py_compile api.py garnet/*.py garnet/utils/*.py tests/*.py` -> pass
  - `cd backend && python -m unittest discover -s tests -p 'test_pipe_crossings.py' -v` -> pass
  - `cd backend && python -m unittest discover -s tests -p 'test_pipe_edges.py' -v` -> pass
  - `cd backend && python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v` -> pass
  - `cd backend && python -m garnet.pid_extractor --image test/ppcl/Test-00001.jpg --ocr-route ocrmac --stop-after 11 --out output/ppcl_crossing_tune_final_test_00001` -> pass
  - `cd backend && for img in test/ppcl/Test-*.jpg; do python -m garnet.pid_extractor --image "$img" --ocr-route ocrmac --stop-after 11 --out "output/ppcl_crossing_tuned_final_20260311/$(basename "${img%.jpg}")"; done` -> pass
  - Accepted Stage 10 baseline:
    - `branch_stub_length_px = 8`
    - `branch_merge_angle_tolerance_deg = 18.0`
    - `opposite_angle_tolerance_deg = 35.0`
    - `center_blob_radius_px = 4`
    - `center_blob_threshold = 0.5`
  - Final 9-image PPCL rerun summary:
    - `candidate_count = 6748`
    - `confirmed_junction_count = 5365`
    - `non_connecting_crossing_count = 627`
    - `unresolved_candidate_count = 756`
    - per-sheet crossing counts ranged from `48` to `81`
    - per-sheet unresolved counts ranged from `41` to `217`
- Next step / blocker:
  - The baseline is now conservative enough to avoid silently forcing most 4-way candidates into junctions, but Stage 10 still needs future Stage 4 symbol hooks and a later review pass on high-unresolved sheets such as `Test-00005`.
