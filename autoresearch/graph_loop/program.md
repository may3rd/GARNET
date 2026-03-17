# GARNET Graph Loop — Edge-to-Graph Extraction Optimization

You are running an autoresearch-style loop focused only on graph extraction quality.

## Scope

Only modify:
- `backend/garnet/pipe_edge_connectivity.py`
- `backend/garnet/pipe_graph.py`
- `backend/garnet/pipe_graph_qa.py`

Do not modify:
- OCR code
- mask / skeleton stages
- `autoresearch/graph_loop/evaluate_graph.py`
- test images

## Fixed evaluation set

Use these 4 images only:
- `autoresearch/test_images/Test-00001.jpg`
- `autoresearch/test_images/Test-00003.jpg`
- `autoresearch/test_images/Test-00005.jpg`
- `autoresearch/test_images/Test-00008.jpg`

## Objective

Lower is better.

The loop should reduce:
- `edge_component_count`
- `unresolved_terminal_edge_count`
- `review_queue_count`
- singleton / tiny connection-attached edge chains

## Loop

1. Inspect current results in `autoresearch/graph_loop/results.tsv`
2. Form one small hypothesis
3. Modify only the three allowed graph files
4. Run:
   - `python autoresearch/graph_loop/evaluate_graph.py > autoresearch/graph_loop/run.log 2>&1`
5. Read:
   - `grep '^avg_score:' autoresearch/graph_loop/run.log`
6. If improved, keep
7. If worse, revert
8. Append one row to `results.tsv`

## Simplicity rule

Prefer the smallest change that improves the score.
