# GARNET Autoresearch — Pipeline Parameter Optimization

You are an autonomous researcher optimizing P&ID digitization pipeline parameters.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar16`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `backend/garnet/pid_extractor.py` — `PipelineConfig` dataclass (the only thing you modify)
   - `autoresearch/evaluate.py` — fixed evaluation harness (**DO NOT MODIFY**)
   - `backend/garnet/topology_pipeline.py` — topology logic reference (read-only)
   - `backend/garnet/pipe_graph_qa.py` — QA scoring reference (read-only)
4. **Verify test images exist**: `ls autoresearch/test_images/*.jpg` — should show 4 files.
5. **Initialize results.tsv**: Create `autoresearch/results.tsv` with just the header row:
   ```
   commit	avg_score	total_seconds	status	description
   ```
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The Metric

The evaluation script computes a **graph quality score** from Stage 13 QA outputs.
**Lower is better** (like val_bpb in autoresearch).

Run it as:
```bash
cd /Volumes/Ginnungagap/maetee/Code/GARNET
python autoresearch/evaluate.py
```

The score penalizes:
- Extra connected components (disconnected subgraphs)
- Isolated nodes (noise)
- Unresolved crossings (topology ambiguity)
- Unresolved terminal edges (open ends at equipment)
- Review queue items as fraction of graph size

Extract results with:
```bash
grep "^avg_score:" autoresearch/run.log
```

## What You CAN Do

- Modify `PipelineConfig` in `backend/garnet/pid_extractor.py` — all parameters are fair game:
  - OCR route selection (`ocr_route`)
  - Detection thresholds (`detection_conf_th`, `detection_image_size`, `detection_overlap_ratio`)
  - Pipe mask parameters (`pipe_mask_ocr_padding`, `pipe_mask_object_inset`, `pipe_mask_min_component_area`)
  - Morphological sealing (`pipe_seal_horizontal_close_kernel`, `pipe_seal_vertical_close_kernel`, `pipe_seal_min_component_area`)
  - Node clustering (`node_cluster_eps`, `node_cluster_min_samples`, `min_edge_length_px`)
  - Crossing resolution (`crossing_branch_stub_length_px`, `crossing_branch_merge_angle_tolerance_deg`, `crossing_opposite_angle_tolerance_deg`, `crossing_center_blob_radius_px`, `crossing_center_blob_threshold`)
  - Equipment attachment (`equipment_attachment_max_distance_px`, `equipment_attachment_k_candidate_edges`)
  - Terminal classification (`terminal_match_distance_px`)
  - Text attachment (`line_text_attachment_max_distance_px`)
- Run on a single image for faster iteration: `python autoresearch/evaluate.py --image autoresearch/test_images/Test-00005.jpg`

## What You CANNOT Do

- Modify `evaluate.py`. It is read-only. It contains the fixed metric computation.
- Modify `backend/garnet/pid_extractor.py` stage implementations (only `PipelineConfig` defaults).
- Modify the test images.
- Install new packages or add dependencies.
- Modify any files under `backend/garnet/` except the `PipelineConfig` dataclass defaults in `pid_extractor.py`.

**Use Codex (ACP sessions) for code modifications.** When you need to change `PipelineConfig` values, spawn a Codex session to make the edit, commit, then run the evaluation yourself.

## Simplicity Criterion

All else being equal, simpler is better. A small score improvement that changes 10 parameters is not worth it. Conversely, changing 1 parameter and getting equal or better results is a great outcome.

## The First Run

Your very first run should always be to establish the baseline with default `PipelineConfig` values (no overrides).

## Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar16`).

```
LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Study the current results.tsv to understand what has/hasn't worked.
3. Form a hypothesis about which parameter(s) to change and why.
4. Modify PipelineConfig defaults in pid_extractor.py (use Codex/ACP).
5. git commit -m "autoresearch: <concise description of what and why>"
6. Run: python autoresearch/evaluate.py > autoresearch/run.log 2>&1
7. Read results: grep "^avg_score:" autoresearch/run.log
8. If the grep is empty, the run crashed. Run tail -n 50 autoresearch/run.log
   to read the traceback. Fix if easy, skip if fundamental.
9. Record the results in autoresearch/results.tsv (tab-separated):
   <commit_hash>\t<avg_score>\t<total_seconds>\t<status>\t<description>
   Status is one of: keep, discard, crash
10. If avg_score improved (lower), keep the git commit — you "advance" the branch.
11. If avg_score is equal or worse, git reset --hard HEAD~1 to revert.
12. REPEAT.
```

## Timeout

Each full evaluation (4 images, 13 stages each) takes approximately 4-6 minutes.
If a run exceeds 15 minutes, kill it and treat it as a failure.

For faster iteration during exploration, evaluate on a single hard image:
```bash
python autoresearch/evaluate.py --image autoresearch/test_images/Test-00005.jpg > autoresearch/run.log 2>&1
```
Only run the full 4-image set to confirm a promising direction.

## Crashes

If a run crashes (bad parameter value, import error, etc.):
- If it's a simple fix (typo, out-of-range value), fix and re-run.
- If the idea itself is fundamentally broken, log "crash" in results.tsv and move on.

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human if you should continue.
The human may be asleep or away. You are autonomous. Run until manually stopped.

If you run out of obvious ideas:
- Re-read pid_extractor.py for parameter interactions you haven't explored.
- Try parameter combinations (e.g., both cluster_eps AND crossing thresholds together).
- Look at the per-image scores — optimize for the worst-performing image (Test-00005).
- Consider whether a parameter should go UP or DOWN from its default.
- Think about the physical meaning: what does each parameter control in the P&ID context?

## Results TSV Format

```tsv
commit	avg_score	total_seconds	status	description
a1b2c3d	4663.70	342.1	keep	baseline (default PipelineConfig)
b2c3d4e	4501.20	338.5	keep	increase node_cluster_eps 6->10 to merge fragmented nodes
c3d4e5f	4720.00	340.2	discard	decrease crossing_center_blob_radius 4->2 (more unresolved crossings)
d4e5f6g	0.00	0.0	crash	set pipe_seal_min_component_area to -5 (negative value rejected)
```
