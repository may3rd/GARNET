1. Implement the output data to graph or [pyDEXPI](https://github.com/process-intelligence-research/pyDEXPI)

## Knowledge graph + HAZOP

Gate and layers are built — `garnet/knowledge_graph.py` (+ `garnet/tests/test_knowledge_graph.py`).
Nothing is wired into a caller yet, and this is not committed.

Representation decision (2026-10-24): property graph is the canonical model, JSON is only the
wire format. Graph-shaped JSON-LD is emitted; no graph DB is committed to until query patterns
are proven. The physical graph and the HAZOP argument graph stay separate, linked by node id.

- [ ] Commit `garnet/knowledge_graph.py` + tests on `feat/garnet-frontend-path` (currently untracked)
- [ ] Wire the gate into a caller — either `garnet/api.py` (job output beside
      `stage6_trace_associations.json`) or keep it a standalone
      `python -m garnet.knowledge_graph <export.json> [--out DIR]`
- [ ] Decide the study-node basis: equipment endpoints (current default) vs line-based nodes,
      if workshops study lines. Contained change in `build_process_knowledge`.
- [ ] Feed it a real released export — has only run against fixtures so far, never a
      `stage10_final_export` / GARNET job output
- [ ] Load the emitted graph into Postgres (`nodes` / `edges`) and prove the HAZOP queries
      with recursive CTEs before considering Neo4j
- [ ] Populate `deviation_graph` — guide word, parameter, cause, consequence, safeguard per
      study node. Schema exists and validates; no workshop data yet.
- [ ] Revisit RDF/OWL only if ISO 15926 / IEC 62424 exchange is actually required
- [ ] Add revision as a property (`rev`, `valid_from`) before any bitemporal modelling
