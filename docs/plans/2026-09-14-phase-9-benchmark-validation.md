# Phase 9 representative benchmark validation

The Phase 9 harness in `backend/garnet/phase9_benchmark.py` checks export
structure and evidence retention on a small synthetic fixture bundle at
`backend/tests/fixtures/phase9_benchmark/benchmark.json`.

It reports coverage and validity metrics for:

- ordered pixel route geometry and physical endpoint connectivity;
- parallel or bypass route preservation;
- canonical line assignment and OCR occurrence traceability;
- equipment/port and typed instrument relationships;
- process-boundary and test-package links;
- explicit multi-sheet connector continuity;
- visual crossings kept separate from marked tee junctions;
- repeated labels retained as sheet-scoped occurrences;
- unresolved off-page connectors never promoted to continuity;
- explicit `measures`/`controls`/`actuates` instrument semantics;
- unknown and conflicting flow states with physical endpoints preserved;
- proximity-only instrument associations retained as unresolved;
- blocked-release redaction and graph/release-gate hash bindings in the
  versioned export; and
- end-to-end validation of `garnet_downstream_export_v1`, including source
  graph hash tampering.

Unresolved, unknown, conflicting, and candidate states are retained in the
`unresolved_evidence` report section. They do not fail a case unless a fixture
expectation requires evidence to have been retained or a required resolved
relationship is absent. Detector accuracy is explicitly marked
`not_scored` because the fixtures carry structural invariants and no annotated
gold truth.

The physical graph contract is an undirected multigraph: parallel routes are
valid, and flow direction is a separate evidence field. A combined graph must
retain each sheet's entities with source-sheet provenance. Only one explicitly
reviewed connector pair is allowed to create `cross_sheet_continues`; an
unresolved connector remains review evidence and creates no continuity.

Run the focused checks from `backend/`:

```text
python -m unittest discover -s tests -p 'test_phase9_benchmark.py' -v
python -m garnet.phase9_benchmark --report /tmp/phase9-report.json
```

The command exits non-zero on contract violations and writes a stable,
machine-readable JSON report when `--report` is supplied.

The pipeline writes `stage10_final_export.json` as the versioned
`garnet_downstream_export_v1` envelope after Stage 10. It includes the legacy
process exports and Phase 8 views, records corrected-graph and release-gate
hashes, and uses an empty redacted graph when the gate is blocked. A complete
multi-page Phase 8 bundle can additionally produce `system_final_export.json`
after connector review. These derived files are invalidated with their source
graph and cannot be replaced through the artifact PUT endpoint.

The portable descriptor is `backend/schema/garnet_downstream_export_v1.json`;
the Python validator additionally enforces typed references, finite values,
ordered route geometry, and hash bindings.
