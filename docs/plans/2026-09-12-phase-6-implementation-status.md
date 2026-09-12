# Phase 6 implementation status

Phase 6 preserves the reviewed single-sheet graph contract in a deterministic
multi-sheet projection.

## Implemented

- `resolve_merge_pairs(...)` retains its legacy graph-v2 summaries and adds a
  `combined_graph` for direct system traversal.
- Nodes, pixel-route edges, equipment, ports, lines, inline objects,
  instruments, and existing relationships are preserved with typed,
  drawing-qualified IDs.
- Off-page connectors are explicit entities linked to their local route.
- A uniquely resolved pair emits one `cross_sheet_continues` relationship with
  automatic or manual match evidence, provenance, and review state.
- Missing, rejected, non-reciprocal, dangling, and ambiguous connectors remain
  explicit issues and do not produce continuity relationships.
- Boundary flow is derived from local forward/reverse/bidirectional evidence
  and the connector's source or target terminal. Physical continuity remains a
  separate fact.
- Input sheet order does not change the serialized merge result.
- The system API persists and serves the additive graph, and atomic JSON writes
  reject non-finite numeric values.

## Compatibility

Existing `cross_sheet_edges`, `merge_issues`, `per_sheet_resolved`, embedded
sheet graph-v1 payloads, and connector review records remain available. The
combined graph is additive.

## Next phase

Phase 7 should complete topology review operations, retain their audit history,
and prevent release while required topology or connector decisions remain
unresolved.
