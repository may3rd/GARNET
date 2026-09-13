# Phase 8 implementation status

Phase 8 extends Stage 10 with release-gated engineering views and structured
LLM inputs. Public stage numbering remains unchanged.

## Implemented

- Process boundary candidates preserve member nodes, routes, line identities,
  cut points, isolation elements, exclusions, uncertainty, confidence, and
  provenance.
- Test-package candidates group routes by canonical line while retaining
  parallel and branch alternatives. Missing line or flow evidence stays
  unresolved.
- Structured process-description and HAZOP projections are emitted for a
  downstream LLM. They contain source evidence and questions, without making
  chemistry, safeguard, cause, or consequence claims.
- Stage 10 reads `stage9_corrected_graph.json` and
  `stage9_release_gate.json`, then writes:
  `stage10_process_boundaries.json`, `stage10_test_package_candidates.json`,
  `stage10_engineering_view_summary.json`, and
  `stage10_llm_projections.json`.
- All four Stage 10 Phase 8 artifacts carry the top-level canonical source
  graph SHA-256. The engineering summary also carries the source graph
  revision and node/edge/relationship counts, allowing downstream consumers to
  correlate every view to one corrected graph snapshot. System loading rejects
  a bundle whose individual hashes do not match one another and the current
  corrected graph.
- The new artifacts use the Stage 9 single-sheet release gate and participate
  in upstream invalidation and resume artifact validation.
- System graphs aggregate complete Phase 8 page bundles with sheet-qualified
  IDs. Test-package candidates remain page-local, while connector continuity
  remains explicit. All-legacy systems retain the prior graph-v2 behavior.
- The generic artifact PUT route is restricted to registered review/input
  artifacts and rejects derived Phase 8 replacements. JSON writes use finite,
  deterministically ordered values.

## Compatibility

Existing Stage 10 exports and Stage 11 overlays remain available with their
existing names. Legacy manifests without Phase 8 artifacts remain readable;
newly generated manifests register and validate the Phase 8 bundle.

## Validation

Focused engineering-view, LLM projection, CLI, and API tests cover release
gating, artifact registration, resume validation, deterministic output, and
system aggregation. The full backend suite remains the final check before
Phase 8 is considered complete.
