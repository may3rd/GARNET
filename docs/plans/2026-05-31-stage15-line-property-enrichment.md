# Stage 15 Line Property Enrichment Plan

Goal: enrich process exports with material and design-condition properties derived from reviewed line numbers.

Current state:
- `stage15_inline_mto.json` contains unique physical inline objects only.
- Each MTO item carries `line_number_ids`, `edge_ids`, and placeholder fields:
  - `material_basis.status = pending_line_property_data`
  - `design_condition_basis.status = pending_line_property_data`
- `stage15_inline_observations.json` preserves all graph/tracer observations for QA, including synthetic Stage 5b hits.

Best-practice add-on path:
1. Add reviewed line-property input artifact.
   - Suggested file: `stage15_line_property_table.json` or future HITL output.
   - Key by normalized line number / line object id.
   - Store material class, piping spec, service, nominal size, insulation, design pressure, design temperature, operating pressure, operating temperature, source, and review state.

2. Normalize line identifiers before joining.
   - Join by `line_number_id` first.
   - Add fallback by normalized line text only after human review.
   - Keep conflicts explicit; do not silently pick one property set when an inline item has multiple line numbers.

3. Enrich line list first.
   - Add `line_properties` to each `stage15_line_list.json` row.
   - Mark `property_state`: `resolved`, `partial`, `conflict`, or `missing`.

4. Propagate from line list to equipment and MTO exports.
   - Inline object properties should be inherited from connected line rows.
   - Equipment connectivity should reference line properties, not duplicate them.

5. Preserve provenance.
   - Every property value should carry `source`, `review_state`, and `source_line_number_id`.
   - Human overrides should be stored as separate review decisions, not overwritten detector data.

6. QA checks.
   - Flag inline objects with multiple conflicting property sets.
   - Flag lines missing material/design conditions.
   - Flag pressure/temperature class mismatches across connected components.

Deferred implementation:
- Do not infer material/design conditions from OCR line strings alone unless a reviewed parser/table exists.
- Do not use placeholder values in MTO calculations.
