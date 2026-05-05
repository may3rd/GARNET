# Multi-Sheet P&ID Merge Contract

**Version:** 1.0  
**Status:** Contract only — full merge engine not yet implemented  
**Stage:** S5-04  
**Schema base:** `graph_v1.json`

---

## 1. Problem Statement

P&ID sheets contain **off-page connectors** — pipe or signal lines that cross the sheet boundary and reference another sheet (e.g., a line exiting the right edge of Sheet 1 labeled "SHEET A-3"). Each sheet is digitized independently, producing a `graph_v1.json`. The merge contract defines how to combine these into a single coherent plant-wide graph.

---

## 2. Concepts

### 2.1 Off-Page Connector

An off-page connector is an **edge terminal** on a sheet boundary that carries a sheet reference label. It has two representations:

| Level | Description |
|-------|-------------|
| **Edge-level** (canonical) | The `edge.off_page_connector` field on the terminating edge |
| **Node-level** (legacy) | A node with `tags.page_reference` — used by `stage12c_page_connector_labeling` |

The edge-level representation is the authoritative source for merge operations.

### 2.2 Connector Reference

The text label associated with an off-page connector, e.g., `"SHEET A-3"`. Parsed by `classify_off_page_reference()` in `page_connector.py`.

```
reference_type: "sheet" | "pid" | "drawing" | "figure"
reference_value: string  — the canonical key, e.g. "A-3"
direction: "output" | "input" | "bidirectional"
```

`reference_value` is the **global merge key**. It must be unique within a sheet (one off-page connector per destination), but the same value may appear on multiple sheets — those are the pairs that get merged.

### 2.3 Sheet Identity

Each `graph_v1.json` carries a `document.doc_id` field. This is the **sheet ID** used in merge operations.

---

## 3. Required Schema Fields (graph_v1.json extensions)

### 3.1 Edge `off_page_connector` field

Add to every edge schema entry in `graph_v1.json`:

```json
"off_page_connector": {
  "type": "object",
  "nullable": true,
  "properties": {
    "reference_type": {
      "enum": ["sheet", "pid", "drawing", "figure"],
      "description": "Category of the referenced document"
    },
    "reference_value": {
      "type": "string",
      "description": "Global merge key. Must be stable across sheets. Example: 'A-3'"
    },
    "direction": {
      "enum": ["output", "input", "bidirectional"],
      "description": "output=signal/pipe leaves this sheet; input=enters this sheet"
    },
    "exit_terminal": {
      "type": "string",
      "enum": ["source", "destination"],
      "description": "Which edge terminal is the off-page junction point"
    },
    "local_edge_id": {
      "type": "string",
      "description": "ID of the edge as defined in this sheet's edge list"
    }
  },
  "required": ["reference_type", "reference_value", "direction", "exit_terminal"]
}
```

**Rule:** `off_page_connector` is `null` for internal edges. Only edges that terminate at a sheet boundary carry this field.

### 3.2 Document `doc_id` field

```json
"doc_id": {
  "type": "string",
  "description": "Unique identifier for this sheet. Used as source/target in merge operations. Example: 'SHEET-001' or 'PID-A3'"
}
```

`doc_id` is **required** for mergeable sheets. It appears in the `document` section of `graph_v1.json`.

### 3.3 Document `sheet_index` field (optional but recommended)

```json
"sheet_index": {
  "type": "integer",
  "description": "Numeric order of this sheet in the drawing set. Used for directional inference."
}
```

---

## 4. Direction Inference Rules

When `direction` is not explicitly set, infer from context:

| Condition | Inferred Direction |
|-----------|-------------------|
| Off-page symbol has an arrow pointing toward the sheet boundary | `output` |
| Off-page symbol has an arrow pointing away from the boundary | `input` |
| No directional arrow visible | `bidirectional` |
| Same `reference_value` appears twice on the same sheet (goes out and comes back) | first occurrence = `output`, second = `input` |

---

## 5. Merge Operation

### 5.1 Merge Key

```
merge_key = (reference_type, reference_value)
```

### 5.2 Connector Pairing Rule

For each unique `(reference_type, reference_value)` pair:

1. Collect all off-page connectors across all sheets with that merge key
2. If exactly 2 connectors are found with different `doc_id` values → **merge pair**
3. If >2 connectors share the same merge key → **ambiguous** → flag for human review
4. If only 1 connector exists globally → **dangling** → flag for human review

### 5.3 Merge Action for a Pair

Given connector A (Sheet A, direction=output) and connector B (Sheet B, direction=input):

1. Create a new **cross-sheet virtual edge** or extend the existing off-page edge in the merged graph
2. The cross-sheet edge connects the internal topology of Sheet A to the internal topology of Sheet B
3. The `reference_value` becomes a property of the merged edge
4. Both original off-page terminals are marked as **resolved** in their respective sheets

### 5.4 Conflict Resolution

| Conflict | Resolution |
|----------|-----------|
| Same merge key, >2 sheets | Flag as `AMBIGUOUS_MERGE` — human reviewer picks which pair to connect |
| Same merge key, same sheet (duplicate) | Flag as `INTRA_SHEET_DUPLICATE` — likely a drafting error |
| reference_value mismatch within same physical connector | Flag as `LABEL_CONFLICT` — inconsistent OCR |

---

## 6. Output Schema: Merged Graph

The merge engine produces a merged `graph_vN.json` (schema versioned forward from `graph_v1`):

```json
{
  "schema_version": "graph_v2",
  "document": {
    "doc_id": "MERGED-PLANT-001",
    "source_sheets": ["SHEET-001", "SHEET-002", "SHEET-A3"]
  },
  "cross_sheet_edges": [
    {
      "id": "xs::SHEET-001::A-3::SHEET-A3",
      "merge_key": {"reference_type": "sheet", "reference_value": "A-3"},
      "sheets": ["SHEET-001", "SHEET-A3"],
      "terminals": [
        {"sheet": "SHEET-001", "local_edge_id": "...", "exit_terminal": "source"},
        {"sheet": "SHEET-A3", "local_edge_id": "...", "exit_terminal": "destination"}
      ],
      "status": "merged"
    }
  ],
  "merge_issues": [
    {
      "issue_id": "AMBIGUOUS_MERGE::sheet::B-2",
      "type": "ambiguous_merge",
      "merge_key": {"reference_type": "sheet", "reference_value": "B-2"},
      "sheets_involved": ["SHEET-001", "SHEET-002", "SHEET-B2"],
      "resolution": "pending_human_review"
    }
  ]
}
```

---

## 7. Stage12c Enhancement Required

Before the merge contract can be used, `stage12c_page_connector_labeling` must be enhanced to produce edge-level off-page connector data. Currently it only produces node-level labels.

**Required change:** After labeling, write `off_page_connector` data linked to the correct edge and terminal, not just the node.

**Files affected:**
- `backend/garnet/page_connector.py` — add edge-terminal enrichment
- `backend/garnet/pid_extractor.py` (stage12c) — write edge-level off-page data
- `backend/schema/graph_v1.json` — add `off_page_connector` field to edge schema
- `backend/garnet/graph_export_adapter.py` — propagate off_page_connector to graph_v1 export

---

## 8. Implementation Phases

| Phase | Description | Blocker |
|-------|-------------|---------|
| **S5-04a** (this contract) | Define schema + contract doc | None |
| **S6-01** | Add `off_page_connector` to edge schema in `graph_v1.json` | None |
| **S6-02** | Enhance stage12c to produce edge-level off-page data | S6-01 |
| **S6-03** | Wire off_page_connector into graph export (graph_v1 adapter) | S6-02 |
| **S7-01** | Implement merge engine (pairing + conflict detection) | S6-03 |

---

## 9. Notes

- This contract assumes **one off-page connector per reference_value per sheet**. If a sheet has multiple lines going to the same destination, they should share the same `reference_value` and be distinguished by `line_tag` or signal class.
- The `reference_value` is parsed by `classify_off_page_reference()` — it handles `SHEET`, `PID`, `DWG`, `FIG.` prefixes. Custom prefixes may need regex updates.
- Coordinate systems must match across sheets for spatial merge operations (future phase).
