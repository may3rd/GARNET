Process this full engineering drawing image as a full-page OCR and text-localization pass.

Objectives:
1. Detect all visible text regions across the entire image.
2. Transcribe each text region as faithfully as possible.
3. Return tight axis-aligned bounding boxes in pixel coordinates.
4. Assign a rough class using the allowed classes.
5. Return valid JSON only using the required schema.

Priorities:
- prioritize recall
- avoid duplicate detections
- preserve engineering notation exactly where visible
- include small, rotated, faint, and partially readable text if supported by pixels
- do not infer connectivity or engineering relationships

Allowed classes:
equipment_tag, line_number, instrument_tag, valve_tag, utility_label, process_label, note, dimension, title_block, table_text, legend_text, unknown

Return JSON only.