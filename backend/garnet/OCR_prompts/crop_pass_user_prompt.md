Process this cropped region from an engineering drawing as a crop OCR refinement pass.

Objectives:

1. Detect all visible text regions inside the crop.
2. Refine transcription as accurately as possible from the crop only.
3. Return tight axis-aligned bounding boxes in crop-local pixel coordinates.
4. Assign a rough class using the allowed classes.
5. Return valid JSON only using the required schema.

Priorities:

- prioritize local accuracy over global recall
- split merged text regions when supported by pixels
- preserve engineering notation exactly where visible
- include partial or degraded text only when supported by the crop
- do not use assumptions from outside the crop

Allowed classes:
equipment_tag, line_number, instrument_tag, valve_tag, utility_label, process_label, note, dimension, title_block, table_text, legend_text, unknown

Return JSON only.
