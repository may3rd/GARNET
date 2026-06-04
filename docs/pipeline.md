# Pipeline: End-to-End P&ID Digitization and Connectivity Analysis

A staged, runnable pipeline with CLI is available in `garnet/pid_extractor.py`. This pipeline performs comprehensive P&ID analysis with the following stages:

## Usage

```bash
python garnet/pid_extractor.py \
    --image path/to/pid_image.png \
    --coco path/to/coco_annotations.json \
    --ocr path/to/ocr_results.json \
    --arrow-coco path/to/coco_arrows.json \
    --out output/ \
    --stop-after 11
```

## Pipeline Stages

1. **Stage 1: Input Normalization**
   - Load P&ID image (BGR format)
   - Apply normalization (deskew, contrast enhancement)
   - Generate binary mask for pipe detection
   - Output: Normalized image, grayscale, binary mask

2. **Stage 2: OCR Discovery**
   - Run selected OCR route (EasyOCR, Gemini, PaddleOCR, or OCRMac)
   - Detect text regions and extract text content
   - Output: OCR results with bounding boxes and text

3. **Stage 3: HITL Review (Manual)**
   - Reserved for human-in-the-loop review
   - Major equipment bounding boxes supplied through LabelMe/frontend review
   - Note: This stage is managed outside the automatic CLI flow

4. **Stage 4: Object Detection & Fusion**
   - Run SAHI-based object detection (YOLOv8/YOLOv11)
   - Process line number fusion and instrument tag fusion
   - Apply topology-marker routing for junction detection
   - Output: Detected objects with bounding boxes, class labels, and confidence scores

5. **Stage 5: Pipe Mask Generation**
   - Create provisional pipe mask from binary image
   - Suppress OCR and object regions to focus on pipe regions
   - Output: Pipe mask image and summary statistics

6. **Stage 5b: Pipe Tracing**
   - Trace pipe centerlines from mask using computer vision algorithms
   - Start from detected ports and follow pipe paths
   - Handle inline objects (valves, reducers) by jumping over them
   - Detect terminals (equipment, tags, junctions, sheet edges)
   - Output: Traced pipe paths, junctions, and endpoints

7. **Stage 6: Trace Associations**
   - Associate traced paths with ports, inline objects, line numbers, instruments
   - Connect detected objects to the pipe network
   - Fill missing line numbers with simulated-HITL placeholders
   - Output: Object-to-pipe associations and connection data

8. **Stage 7: Geometric Graph Assembly**
   - Assemble and normalize the geometric trace graph
   - Label page connectors for multi-document support
   - Run graph quality assurance checks
   - Export initial graph payload (v1)
   - Output: Geometric graph representation (nodes, edges, connectivity)

9. **Stage 8: HITL Review Package**
   - Build graph/line-number human-in-the-loop review package
   - Prepare items for validation (junctions, connections, line numbers)
   - Output: Review items ready for frontend inspection

10. **Stage 9: Apply Review Decisions**
    - Apply human review decisions to correct the graph
    - Pass through unchanged when no decisions exist
    - Output: Corrected graph based on review feedback

11. **Stage 10: Process Exports**
    - Generate line lists, equipment connectivity, inline MTO
    - Create inline observations and instrument index
    - Output: Process exports in JSON/CSV formats

12. **Stage 11: Connection Overlay**
    - Render connection-pipeline overlay for visual review
    - Create visual representation of connections on original image
    - Output: Connection pipeline overlay image

## Configurable Parameters (PipelineConfig)

- Image processing: DPI, Canny thresholds, binarization settings
- Detection: Confidence thresholds per class, image size, overlap ratio
- Text association: Multiplier for bbox diagonal
- Graph: Connection radius, port counts, angle separation
- Valve linking: Directional strategy, edge offset, raycast step
- Template matching: For valve orientation detection
- Cleanup: Bridge max distance, angle tolerance
- Tracing: Branch stub length, merge angle tolerance, opposite angle tolerance
- Crossing resolution: Center blob radius, threshold, marker match distance

## Note

This pipeline is designed to be modular, so each step can be run independently or as part of the full digitization workflow. Use `--stop-after N` to run only specific stages. The stage numbering intentionally skips Stage 3 in the automatic CLI flow because HITL input is managed outside this runner.