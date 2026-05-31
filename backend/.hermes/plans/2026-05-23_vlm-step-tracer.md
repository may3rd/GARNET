# VLM-Guided Step-by-Step Pipeline Tracer — Implementation Plan

**Date:** 2026-05-23
**Status:** Draft

## Goal

Replace CV pipe-mask walking with VLM-guided step-by-step tracing. VLM sees the actual P&ID drawing (not binary mask) and makes direction/turn/terminal decisions at each crop. CV is completely removed from the tracing loop — VLM handles all interpretation.

## Motivation

CV pipe follower has fundamental limitations:
- Cannot distinguish pipe lines from text/annotations/symbols on the raw mask
- False junctions from mask noise (3500+ components)
- Cannot identify instrument tags, valve types, or equipment — must fall back to YOLO objects
- Cannot "jump" gaps intelligently — needs YOLO bbox coordinates
- Misses corners when `_walk_distance` jumps too far

VLM sees the actual drawing — it knows what a pipe looks like vs. a text label. It can identify turns, junctions, valves, and equipment naturally.

## Architecture

### Core Loop

```
1. Place MARK (grey circle + connection line) at current position
2. CROP image (300-400px) centered on mark
3. Send marked crop to VLM with tracing prompt
4. Parse VLM response tokens
5. Interpret: go straight? turn? hit object? terminal?
6. Place new MARK at target position
7. Recenter crop, repeat until terminal
```

### Marking System

Each step leaves a visible trace on the image so VLM can see where it's been:

- **Current position**: filled grey circle (radius 5px), color `(128, 128, 128)`
- **Previous path**: grey line (width 2px) connecting all previous positions
- **All marks drawn on a temp copy** of the image — crop is taken from this marked copy

Rationale: grey is neutral — visible but doesn't confuse VLM with pipe colors (black/blue). The line shows "already traced" so VLM doesn't backtrack.

### Cropping Strategy

- Crop size: 300×300 to 400×400 pixels
- Centered on the LATEST mark point
- If the mark is near image edge, crop is asymmetric (extend opposite direction)
- Crop includes: current mark, some path history, and enough context ahead for VLM to see turns/junctions

### VLM Protocol

System prompt instructs VLM to trace from the grey dot along the pipe line. VLM responds with structured tokens:

```
<|go|>DIRECTION DISTANCE
  Walk straight. DIRECTION = UP/DOWN/LEFT/RIGHT. DISTANCE in pixels.

<|turn|>DIRECTION
  Pipe turns 90° at a corner. Mark the corner, then turn.

<|hit|>CLASS
  Pipe reaches an inline object (valve, reducer, instrument). CLASS free-form.

<|jump|>DIRECTION DISTANCE  
  Jump past an object/gap to resume tracing. Pipe continues same direction.

<|term|>CLASS
  Terminal reached. CLASS = junction, equipment, page_connection, instrument_tag, sheet_edge.
```

### Response Handling

- **`<|go|>`**: Move CURSOR in direction by distance. Draw line from current mark to new position. Place new mark circle. Recenter crop.
- **`<|turn|>`**: Place mark at corner position (don't move past it). Update CURSOR direction. Recenter crop.
- **`<|hit|>`**: Record hit. If inline object, expect next token to be `<|jump|>`.
- **`<|jump|>`**: Place mark at jump destination. Draw dashed line through object area.
- **`<|term|>`**: Trace complete. Record terminal class.

### Cursor State

```python
@dataclass
class VLMCursor:
    x: int          # current mark position (image px)
    y: int
    direction: str  # current trace direction
    path: list[tuple[int, int]]  # all mark positions
    total_distance: int
```

### Crop with Marks

```python
def crop_marked(image, cursor, crop_size=350):
    """Draw all path marks + lines on image copy, crop around cursor."""
    marked = image.copy()
    # Draw path lines
    for i in range(1, len(cursor.path)):
        cv2.line(marked, cursor.path[i-1], cursor.path[i], (128,128,128), 2)
    # Draw current mark
    cv2.circle(marked, (cursor.x, cursor.y), 5, (128,128,128), -1)
    # Crop centered on cursor
    half = crop_size // 2
    x1, y1 = max(0, cursor.x - half), max(0, cursor.y - half)
    x2, y2 = min(w, cursor.x + half), min(h, cursor.y + half)
    return marked[y1:y2, x1:x2], (x1, y1, x2, y2)
```

### VLM Call

Same pattern as existing `_call_vlm_raw`:
- OpenRouter, model `google/gemini-2.5-pro`
- System prompt with protocol definition
- User prompt: "Trace from the grey dot along the pipe line."
- Image: marked crop as base64
- Max tokens: 256
- Temperature: 0.0

### Trace Termination

Trace ends when VLM returns `<|term|>` with one of:
- `junction` — reached a tee/cross
- `equipment` — reached vessel/pump/tank
- `page_connection` — reached another sheet connector
- `instrument_tag` — reached instrument label
- `sheet_edge` — pipe exits drawing
- `dead_end` — no more pipe

### Orchestrator Integration

Replace `_trace_segment_with` in `agent2_hybrid.py` with `_trace_vlm_segment`:

```python
def _trace_vlm_segment(self, image, port_x, port_y, direction, model):
    cursor = VLMCursor(x=port_x, y=port_y, direction=direction)
    cursor.path.append((port_x, port_y))
    
    for step_i in range(MAX_VLM_STEPS):
        marked_crop, crop_bbox = crop_marked(image, cursor, CROP_SIZE)
        response = call_vlm(marked_crop, model)
        tokens = parse_trace_response(response)
        
        for token in tokens:
            if token.type == 'go':
                cursor = apply_go(cursor, token)
            elif token.type == 'turn':
                cursor = apply_turn(cursor, token)
            elif token.type == 'hit':
                record_hit(token)
            elif token.type == 'jump':
                cursor = apply_jump(cursor, token)
            elif token.type == 'term':
                return build_result(cursor, token)
    
    return build_result(cursor, 'max_steps')
```

## Files

| File | Action | Purpose |
|------|--------|---------|
| `garnet/visual_primitives/vlm_cursor.py` | NEW | VLMCursor dataclass + marking/cropping logic |
| `garnet/visual_primitives/vlm_trace_parser.py` | NEW | Parse `<|go|>`, `<|turn|>`, `<|jump|>`, `<|hit|>`, `<|term|>` tokens |
| `garnet/visual_primitives/prompts.py` | PATCH | Add VLM tracing system+user prompts |
| `garnet/visual_primitives/agent2_hybrid.py` | PATCH | Add `_trace_vlm_segment`, wire as alternative to CV |
| `tests/test_visual_primitives/test_vlm_cursor.py` | NEW | Unit tests for cursor+marking |
| `tests/test_visual_primitives/test_vlm_trace_parser.py` | NEW | Unit tests for token parser |

## Steps

### Step 1: VLMCursor class (`vlm_cursor.py`)
- Dataclass with x, y, direction, path, total_distance
- `crop_marked(image, cursor, crop_size)` — draw marks+lines, return crop
- `apply_go(cursor, direction, distance)` — move and draw
- `apply_turn(cursor, direction)` — turn at current position
- `apply_jump(cursor, direction, distance)` — jump with dashed line

### Step 2: VLM token parser (`vlm_trace_parser.py`)
- Regex patterns for each token type
- Parse multi-token responses (VLM may return multiple tokens)
- Handle malformed/cut-off responses gracefully

### Step 3: VLM tracing prompts (`prompts.py`)
- System prompt: protocol definition with visual description of marks
- User prompt: short instruction to trace from grey dot
- Include edge cases: "what if no pipe?" "what if multiple pipes?"

### Step 4: VLM trace orchestrator (patch `agent2_hybrid.py`)
- Add `_trace_vlm_segment()` method
- Wire into Phase 2: use VLM trace instead of CV trace per connection
- Keep existing port detection, Phase 3 classification
- Add `--tracer {cv|vlm}` CLI arg to switch between modes

### Step 5: Unit tests
- `test_vlm_cursor.py`: mark drawing, crop coordinates, path accumulation
- `test_vlm_trace_parser.py`: token parsing, edge cases, multi-token

### Step 6: Integration test
- Run VLM tracer on Test-00001 with all 9 connections
- Compare paths against ground truth
- Measure speed and token usage

## Risks & Tradeoffs

| Risk | Mitigation |
|------|-----------|
| VLM speed: 3-5s per step × 20 steps = 60-100s per connection | Parallel VLM calls across connections? Accept speed for accuracy |
| VLM inconsistency: same crop may get different answers | Temperature 0.0, structured output protocol |
| Token cost: 9 connections × 20 steps × ~500 tokens = 90K tokens | ~$0.25/sheet at Gemini pricing |
| VLM may "hallucinate" pipe direction | Grey visited path prevents backtracking; crop context keeps it grounded |
| Crop edge artifacts: VLM can't see beyond crop boundary | VLM should report `<|go|>crop_edge` or `<|go|>DIR MAX` when pipe leaves crop |

## Open Questions

1. Crop size: 300px or 400px? Larger = VLM sees more context but image is lower detail. Start with 350px.
2. Max VLM steps per connection: 20 or 30? Each step covers 50-150px of pipe at a corner. 20 steps = ~2000px. Start with 25.
3. VLM model: Gemini 2.5 Pro is reliable. Try Gemini 3.0 Flash for speed? Risk: Flash gave bad bboxes before.
4. Should we keep CV tracer as fallback? Yes — `--tracer vlm` flag, default stays `cv`.

## Success Criteria

- 7+ of 9 connections follow correct path per ground truth
- VLM identifies inline objects (valves, reducers) correctly
- VLM correctly identifies terminals (instrument tags, equipment, junctions)
- Per-sheet runtime < 5 minutes (acceptable for batch processing)
