# GARNET Web Application - Product Overview

**Document Purpose**: Reference specification for redesigning the P&ID object detection web interface with state-of-the-art UX/UI.

---

## 1. Product Summary

**GARNET** (GCME AI-Recognition Network for Engineering Technology) is a web application for automated symbol detection and analysis in Piping and Instrumentation Diagrams (P&IDs).

### Core Value Proposition

Transform static P&ID images into structured, queryable data in minutes—not hours of manual work.

### User Workflow

```
UPLOAD → DETECT → REVIEW → CORRECT → EXPORT
   │        │        │         │         │
   └────────┴────────┴─────────┴─────────┘
          All in one seamless interface
```

### Core Functionality

- Upload P&ID images (JPG, PNG, PDF)
- AI-powered object detection using YOLO models
- Interactive review with accept/reject/edit workflow
- Text extraction with OCR (EasyOCR)
- Export to JSON, Excel, COCO, and PDF reports

---

## 2. Tech Stack

| Component     | Technology                            | Rationale                       |
| ------------- | ------------------------------------- | ------------------------------- |
| Backend       | FastAPI (Python), use port: 8090      | Async, modern, ML-friendly      |
| Frontend      | React 18 + TypeScript                 | Type safety, ecosystem          |
| UI Components | Shadcn/ui + Radix                     | Accessible, unstyled primitives |
| Styling       | Tailwind CSS                          | Utility-first, consistent       |
| Canvas        | Konva.js (reuse code from existing\*) | Rich interactions               |
| State         | Zustand                               | Simple, scalable                |
| Data Grid     | TanStack Table + Virtual              | Virtualized for 1000+ rows      |
| Icons         | Lucide React                          | Modern, consistent              |

> **Note**: Avoid mixing styling systems. Shadcn/ui is Tailwind-native—do not add Material UI.

> **Existing Code**: Reuse Fabric.js implementation from current prototype, /static/scripts/scripts.js.

---

## 3. Application States

### 3.1 Empty State (First Load)

```
┌──────────────────────────────────────────────────────────────────┐
│  [GARNET Logo]                          [Settings] [Dark Mode]  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│                                                                  │
│      ┌─────────────────────────────────────────────────────┐     │
│      │                                                     │     │
│      │    ┌─────────────────────────────────────────┐     │     │
│      │    │                                         │     │     │
│      │    │     📄  Drop P&ID image here            │     │     │
│      │    │         or click to browse              │     │     │
│      │    │                                         │     │     │
│      │    │     Supports: JPG, PNG, PDF             │     │     │
│      │    │                                         │     │     │
│      │    └─────────────────────────────────────────┘     │     │
│      │                                                     │     │
│      │    ────────────── or ──────────────                │     │
│      │                                                     │     │
│      │    [🎯 Try with Sample P&ID]                       │     │
│      │                                                     │     │
│      └─────────────────────────────────────────────────────┘     │
│                                                                  │
│      Recent: [P&ID-001.png ×] [Process-Unit-A.jpg ×]            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Key Features**:

- Large, obvious drop zone (minimum 400x300px)
- Sample P&ID demo button for first-time users
- Recent files for quick access
- No configuration dialogs until file is selected

---

### 3.2 Preview State (File Selected)

```
┌──────────────────────────────────────────────────────────────────┐
│  [← Back]  P&ID-001.png                 [Settings] [Dark Mode]  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────┐  ┌─────────────────┐  │
│  │                                      │  │ Detection Setup │  │
│  │                                      │  ├─────────────────┤  │
│  │                                      │  │                 │  │
│  │        [P&ID Image Preview]          │  │ Model           │  │
│  │        (Zoomable thumbnail)          │  │ [YOLOv11 ▼]     │  │
│  │                                      │  │                 │  │
│  │                                      │  │ Confidence      │  │
│  │                                      │  │ [────●────] 80% │  │
│  │                                      │  │                 │  │
│  │                                      │  │ ☑ Extract text  │  │
│  │                                      │  │                 │  │
│  └──────────────────────────────────────┘  │ ───────────────│  │
│                                            │                 │  │
│                                            │ [🚀 Run Detection] │
│                                            │                 │  │
│                                            │ Advanced ▼      │  │
│                                            └─────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Key Features**:

- Image preview before processing
- Minimal required options visible
- Advanced options collapsed by default
- Clear primary action button

---

### 3.3 Processing State (Detection Running)

```
┌──────────────────────────────────────────────────────────────────┐
│  [← Cancel]  Analyzing P&ID-001.png...               [Settings] │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │    ████████████████████░░░░░░░░░░░░░░░░░░░  58%         │   │
│  │                                                          │   │
│  │    ✓ Preprocessing image                                │   │
│  │    ✓ Object detection (slice 14/24)                     │   │
│  │    ○ Text extraction (OCR)                              │   │
│  │    ○ Post-processing results                            │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Live Preview                                            │   │
│  │  ─────────────────────────────────────────────────────── │   │
│  │                                                          │   │
│  │   🔲 47 objects detected                                │   │
│  │                                                          │   │
│  │   [⬛ Gate Valve] 23    [⬛ Check Valve] 8              │   │
│  │   [⬛ Instrument] 12    [⬛ Pump] 4                     │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Estimated time remaining: ~15 seconds                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Key Features**:

- Step-by-step progress with current action
- Live object count preview
- Category breakdown updates in real-time
- Time estimate based on image size
- Cancel button to abort

---

### 3.4 Results State (Review Mode)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  P&ID-001.png  │  171 objects  │  Review: 0/171        [Export ▼] [Settings]│
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────┐  ┌────────────────┐ │
│  │                                                    │  │ ≡ Objects      │ │
│  │                                                    │  ├────────────────┤ │
│  │                                                    │  │ [🔍 Search...] │ │
│  │                                                    │  │                │ │
│  │              [ CANVAS AREA ]                       │  │ ▼ Gate Valve   │ │
│  │                                                    │  │   (41) [👁]    │ │
│  │   - Full-screen canvas                             │  │   ├─ GV-001 ✓  │ │
│  │   - Bounding boxes with confidence opacity         │  │   ├─ GV-002 ✓  │ │
│  │   - Hover shows tooltip                            │  │   └─ GV-003    │ │
│  │   - Click to select                                │  │                │ │
│  │                                                    │  │ ▶ Check Valve  │ │
│  │                                                    │  │   (8) [👁]     │ │
│  │                                                    │  │                │ │
│  │                                                    │  │ ▶ Pump (3)     │ │
│  │                                                    │  │                │ │
│  └────────────────────────────────────────────────────┘  │ ▶ Instrument   │ │
│                                                          │   Tag (25)     │ │
│  ┌─────────────────────────────────────────────────────┐ │                │ │
│  │ [−] [+] [1:1] [Fit]  ────●──── 125%    [Minimap ▼] │ │ ────────────── │ │
│  └─────────────────────────────────────────────────────┘ │ Confidence ≥   │ │
│                                                          │ [────●───] 80% │ │
│                                                          └────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key Features**:

- **Docked sidebar** (not floating) - collapsible but persistent
- **Category tree** with visibility toggles
- **Confidence filter** slider to hide low-confidence detections
- **Search** to find specific objects
- **Review counter** showing progress
- **Zoom toolbar** always visible at bottom
- **Minimap** for large P&IDs

---

### 3.5 Object Selection State

When an object is clicked:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌────────────────────────────────────────────────────┐  ┌────────────────┐ │
│  │                                                    │  │                │ │
│  │                    ┌─────────────┐                 │  │                │ │
│  │                    │   GV-042    │                 │  │                │ │
│  │        ┌───────────┼─────────────┼───────────┐     │  │                │ │
│  │        │           └─────────────┘           │     │  │                │ │
│  │        │      [Selected Object Box]          │     │  │                │ │
│  │        │       with resize handles           │     │  │                │ │
│  │        └─────────────────────────────────────┘     │  │                │ │
│  │                         │                          │  │                │ │
│  │              ┌──────────┴──────────┐               │  │                │ │
│  │              │                     │               │  │                │ │
│  │              │  Gate Valve         │               │  │                │ │
│  │              │  Confidence: 94%    │               │  │                │ │
│  │              │  OCR: "GV-042"      │               │  │                │ │
│  │              │                     │               │  │                │ │
│  │              │  [✓ Accept] [× Reject] [✎ Edit]    │               │  │                │ │
│  │              │                     │               │  │                │ │
│  │              └─────────────────────┘               │  │                │ │
│  │                                                    │  │                │ │
│  └────────────────────────────────────────────────────┘  └────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key Features**:

- **Floating detail card** near selected object
- **Accept/Reject/Edit** buttons for review workflow
- **Resize handles** for box correction
- **Class dropdown** to relabel
- **Escape** to deselect

---

## 4. Canvas Interactions

### 4.1 Mouse Interactions

| Action          | Empty Space    | On Object               |
| --------------- | -------------- | ----------------------- |
| Click           | Deselect all   | Select object           |
| Double-click    | —              | Enter edit mode         |
| Drag            | Pan canvas     | Move object (edit mode) |
| Right-click     | Context menu   | Object context menu     |
| Scroll wheel    | Zoom at cursor | Zoom at cursor          |
| Shift + Click   | —              | Add to selection        |
| Ctrl/Cmd + Drag | Box select     | Box select              |

### 4.2 Keyboard Shortcuts

| Category           | Key                  | Action                     |
| ------------------ | -------------------- | -------------------------- |
| **Navigation**     | Space + drag         | Pan mode                   |
|                    | F                    | Fit to window              |
|                    | 0                    | Zoom 100%                  |
|                    | - / =                | Zoom out / in              |
|                    | Arrow keys           | Pan canvas                 |
| **Selection**      | Tab                  | Select next object         |
|                    | Shift + Tab          | Select previous            |
|                    | Ctrl/Cmd + A         | Select all visible         |
|                    | Escape               | Deselect all               |
| **Object Actions** | Enter                | Accept selected            |
|                    | Backspace/Delete     | Reject selected            |
|                    | E                    | Edit selected              |
|                    | H                    | Hide selected              |
|                    | 1-9                  | Quick relabel to class 1-9 |
| **Global**         | Ctrl/Cmd + Z         | Undo                       |
|                    | Ctrl/Cmd + Shift + Z | Redo                       |
|                    | Ctrl/Cmd + S         | Save/Export                |
|                    | ?                    | Show keyboard shortcuts    |

### 4.3 Bounding Box Visualization

| Confidence | Border Width | Opacity | Color          |
| ---------- | ------------ | ------- | -------------- |
| 90-100%    | 3px solid    | 100%    | Category color |
| 70-89%     | 2px solid    | 80%     | Category color |
| 50-69%     | 1px dashed   | 60%     | Category color |
| < 50%      | 1px dotted   | 40%     | Gray           |

| Status     | Visual Indicator      |
| ---------- | --------------------- |
| Unreviewed | Default appearance    |
| Accepted   | Green checkmark badge |
| Rejected   | Red X, 20% opacity    |
| Edited     | Blue pencil badge     |

---

## 5. Sidebar Components

### 5.1 Object Tree

```
▼ Gate Valve (41)                    [👁 Hide All]
    ├─ GV-001  94%  ✓                [👁]
    ├─ GV-002  91%  ✓                [👁]
    ├─ GV-003  87%                   [👁]
    ├─ GV-004  82%                   [👁]
    └─ ... +37 more

▶ Check Valve (8)                    [👁]
▶ Control Valve (2)                  [👁]
▶ Instrument Tag (25)                [👁]
```

**Features**:

- Expand/collapse categories
- Confidence % shown inline
- Review status indicator (✓ / × / ✎)
- Per-item and per-category visibility toggle
- Click item to zoom to it
- Double-click to select and edit

### 5.2 Statistics Panel (Collapsible)

```
┌─────────────────────────────────┐
│ 📊 Statistics                   │
├─────────────────────────────────┤
│                                 │
│  Total Objects: 171             │
│  ──────────────────────────     │
│  ✓ Accepted: 45 (26%)          │
│  × Rejected: 3 (2%)            │
│  ○ Pending: 123 (72%)          │
│                                 │
│  Confidence Distribution:       │
│  [██████████░░░░] avg 84%      │
│                                 │
│  [View Full Report →]           │
│                                 │
└─────────────────────────────────┘
```

---

## 6. Export Options

### 6.1 Quick Export Menu

```
┌─────────────────────────────────┐
│  Export                         │
├─────────────────────────────────┤
│  📄 JSON (COCO format)         │
│  📊 Excel (with images)        │
│  📑 PDF Report                 │
│  ─────────────────────────────  │
│  ⚙ Custom export settings...   │
└─────────────────────────────────┘
```

### 6.2 Export Filters

- Only accepted objects
- Only rejected objects
- All objects
- Above confidence threshold
- Specific categories only

---

## 7. Dark Mode & Theming

### 7.1 Color Tokens

```css
/* Light Mode */
--bg-primary: #ffffff;
--bg-secondary: #f8fafc;
--bg-canvas: #e2e8f0;
--text-primary: #0f172a;
--text-secondary: #64748b;
--accent: #3b82f6;

/* Dark Mode */
--bg-primary: #0f172a;
--bg-secondary: #1e293b;
--bg-canvas: #334155;
--text-primary: #f8fafc;
--text-secondary: #94a3b8;
--accent: #60a5fa;
```

### 7.2 Category Colors (WCAG AA Contrast)

| Category      | Light Mode | Dark Mode |
| ------------- | ---------- | --------- |
| Gate Valve    | #dc2626    | #f87171   |
| Check Valve   | #2563eb    | #60a5fa   |
| Control Valve | #16a34a    | #4ade80   |
| Pump          | #9333ea    | #c084fc   |
| Instrument    | #ea580c    | #fb923c   |
| Line Number   | #0891b2    | #22d3ee   |
| ...           | ...        | ...       |

---

## 8. Detected Object Classes

| ID  | Class Name            | Icon | Has Text |
| --- | --------------------- | ---- | -------- |
| 1   | check_valve           | ⟨⟩   | No       |
| 2   | connection            | ○    | No       |
| 3   | control_valve         | ⧫    | No       |
| 4   | gate_valve            | ═╪═  | No       |
| 5   | globe_valve           | ⊙    | No       |
| 7   | instrument_dcs        | ◯    | Yes      |
| 8   | instrument_logic      | ◇    | Yes      |
| 9   | instrument_tag        | ○─   | Yes      |
| 10  | line_number           | ▭    | Yes      |
| 11  | page_connection       | ⬡    | Yes      |
| 12  | pressure_relief_valve | ⋈    | No       |
| 13  | pump                  | ◐    | No       |
| 14  | reducer               | ▷◁   | No       |
| 15  | sampling_point        | ↓    | No       |
| 16  | spectacle_blind       | ⊖    | No       |
| 17  | strainer              | ⏛    | No       |
| 20  | utility_connection    | ⬢    | Yes      |

---

## 9. API Endpoints

### 9.1 REST API

| Method | Endpoint                             | Description           |
| ------ | ------------------------------------ | --------------------- |
| GET    | `/api/health`                        | Health check          |
| GET    | `/api/models`                        | List available models |
| POST   | `/api/detect`                        | Run detection         |
| GET    | `/api/results/{id}`                  | Get detection results |
| PATCH  | `/api/results/{id}/objects/{obj_id}` | Update object         |
| DELETE | `/api/results/{id}/objects/{obj_id}` | Delete object         |
| GET    | `/api/export/{id}`                   | Export results        |

### 9.2 WebSocket (Optional - Real-time Progress)

```javascript
// Client
const ws = new WebSocket("/ws/detect/{session_id}");

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // { type: 'progress', step: 'detection', percent: 45, objects: 47 }
    // { type: 'complete', result_id: 'abc123' }
};
```

---

## 10. File Structure (New)

```
GARNET/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   ├── routes/
│   │   │   ├── detect.py       # Detection endpoints
│   │   │   ├── results.py      # Results CRUD
│   │   │   └── export.py       # Export endpoints
│   │   └── models/
│   │       ├── request.py      # Request schemas
│   │       └── response.py     # Response schemas
│   ├── core/
│   │   ├── detection.py        # YOLO + SAHI logic
│   │   ├── ocr.py              # EasyOCR wrapper
│   │   └── export.py           # Export generators
│   └── config.py               # Settings
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── canvas/
│   │   │   │   ├── Canvas.tsx
│   │   │   │   ├── BoundingBox.tsx
│   │   │   │   ├── Minimap.tsx
│   │   │   │   └── ZoomControls.tsx
│   │   │   ├── sidebar/
│   │   │   │   ├── ObjectTree.tsx
│   │   │   │   ├── Statistics.tsx
│   │   │   │   └── FilterPanel.tsx
│   │   │   ├── dialogs/
│   │   │   │   ├── UploadDialog.tsx
│   │   │   │   ├── SettingsDialog.tsx
│   │   │   │   └── ExportDialog.tsx
│   │   │   └── ui/             # Shadcn components
│   │   ├── hooks/
│   │   │   ├── useCanvas.ts
│   │   │   ├── useDetection.ts
│   │   │   └── useKeyboard.ts
│   │   ├── stores/
│   │   │   ├── detectionStore.ts
│   │   │   └── uiStore.ts
│   │   ├── lib/
│   │   │   └── api.ts          # API client
│   │   └── App.tsx
│   ├── tailwind.config.js
│   └── package.json
│
├── models/                     # YOLO weights
├── output/                     # Detection results
└── docker-compose.yml          # Dev environment
```

---

## 11. Success Metrics

| Metric                  | Target           | Measurement               |
| ----------------------- | ---------------- | ------------------------- |
| Time to first detection | < 30 seconds     | From page load to results |
| Review throughput       | > 50 objects/min | With keyboard shortcuts   |
| Error rate              | < 5%             | Incorrect accept/reject   |
| Export time             | < 3 seconds      | For 500 objects           |
| Mobile usability        | Usable           | Review on tablet          |
| Accessibility           | WCAG AA          | Keyboard-only operation   |

---

## 12. Design Inspiration

| Product                              | What to Learn                        |
| ------------------------------------ | ------------------------------------ |
| [Figma](https://figma.com)           | Canvas interactions, floating panels |
| [Label Studio](https://labelstud.io) | Annotation review workflow           |
| [Roboflow](https://roboflow.com)     | ML tool UX, progress indicators      |
| [Linear](https://linear.app)         | Keyboard-first design, minimal UI    |
| [Vercel](https://vercel.com)         | Dark mode, modern aesthetics         |

---

## 13. Implementation Priority

### Phase 1: Core (MVP)

- [ ] Empty state with drag-drop upload
- [ ] Basic detection with progress bar
- [ ] Canvas with pan/zoom
- [ ] Object list sidebar
- [ ] Basic export (JSON)

### Phase 2: Review Workflow

- [ ] Click-to-select objects
- [ ] Accept/Reject buttons
- [ ] Keyboard shortcuts (Tab, Enter, Delete)
- [ ] Review progress counter
- [ ] Undo/Redo

### Phase 3: Polish

- [ ] Dark mode
- [ ] Minimap
- [ ] Batch edit
- [ ] PDF report export
- [ ] Real-time WebSocket progress

### Phase 4: Advanced

- [ ] Annotation editing (resize, relabel)
- [ ] Create new boxes
- [ ] Comparison mode
- [ ] Batch processing
- [ ] User authentication
