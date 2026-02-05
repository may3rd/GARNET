# 🛠️ GARNET: AI-Driven P&ID Symbol Detection and Analysis

**G**CME **A**I-**R**ecognition **N**etwork for **E**ngineering **T**echnology  
_Precision in Every Connection_

[![YOLOv11](https://img.shields.io/badge/YOLOv11-💻-brightgreen)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-🖼️-orange)](https://opencv.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-📊-blue)](https://networkx.org/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-🔤-yellow)](https://github.com/JaidedAI/EasyOCR)
[![FastAPI](https://img.shields.io/badge/FastAPI-⚡-teal)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://react.dev/)

GARNET is an AI-powered tool designed to **automate symbol detection, classification, and connectivity analysis** in Piping and Instrumentation Diagrams (P&IDs). Built for engineers and maintenance teams, it combines state-of-the-art object detection (YOLOv11/YOLOv8) with graph-based analytics to transform P&ID workflows.

---

## 🚀 Features

- **Symbol Detection**: Identify valves (gate, globe, check), pumps, tanks, and more using YOLOv11/YOLOv8.
- **Automated Counting**: Generate counts for each symbol type in a P&ID.
- **Graph-Based Analysis**: Model P&IDs as networks to analyze connectivity, critical paths, and system dependencies.
- **Text Recognition (OCR)**: Extract text annotations from symbols using EasyOCR with support for vertical text.
- **Export Results**: Export detection results and connectivity graphs to CSV, PDF, Excel, or JSON.
- **Interactive Review UI**: Modern React-based interface for reviewing, editing, and validating detections.
- **Batch Processing**: Process multiple P&ID images in a single run with progress tracking.

---

## 📐 Architecture

### System Overview

GARNET follows a modern client-server architecture with a clear separation between the interactive frontend and the AI-powered backend:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GARNET Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────┐          ┌─────────────────────────────┐ │
│  │      React Frontend          │          │      FastAPI Backend        │ │
│  │  ┌────────────────────────┐  │          │  ┌───────────────────────┐  │ │
│  │  │  React 18 + TypeScript │  │◄────────►│  │  FastAPI              │  │ │
│  │  │  Zustand (State)       │  │   HTTP   │  │  SAHI (Slicing)       │  │ │
│  │  │  Tailwind CSS + Radix  │  │          │  │  Ultralytics (YOLO)   │  │ │
│  │  │  Vite (Build Tool)     │  │          │  │  EasyOCR (Text)       │  │ │
│  │  └────────────────────────┘  │          │  │  OpenCV (Image Proc)  │  │ │
│  │                              │          │  └───────────────────────┘  │ │
│  │  Port: 5173 (dev) / 80/443   │          │  Port: 8001                 │ │
│  frontend/                   │          │  backend/main.py            │ │
│  └──────────────────────────────┘          └─────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Frontend Stack

| Technology       | Version | Purpose                                         |
| ---------------- | ------- | ----------------------------------------------- |
| **React**        | 18.3.1  | UI library with hooks and functional components |
| **TypeScript**   | 5.7.3   | Type-safe development                           |
| **Vite**         | 6.0.7   | Fast development server and optimized builds    |
| **Zustand**      | 5.0.3   | Lightweight state management                    |
| **Tailwind CSS** | 3.4.17  | Utility-first CSS framework                     |
| **Radix UI**     | Latest  | Accessible, unstyled UI primitives              |
| **Lucide React** | Latest  | Modern icon library                             |
| **jsPDF**        | 4.0.0   | PDF export functionality                        |

#### Frontend Structure

```
frontend/
├── src/
│   ├── components/           # React components
│   │   ├── ui/              # Reusable UI primitives (Radix-based)
│   │   ├── App.tsx          # Main application component
│   │   ├── UploadZone.tsx   # Image upload with drag-and-drop
│   │   ├── DetectionSetup.tsx  # Detection parameter configuration
│   │   ├── ResultsView.tsx  # Detection results with interactive canvas
│   │   ├── BatchResultsView.tsx  # Batch processing interface
│   │   └── ...
│   ├── stores/
│   │   ├── appStore.ts      # Main application state (Zustand)
│   │   └── historyStore.ts  # Undo/redo action history
│   ├── lib/
│   │   ├── api.ts           # API client functions
│   │   ├── pdfExport.ts     # PDF generation utilities
│   │   └── exportFormats.ts # CSV/JSON export utilities
│   ├── hooks/               # Custom React hooks
│   ├── types.ts             # TypeScript type definitions
│   └── styles/              # Global styles and CSS variables
├── package.json
├── vite.config.ts           # Vite configuration with API proxy
└── tsconfig.json
```

### Backend Stack

| Technology           | Purpose                                        |
| -------------------- | ---------------------------------------------- |
| **FastAPI**          | High-performance Python web framework          |
| **SAHI**             | Slicing Aided Hyper Inference for large images |
| **Ultralytics**      | YOLOv11/YOLOv8 object detection models         |
| **EasyOCR**          | Text recognition from detected symbols         |
| **OpenCV**           | Image processing and manipulation              |
| **NetworkX**         | Graph construction and connectivity analysis   |
| **Pydantic**         | Data validation and settings management        |
| **python-multipart** | File upload handling                           |

#### Backend Structure

```
backend/
├── main.py                  # Main FastAPI application entry point
├── api.py                   # Alternative API service (optimized)
├── requirements.txt         # Python dependencies
├── .env                     # Environment configuration
├── garnet/                  # Core logic package
├── datasets/                # Dataset configuration files
├── yolo_weights/            # Model weights
└── static/                  # Static assets
    └── images/predictions/  # Generated prediction images
```

### Data Flow

```
1. Upload P&ID Image
        │
        ▼
2. Frontend → POST /api/detect
   (with detection parameters)
        │
        ▼
3. Backend: SAHI Slicing
   ├─ Slice large image into tiles
   ├─ Run YOLO inference on each tile
   └─ Merge overlapping detections (NMM)
        │
        ▼
4. Backend: OCR (optional)
   └─ Extract text from symbol regions
        │
        ▼
5. Backend: Response
   └─ JSON with detections + image URL
        │
        ▼
6. Frontend: ResultsView
   ├─ Interactive canvas with annotations
   ├─ Object list with editing capabilities
   └─ Export options (CSV, PDF, JSON)
```

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- Node.js 18+ (for frontend)
- Bun or npm (package manager)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/may3rd/GARNET.git
cd GARNET
```

### 2. Install DeepLSD (Optional - for line extraction)

```bash
git clone --recurse-submodules https://github.com/cvg/DeepLSD.git
cd DeepLSD
bash quickstart_install.sh
cd ..
```

### 3. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 4. Install Frontend Dependencies

```bash
cd frontend
bun install
# or: npm install
cd ..
```

---

## 🖥️ Usage

### 1. React Frontend + API Backend (Recommended)

This is the primary mode for interactive P&ID analysis. The React frontend provides a modern UI for uploading images, configuring detection parameters, reviewing results, and exporting data.

#### Tech Stack Summary

| Component  | Technology         | Version |
| ---------- | ------------------ | ------- |
| Frontend   | React + TypeScript | 18.3.1  |
| Build Tool | Vite               | 6.0.7   |
| Styling    | Tailwind CSS       | 3.4.17  |
| State      | Zustand            | 5.0.3   |
| Backend    | FastAPI            | Latest  |
| AI Engine  | SAHI + Ultralytics | Latest  |

#### Quick Start

**Terminal 1 - Start the API backend:**

```bash
# Copy environment file and configure
cp .env.example .env

# Start FastAPI server
cd backend
uvicorn main:app --reload --port 8001
```

Backend runs at `http://localhost:8001` with auto-reload enabled for development.

**Terminal 2 - Start the React frontend:**

```bash
cd frontend

# Copy environment file and configure
cp .env.example .env.local

# Start development server
bun run dev
# or: npm run dev
```

Frontend runs at `http://localhost:5173` with hot module replacement.

#### Environment Variables

**Backend Configuration (`.env`):**

```bash
# Environment (development, production)
ENV=development
DEBUG=true

# Server Configuration
HOST=localhost
PORT=8001

# CORS - Comma-separated list of allowed origins
CORS_ORIGINS=http://localhost:5173,http://localhost:8001

# File Upload Limits
MAX_FILE_SIZE_MB=50
ALLOWED_IMAGE_EXTENSIONS=.jpg,.jpeg,.png,.webp,.bmp,.tiff

# Model Defaults
DEFAULT_CONF_THRESHOLD=0.8
DEFAULT_IMAGE_SIZE=640
DEFAULT_OVERLAP_RATIO=0.2

# Cache Configuration
RESULTS_CACHE_MAX_SIZE=100
RESULTS_CACHE_TTL=3600
MODEL_CACHE_MAX_SIZE=10

# OCR Configuration
OCR_CACHE_ENABLED=true
OCR_LANGUAGES=en
OCR_GPU=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=garnet.log
```

**Frontend Configuration (`frontend/.env.local`):**

```bash
# API Configuration
VITE_API_URL=http://localhost:8001

# Development Server
VITE_PORT=5173
VITE_HOST=localhost

# Build Configuration
VITE_SOURCEMAP=false
VITE_OUT_DIR=dist
```

#### Configuration Options

The Detection Setup panel in the frontend provides the following options:

| Parameter                | Description                              | Default                   | Range         |
| ------------------------ | ---------------------------------------- | ------------------------- | ------------- |
| **Model**                | Detection model type                     | `ultralytics`             | `ultralytics` |
| **Weight File**          | Path to model weights (`.pt` or `.onnx`) | Auto-selected             | -             |
| **Config File**          | YAML dataset configuration               | `datasets/yaml/data.yaml` | -             |
| **Confidence Threshold** | Minimum detection confidence             | `0.8`                     | `0.0 - 1.0`   |
| **Image Size**           | Input size for model inference           | `640`                     | `320 - 2048`  |
| **Overlap Ratio**        | SAHI slice overlap for large images      | `0.2`                     | `0.0 - 0.95`  |
| **Text OCR**             | Enable text extraction from symbols      | `false`                   | `true/false`  |

#### Production Deployment

**Build the frontend for production:**

```bash
cd frontend
bun run build
# or: npm run build
```

This creates an optimized build in the `frontend/dist/` directory.

**Production deployment options:**

1. **Separate Services (Recommended)**
    - Serve frontend via nginx/Apache or CDN
    - Run backend with production ASGI server (Uvicorn + Gunicorn)

    ```bash
    # Backend production start
    cd backend
    gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
    ```

2. **Combined Serving**
    - Mount the built frontend static files in FastAPI:

    ```python
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
    ```

**Production Environment Variables:**

```bash
# Backend
ENV=production
DEBUG=false
HOST=0.0.0.0
PORT=8001
CORS_ORIGINS=https://yourdomain.com

# Frontend (build time)
VITE_API_URL=https://api.yourdomain.com
VITE_SOURCEMAP=false
```

---

### 2. Batch Inference Script

Run inference on multiple P&IDs in a folder using `predict_images.py`:

**Command-Line Arguments:**

```bash
python predict_images.py \
    --image_path path/to/pids_folder \
    --model_type yolov8 \
    --model_path path/to/model_weights.pt \
    --output_path results/
```

**Output:**

- Annotated images (saved in output_path).
- CSV file with symbol counts (`output_path/symbol_counts.csv`).

Example code snippet:

```python
from garnet.inference import predict_images

predict_images(
    image_path="path/to/pids_folder",
    model_type="yolov8",
    model_path="path/to/model_weights.pt",
    output_path="results/"
)
```

---

### 3. Pipeline: End-to-End P&ID Digitization and Connectivity Analysis

Based on our reference methodology, the following pipeline will be implemented:

1. **Preprocessing**
    - Load and normalize the P&ID image.
    - Optional: Denoising, binarization, and removal of scanning artifacts.

2. **Object Detection**
    - Detect P&ID symbols using YOLOv11/YOLOv8.
    - Classes include valves, pumps, tanks, instruments, and custom symbols.
    - Export detections as bounding boxes (YOLO or COCO format).

3. **Text Recognition**
    - Use EasyOCR or PaddleOCR for text extraction (supports horizontal and vertical text).
    - Merge OCR results from multiple rotations.
    - Output text annotations (JSON).

4. **Line Extraction**
    - Remove detected symbols and text areas from the image to isolate pipelines.
    - Apply morphological operations to extract lines (handles 90° turns and intersections).
    - Merge fragmented line segments.

5. **Line-Symbol Connection**
    - Detect intersection points between line segments and symbol bounding boxes.
    - Assign connectivity relationships between symbols and lines.

6. **Graph Construction**
    - Convert connected symbols and pipelines into a NetworkX graph.
    - Nodes = symbols (with attributes like tag, type).
    - Edges = pipelines (with attributes like length, type).

7. **Graph Analysis**
    - Perform connectivity analysis: critical paths, loops, flow direction inference.
    - Detect isolated components or redundant loops.

8. **DEXPI Export**
    - Convert the annotated P&ID into DEXPI-compliant XML.
    - Include geometry, symbol metadata, and connectivity.

9. **Visualization and Reporting**
    - Overlay detected objects, text, and pipelines on the original P&ID.
    - Export results as PNG/PDF/JSON.
    - Provide interactive graph visualization (future feature).

**Note:** This pipeline is designed to be modular, so each step can be run independently or as part of the full digitization workflow.

_Reference:_ Adapted from "End-to-End Digitization of Image Format Piping and Instrumentation Diagrams" (2024).

Note: A staged, runnable pipeline with CLI is available in `garnet/pid_extractor.py`. See `garnet/README_pid_pipeline.md` for usage, stages, tuning knobs, and output artifacts.

---

### 4. Model Training (Optional)

To train custom YOLO models for P&ID symbols:

```bash
yolo train \
    data=data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16
```

## 📂 Dataset

GARNET uses the YOLOv8 dataset format. Example structure:

```
dataset/
├── train/
│   ├── images/  # P&ID images (.jpg, .png)
│   └── labels/  # YOLO-format labels (.txt)
├── val/
│   ├── images/
│   └── labels/
└── data.yaml     # Dataset config (class names, paths)
```

Example `data.yaml`:

```yaml
train: dataset/train/images
val: dataset/val/images

nc: 6 # Number of classes
names: ["valve", "gate_valve", "globe_valve", "check_valve", "pump", "tank"]
```

---

## 📊 Results

| **Detection**                                     | **Graph Analysis**                               |
| ------------------------------------------------- | ------------------------------------------------ |
| ![Detected Symbols](assets/detection_example.jpg) | ![Graph Visualization](assets/graph_example.png) |

_Example output: Symbol counts and connectivity graph for a P&ID._

---

## 📈 Future Outcomes

Additional planned outcomes from the GARNET project include:

- **MTO for Valves**: Automated generation of material take-off lists for all detected valve types, including specifications and quantities.
- **Line List**: Extraction and tabulation of pipeline data, including line tags, sizes, service, and connected equipment.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.  
For major changes, open an issue first to discuss your ideas.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📧 Contact

For questions or collaborations, contact:

- **Your Name** - [may3rd@gmail.com](mailto:may3rd@gmail.com)
- **GCME (GC Maintenance and Engineering Co., Ltd.)** - [www.gcme.com](https://www.gcme.com)
