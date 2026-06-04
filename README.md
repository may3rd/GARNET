# 🛠️ GARNET: AI-Driven P&ID Symbol Detection and Analysis

**G**CME **A**I-**R**ecognition **N**etwork for **E**ngineering **T**echnology  
_Precision in Every Connection_

[![YOLOv11](https://img.shields.io/badge/YOLOv11-💻-brightgreen)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-🖼️-orange)](https://opencv.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-📊-blue)](https://networkx.org/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-🔤-yellow)](https://github.com/JaidedAI/EasyOCR)
[![FastAPI](https://img.shields.io/badge/FastAPI-⚡-teal)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3.1-61DAFB)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9.3-3178C6)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-6.4.3-646CFF)](https://vitejs.dev/)

GARNET is an AI-powered tool designed to **automate symbol detection, classification, and connectivity analysis** in Piping and Instrumentation Diagrams (P&IDs). Built for engineers and maintenance teams, it combines state-of-the-art object detection (YOLOv11/YOLOv8) with graph-based analytics to transform P&ID workflows.

## 🚀 Features

### Core Capabilities
- **Symbol Detection**: Identify valves (gate, globe, check), pumps, tanks, and more using YOLOv11/YOLOv8
- **SAHI Integration**: Slicing Aided Hyper Inference for accurate detection on large images
- **Text Recognition (OCR)**: Extract text annotations from symbols using multiple OCR engines
- **Graph Construction**: Build connectivity graphs from detected symbols and pipes
- **Interactive Review**: Human-in-the-loop validation and correction of results
- **Export Options**: Results available in JSON, YOLO, COCO, LabelMe, or PDF formats

### Frontend Features
- Interactive canvas with pan/zoom and minimap support
- Object detection visualization with color-coded bounding boxes
- Object editing (create, update, delete) directly on canvas
- Review workflow with accept/reject and visual status indicators
- Batch processing with queue management and progress tracking
- Undo/redo support for all editing operations
- Keyboard shortcuts for efficient navigation
- Dark/light theme toggle
- Confidence filtering and class visibility controls

### Backend Features
- Automated counting and symbol type classification
- Model caching for faster subsequent detections
- Results caching with TTL for improved performance
- Automatic cleanup of temporary files
- Health monitoring endpoint
- Environment-configurable for development and production

## 📦 Installation

### Prerequisites
- Python 3.9+
- Node.js 18+ (for frontend)
- Bun or npm (package manager)
- Git

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/may3rd/GARNET.git
   cd GARNET
   ```

2. **Install Dependencies**
   ```bash
   # Backend
   cd backend
   pip install -r requirements.txt
   cd ..
   
   # Frontend
   cd frontend
   bun install  # or: npm install
   cd ..
   ```

3. **Configure Environment**
   ```bash
   # Backend
   cp .env.example .env
   # Edit .env as needed
   
   # Frontend
   cd frontend
   cp .env.example .env.local
   # Edit .env.local as needed
   cd ..
   ```

4. **Start the Application**
   ```bash
   # Terminal 1 - Start API backend
   cd backend
   uvicorn api:app --reload --port 8001
   
   # Terminal 2 - Start React frontend
   cd frontend
   bun run dev  # or: npm run dev
   ```

## 🖥️ Usage

### 1. React Frontend + API Backend (Recommended)

This is the primary mode for interactive P&ID analysis. The React frontend provides a modern UI for uploading images, configuring detection parameters, reviewing results, and exporting data.

#### Tech Stack Summary
| Component  | Technology         | Version     |
| ---------- | ------------------ | ----------- |
| Frontend   | React + TypeScript | 18.3.1      |
| Build Tool | Vite               | 6.4.3       |
| Styling    | Tailwind CSS       | 3.4.19      |
| State      | Zustand            | 5.0.14      |
| Backend    | FastAPI            | Latest      |
| AI Engine  | SAHI + Ultralytics | Latest      |

#### Detection vs Pipeline Modes
- **Detection Mode**: Symbol detection and OCR with interactive review
- **Pipeline Mode**: End-to-end P&ID digitization including connectivity analysis

### 2. Batch Inference Script

Run inference on multiple P&IDs using `garnet/predict_images.py`:
```bash
python garnet/predict_images.py \
    --image_path path/to/pids_folder \
    --model_type yolov8 \
    --model_path path/to/model_weights.pt \
    --output_path results/
```

### 3. Pipeline: End-to-End P&ID Digitization

Comprehensive P&ID analysis pipeline available in `garnet/pid_extractor.py`:
```bash
python garnet/pid_extractor.py \
    --image path/to/pid_image.png \
    --out output/ \
    --stop-after 11
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:
- [Frontend Structure](docs/frontend-structure.md)
- [Backend Structure](docs/backend-structure.md)
- [API Endpoints](docs/api-endpoints.md)
- [Data Flow](docs/data-flow.md)
- [Pipeline Stages](docs/pipeline.md)
- [Environment Variables](docs/environment-variables.md)
- [Model Training](docs/model-training.md)
- [Dataset Format](docs/dataset.md)

## 📈 Future Outcomes

Additional planned outcomes from the GARNET project include:
- **MTO for Valves**: Automated generation of material take-off lists for all detected valve types
- **Line List**: Extraction and tabulation of pipeline data including line tags, sizes, service, and connected equipment

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss your ideas.

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 📧 Contact

For questions or collaborations, contact:
- **Maetee Lorprajuksiri** - [may3rd@gmail.com](mailto:may3rd@gmail.com)
- **GCME (GC Maintenance and Engineering Co., Ltd.)** - [www.gcme.com](https://www.gcme.com)