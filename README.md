# 🛠️ GARNET: AI-Driven P&ID Symbol Detection and Analysis

**G**CME **A**I-**R**ecognition **N**etwork for **E**ngineering **T**echnology  
_Precision in Every Connection_

[![YOLOv8](https://img.shields.io/badge/YOLOv8-💻-brightgreen)](https://ultralytics.com/yolov8)
[![OpenCV](https://img.shields.io/badge/OpenCV-🖼️-orange)](https://opencv.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-📊-blue)](https://networkx.org/)

GARNET is an AI-powered tool designed to **automate symbol detection, classification, and connectivity analysis** in Piping and Instrumentation Diagrams (P&IDs). Built for engineers and maintenance teams, it combines state-of-the-art object detection (YOLOv8) with graph-based analytics to transform P&ID workflows.

---

## 🚀 Features

-   **Symbol Detection**: Identify valves (gate, globe, check), pumps, tanks, and more using YOLOv8.
-   **Automated Counting**: Generate counts for each symbol type in a P&ID.
-   **Graph-Based Analysis**: Model P&IDs as networks to analyze connectivity, critical paths, and system dependencies.
-   **Export Results**: Export detection results and connectivity graphs to CSV, PDF, or JSON.
-   **User-Friendly Interface**: Simple CLI and scriptable API for integration into existing workflows.

---

## 📦 Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/your-username/GARNET.git
    cd GARNET
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🖥️ Usage

### 1. **Web Application (Interactive Inference)**

Run the web interface to upload P&IDs, select models, and view results in real time:

1. **Start the Web Server**:

    ```bash
    uvicorn main:app --reload
    ```

    Access the app at `http://localhost:8000`.

2. **Using the Web Interface**:
    - **Upload a P&ID**: Select an image file (JPG/PNG).
    - **Model Configuration**: Choose a model type (e.g., YOLOv5, YOLOv8) and upload custom weights (`.pt` file).
    - **Run Inference**: Click "Submit" to detect symbols and display results.
    - **Results**: View annotated images, symbol counts, and download reports (CSV/JSON).

![Web Interface Demo](assets/web_demo.gif)

---

### 2. **Batch Inference Script**

Run inference on multiple P&IDs in a folder using `predict_images.py`:

1. **Command-Line Arguments**:

    ```bash
    python predict_images.py \
        --image_path path/to/pids_folder \
        --model_type yolov8 \
        --model_path path/to/model_weights.pt \
        --output_path results/
    ```

2. **Output**:
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

### 3. **Todo: Graph-Based Connectivity Analysis** _(Coming Soon!)_

-   **Feature**: Automatically generate connectivity graphs from P&IDs.
-   **Planned Workflow**:
    1. Detect symbols and pipelines.
    2. Build a graph network using NetworkX.
    3. Analyze critical paths, cycles, and dependencies.
    4. Export graphs as PNG/PDF or integrate with CAD tools.

---

### 4. **Model Training (Optional)**

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

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.  
For major changes, open an issue first to discuss your ideas.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📧 Contact

For questions or collaborations, contact:

-   **Your Name** - [may3rd@gmail.com](mailto:may3rd@gmail.com)
-   **GCME (GC Maintenance and Engineering Co., Ltd.)** - [www.gcme.com](https://www.gcme.com)
