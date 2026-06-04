# Model Training (Optional)

To train custom YOLO models for P&ID symbols using Ultralytics:

```bash
cd backend
yolo train \
    data=datasets/yaml/data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16
cd ..
```

**Available dataset configurations:**

- `backend/datasets/yaml/data.yaml` - Default dataset configuration
- `backend/datasets/yaml/balanced.yaml` - Balanced class distribution
- `backend/datasets/yaml/iso.yaml` - ISO standard symbols
- `backend/datasets/yaml/pttep.yaml` - PTEP-specific symbols

**Training tips:**

- Use balanced datasets for better model performance
- Adjust `imgsz` based on your P&ID image resolution
- Increase epochs for better convergence (100-300 typical)
- Use data augmentation for improved generalization