# Dataset

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

**Available dataset configurations:**

- `backend/datasets/yaml/data.yaml` - Default dataset configuration
- `backend/datasets/yaml/balanced.yaml` - Balanced class distribution
- `backend/datasets/yaml/iso.yaml` - ISO standard symbols
- `backend/datasets/yaml/pttep.yaml` - PTEP-specific symbols

**Class definitions:**

- `backend/datasets/classes.txt` - List of all class names
- `backend/datasets/predefined_classes.txt` - Predefined class mappings
- `backend/datasets/settings_labels.json` - Label settings configuration

Example `backend/datasets/yaml/data.yaml`:

```yaml
train: images/train
val: images/val

# Classes
names:
    0: butterfly valve
    1: check valve
    2: control valve
    3: gate valve
    4: globe valve
    5: heat exchanger
    6: instrument DCS
    7: instrument tag
    8: page connection
    9: three way valve
    10: utility connection
```