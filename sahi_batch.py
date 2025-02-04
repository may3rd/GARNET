from sahi.predict import predict, get_sliced_prediction
from sahi import AutoDetectionModel
import os
import json
from collections import defaultdict

# Set the selected model and weight file
selected_model = "ultralytics"
weight_file = "yolo_weights/yolov8n_PTTEP_running.pt"
config_file = "datasets/yaml/pttep.yaml"
path_to_images = "PTTEP/ER"
dataset_json_path = "coco_images2.json"
name = "ER"
project = "runs/PTTEP"

# Set category_mapping for ONNX model, required by updated version of SAHI
if "yolov8onnx" == selected_model:
    import onnx
    import ast
    model = onnx.load(weight_file)
    props = { p.key: p.value for p in model.metadata_props }
    names = ast.literal_eval(props['names'])
    category_mapping = { str(key): value for key, value in names.items() }
else:
    category_mapping = None

# confidence threshold
conf_th = 0.8

# device
device = "cpu"

# overlap ratio for sliced prediction
overlap_ratio = 0.5

# init a model
detection_model = AutoDetectionModel.from_pretrained(
        model_type=selected_model,
        model_path=weight_file,
        config_path=config_file,
        confidence_threshold=conf_th,
        category_mapping=category_mapping,
        device=device,
    )

# get batch predict result
result = predict(
    detection_model=detection_model,
    name=name,
    project=project,
    source=path_to_images, # image or folder path
    dataset_json_path=dataset_json_path, # for mmdet models
    no_standard_prediction=True,
    no_sliced_prediction=False,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=overlap_ratio, 
    overlap_width_ratio=overlap_ratio,
    export_pickle=False,
    export_crop=False,
    visual_bbox_thickness=5,
    visual_text_size=0.5,
    visual_text_thickness=2,
    visual_hide_labels=True,
    visual_hide_conf=True,
    visual_export_format="png",
    return_dict=True,
)

# load the result json file
with open(os.path.join(result['export_dir'], "result.json"), "r") as f:
    data = json.load(f)

# Initialize a dictionary to store the counts
category_counts = defaultdict(lambda: defaultdict(int))

# Iterate through the data and count the categories for each image
for item in data:
    image_id = item["image_id"]
    category_name = item["category_name"]
    category_counts[image_id][category_name] += 1

# Print the results
for image_id, counts in category_counts.items():
    print(f"Image ID: {image_id}")
    for category_name, count in counts.items():
        print(f"  {category_name}: {count}")
