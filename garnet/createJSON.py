import os
import json
import yaml
from PIL import Image

def load_categories_from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    categories = []
    
    # Add an ID to each category
    for i, category in enumerate(data["names"]):
        id = i
        name = category
        categories.append({
            "id": id,
            "name": name
        })

    return categories

def create_coco_images_json(
    image_dir: str = "test/fiftyone_test",
    output_file: str = "json/coco_images.json",
    category_file: str = "datasets/yaml/data.yaml"
    ):
    # Create an empty COCO JSON object
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Load categories from the YAML file
    categories = load_categories_from_yaml(category_file)
    
    # Populate coco categories
    for category in categories:
        coco["categories"].append({
            "id": category['id'],
            "name": category['name'],
        })

    image_id = 1

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)
            width, height = image.size

            image_info = {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
                "date_captured": "",
                "license": 0,
                "coco_url": "",
                "flickr_url": ""
            }
            coco["images"].append(image_info)
            image_id += 1

    with open(output_file, 'w') as f:
        json.dump(coco, f, indent=4)

if __name__ == '__main__':
    # Usage
    image_dir = 'test/fiftyone_test'
    output_file = 'json/coco_images.json'
    category_file = 'datasets/yaml/data.yaml'  # Your YAML file with categories
    create_coco_images_json(image_dir, output_file, category_file)
