import json
import os
from collections import defaultdict

def labelme_to_coco(labelme_file, image_id=1):
    """Converts a Labelme JSON file to COCO format.

    Args:
        labelme_file (str): Path to the Labelme JSON file.
        image_id (int, optional): ID of the image. Defaults to 1.

    Returns:
        dict: COCO JSON data.
    """

    # Load the Labelme JSON file
    with open(labelme_file, 'r') as f:
        labelme_data = json.load(f)

    # Initialize COCO data
    coco = {
        "images": [
            {
                "id": image_id,
                "file_name": labelme_data["imagePath"],
                "height": labelme_data["imageHeight"],
                "width": labelme_data["imageWidth"],
            }
        ],
        "annotations": [],
        "categories": [],
    }

    # Create a dictionary to store category IDs by name
    category_ids = defaultdict(lambda: len(category_ids) + 1)

    # Process shapes and create annotations
    for shape in labelme_data["shapes"]:
        label = shape["label"]
        category_id = category_ids[label]

        # Add category if it doesn't exist
        if category_id not in [cat["id"] for cat in coco["categories"]]:
            coco["categories"].append({"id": category_id, "name": label}) 
        
        # Get bounding box coordinates
        if shape["shape_type"] == "rectangle":
            xmin = min(shape["points"][0][0], shape["points"][1][0])
            ymin = min(shape["points"][0][1], shape["points"][1][1])
            xmax = max(shape["points"][0][0], shape["points"][1][0])
            ymax = max(shape["points"][0][1], shape["points"][1][1])
            width = xmax - xmin
            height = ymax - ymin

            # Create annotation
            annotation = {
                "id": len(coco["annotations"]) + 1,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, width, height],
                "segmentation": [],  # You might need to update this for polygons
                "area": width * height,
                "iscrowd": 0,
            }
            coco["annotations"].append(annotation)

    return coco

def convert_labelme_folder_to_coco(labelme_folder, coco_file):
    """Converts a folder of Labelme JSON files to a single COCO JSON file.

    Args:
        labelme_folder (str): Path to the folder containing Labelme JSON files.
        coco_file (str): Path to the output COCO JSON file.
    """

    # Initialize COCO data
    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Create a dictionary to store category IDs by name
    category_ids = defaultdict(lambda: len(category_ids) + 1)

    # Process each JSON file in the folder
    image_id = 1
    annotation_id = 1
    for filename in os.listdir(labelme_folder):
        if filename.endswith(".json"):
            labelme_file = os.path.join(labelme_folder, filename)

            # Convert the current Labelme file to COCO format
            coco_partial = labelme_to_coco(labelme_file, image_id)

            # Update image and annotation IDs
            for annotation in coco_partial["annotations"]:
                annotation["id"] = annotation_id
                annotation_id += 1

            # Merge the partial COCO data into the main COCO data
            coco["images"].extend(coco_partial["images"])
            coco["annotations"].extend(coco_partial["annotations"])

            # Update category IDs
            for category in coco_partial["categories"]:
                if category["name"] not in category_ids:
                    category_ids[category["name"]] = category["id"]
                else:
                    # If category already exists, update the IDs in the annotations
                    existing_id = category_ids[category["name"]]
                    for annotation in coco["annotations"]:
                        if annotation["category_id"] == category["id"]:
                            annotation["category_id"] = existing_id

            image_id += 1

    # Update categories with final IDs
    coco["categories"] = [{"id": id, "name": name} for name, id in category_ids.items()]

    # Save the COCO JSON data
    with open(coco_file, "w") as f:
        json.dump(coco, f, indent=4)

# Usage
labelme_folder = "sahi_slice/data"
coco_file = "sahi_slice/coco_format.json"

convert_labelme_folder_to_coco(labelme_folder, coco_file)