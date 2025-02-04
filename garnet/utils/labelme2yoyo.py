import os
import json

def collect_classes_from_json(source_folder):
    """
    Collect unique class names from all JSON files in the source folder.
    Returns a sorted list of class names.
    """
    class_set = set()
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(source_folder, filename)
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            for shape in data['shapes']:
                label = shape['label'].strip().lower()  # Normalize class names
                class_set.add(label)
    
    # Sort class names alphabetically
    classes = sorted(class_set)
    return classes

def save_classes_to_file(classes, destination_folder):
    """
    Save the sorted classes to a classes.txt file in the destination folder.
    """
    classes_path = os.path.join(destination_folder, 'classes.txt')
    with open(classes_path, 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

def labelme_to_yolo(source_folder, destination_folder):
    # Collect and sort class names from JSON files
    classes = collect_classes_from_json(source_folder)
    print(f"Detected classes: {classes}")

    # Save the sorted classes to a classes.txt file
    save_classes_to_file(classes, destination_folder)

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Loop through all JSON files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(source_folder, filename)
            
            # Load LabelMe JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Get image dimensions from the JSON
            img_width = data['imageWidth']
            img_height = data['imageHeight']
            image_filename = data['imagePath']
            
            # Prepare the YOLO annotations list
            yolo_annotations = []
            for shape in data['shapes']:
                label = shape['label'].strip().lower()  # Normalize class names
                points = shape['points']

                # Convert LabelMe rectangle points to YOLO format
                if shape['shape_type'] == 'rectangle':
                    # Top-left and bottom-right points
                    x_min, y_min = points[0]
                    x_max, y_max = points[1]

                    # Correct if x_min > x_max
                    if x_min > x_max:
                        x_min, x_max = x_max, x_min
                    
                    if y_min > y_max:
                        y_min, y_max = y_max, y_min

                    # Calculate YOLO format values
                    x_center = ((x_min + x_max) / 2) / img_width
                    y_center = ((y_min + y_max) / 2) / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    # Get class ID from the sorted classes list
                    if label in classes:
                        class_id = classes.index(label)
                    else:
                        print(f"Label '{label}' not found in classes list.")
                        continue

                    # Add to YOLO annotations
                    yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

            # Save YOLO annotations to a text file
            yolo_filename = filename.replace('.json', '.txt')
            yolo_path = os.path.join(destination_folder, yolo_filename)
            with open(yolo_path, 'w') as yolo_file:
                yolo_file.write('\n'.join(yolo_annotations))

    print(f"Conversion complete. Files saved to {destination_folder}")


# Example usage:
source_folder = 'datasets/input'  # Replace with your source folder path
destination_folder = 'datasets/yolo'  # Replace with your destination folder path

labelme_to_yolo(source_folder, destination_folder)
