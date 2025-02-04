import albumentations as am
import cv2
import os
import random
import shutil

def augement_images(
    input_folder: str,
    output_folder: str,
    labels_folder: str | None = None,
    image_size: int = 640,
    max_file_number: int = 100,
    bbox_format='yolo',
    min_visibility=0.55
    ):

    if not os.path.isdir(input_folder):
        print('input path is not exist!')
        return

    # Set up the transform metrix
    if labels_folder is None:
        bbox_params = None
    else:
        bbox_params=am.BboxParams(format=bbox_format, min_visibility=min_visibility)

    # Crop, flip horizontally and vertically, and rotate
    transform = am.Compose([
        am.RandomCrop(width=image_size, height=image_size, p=1.0),
#       am.HorizontalFlip(p=0.5),
#       am.VerticalFlip(p=0.2),
#       am.RandomRotate90(p=0.3),
    ], bbox_params=bbox_params)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Log start of reading files
    print('Read from ', input_folder)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg', '.png')]
    image_files = sorted(image_files)

    # Log the number of input images
    print(f'Total files in {input_folder} = {len(image_files)}.')

    # Apply transformations and save the transformed images
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if labels_folder is None:
            # Create transformed images without bbox
            for idx in range(max_file_number):
                transformed = transform(image=image)
                transformed_image = transformed['image']
                output_path = os.path.join(output_folder, f"{image_file.split('.')[0]}_{idx}.png")
                cv2.imwrite(output_path, transformed_image)
        else:
            labels_file_path = os.path.join(labels_folder, image_file[:-4] + '.txt')

            with open(labels_file_path, 'r') as f:
                labels = f.readlines()

            bboxes = []
            for label in labels:
                class_id, x_center, y_center, width, height = map(
                    float, label.strip().split())
                bboxes.append([x_center, y_center, width, height, class_id])

            # Create transformed images
            idx = 0
            while idx < max_file_number:
                transformed = transform(image=image, bboxes=bboxes)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']

                # save only when transformed image has at least one object
                if len(transformed_bboxes) > 0:
                    # Save the transformed image
                    output_path = os.path.join(output_folder, f"{image_file.split('.')[0]}_{idx}.png")

                    cv2.imwrite(output_path, transformed_image)

                    # Save the annotation file
                    transformed_labels_file_path = output_path[:-4] + r".txt"
                    with open(transformed_labels_file_path, 'w') as f:
                        for bbox in transformed_bboxes:
                            f.write(
                                f"{int(bbox[4])} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                        idx = idx + 1

    # Log that all commands have been finished
    print("Transformations applied and images saved with updated labels.")


def shuffle_images(
        source_directory: str = "datasets/output",
        dist_directory: str = "datasets/test",
        split_ratio: float = 0.8,
        copy: bool = True,
    ):

    # Set your destination directories for train and validation data
    images_train_path = os.path.join(dist_directory, "images/train",)
    images_val_path = os.path.join(dist_directory, "images/valid",)
    labels_train_path = images_train_path.replace('images', 'labels')
    labels_val_path = images_val_path.replace('images', 'labels')

    # Make dir if not exist
    for path in [images_train_path, images_val_path, labels_train_path, labels_val_path]:
        if not os.path.isdir(path):
            os.makedirs(path)

    # Get a list of image files in the source directory
    image_files = [file for file in os.listdir(
        source_directory) if file.endswith(".jpg", ".png")]

    # Shuffle the list of image files randomly
    random.shuffle(image_files)

    # Calculate the number of images for training and validation
    num_train = int(len(image_files) * split_ratio)

    # Split the image files into train and validation sets
    train_files = image_files[:num_train]
    validate_files = image_files[num_train:]

    # Copy or move image and annotation files to respective directories
    if copy:
        operator = shutil.copy
    else:
        operator = shutil.move

    for file in train_files:
        image_path = os.path.join(source_directory, file)
        annotation_path = os.path.join(
            source_directory, file[:-4] + ".txt")

        operator(image_path, images_train_path)
        operator(annotation_path, labels_train_path)

    for file in validate_files:
        image_path = os.path.join(source_directory, file)
        annotation_path = os.path.join(
            source_directory, file[:-4] + ".txt")

        operator(image_path, images_val_path)
        operator(annotation_path, labels_val_path)

    print("Dataset split completed!")

