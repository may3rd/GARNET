import torch
from torch.utils.data import Dataset
from torchvision import datasets
import matplotlib.pyplot as plt
import yaml

import os
import pandas as pd
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import datapoints as dp
from torchvision.transforms.v2 import functional as F

# myCustomDataset
class myDataset(Dataset):
    def __init__(self, data, transform=None):

        # Load dataset file
        with open(data) as file_stream:
            # Load using yaml utils
            dict = yaml.load(file_stream, Loader=yaml.FullLoader)

            # If no path in dataset.yaml then set path = dataset path
            if "path" in dict:
                self.root = dict["path"]
            else:
                self.root = os.path.dirname(data)

            if "train" in dict:
                self.images_train_path = dict["train"]
            else:
                self.images_train_path = "images/train"

            if "val" in dict:
                self.images_val_path = dict["val"]
            elif "validata" in dict:
                self.images_val_path = dict["validate"]
            else:
                self.images_val_path = "images/val"

            if "names" in dict:
                self.names = dict["names"]

        self.images_train = [f for f in os.listdir(os.path.join(self.root, self.images_train_path)) if f.endswith(('.jpg','.png'))]
        self.images_val = [f for f in os.listdir(os.path.join(self.root, self.images_val_path)) if f.endswith(('.jpg','.png'))]

        self.transform = transform

    def __len__(self):
        return len(self.images_train)
    
    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.root, self.images_train_path, self.images_train[idx])
        label_path = image_path.replace("images/", "labels/").replace(".jpg", ".txt")
        image = read_image(image_path)
        
        image_size = image.shape  # torch image: C x H x W
        image_width = image_size[2]
        image_height = image_size[1]

        annotations = []
        boxes = []
        labels = []
        image_id = torch.tensor([idx])
        area = []
        iscrowd = []

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            center_x, center_y, box_width, box_height = map(float, parts[1:5])

            # Convert normalized coordinates to pixel values
            left = int((center_x - box_width / 2) * image_width)
            top = int((center_y - box_height / 2) * image_height)
            width = int(box_width * image_width)
            height = int(box_height * image_height)

            # Optionally map class IDs to class names
            class_name = self.names.get(class_id, str(class_id)) if self.names else str(class_id)

            boxes.append(dp.BoundingBox([left, top, left + width, top + height], format=datapoints.BoundingBoxFormat.XYXY, spatial_size=image.shape[-2:])) # type: ignore
            labels.append(torch.tensor(class_id))
            area.append(torch.tensor(float(width * height)))
            iscrowd.append(torch.zeros((1,), dtype=torch.int64))

            annotations.append({
                'class_id': class_id,
                'class_name': class_name,
                'image_path': image_path,
                'label_path': label_path,
                'image_width': image_width,
                'image_height': image_height,
                'top': top,
                'left': left,
                'width': width,
                'height': height
            })
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        sample = {'image': image, 'annotations': annotations}

        if self.transform:
            sample = self.transform(sample)
            image, target = self.transform(image, target)

        return image, target
