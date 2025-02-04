import os

from sahi.slicing import slice_coco
from sahi.utils.file import load_json

# Perform slicing
coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="sahi_slice/PPCL_coco.json",
    image_dir="sahi_slice/data",
    output_coco_annotation_file_name="sliced_PPCL_coco.json",
    ignore_negative_samples=False,
    output_dir="sahi_slice/sliced",
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,
    verbose=True,
) # type: ignore

