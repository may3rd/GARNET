import os

"""
This class contains all the settings for the scraper.
"""

class Settings:
    OUTPUT_PATH = "output"
    CROPPED_OBJECT_PATH = os.path.join(OUTPUT_PATH, "cropped object detected")
    TEXT_PATH = os.path.join(OUTPUT_PATH, "text detected")
    MODEL_PATH = "yolo_weights"
    MODEL_TYPES = [
        {"name": 'yolov8onnx', "value": 'yolov8onnx'},
        {"name": 'ultralytics', "value": 'ultralytics'},
        ]
    SYMBOL_WITH_TEXT = [
        "page connection",
        "instrument dcs",
        "instrument logic",
        "instrument tag",
        "line number",
        "utility connection",
    ]
    VERTICAL_TEXT = ["page connection", "line number"]