from paddleocr import PaddleOCR
import os

# Initialize PaddleOCR with angle classification
# This will help in detecting text with different orientations
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

# Path to the image file in the main directory
image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pid_image.png'))

if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
else:
    print(f"Processing image: {image_path}")
    # Run OCR on the image
    result = ocr.predict(image_path)

    # Print the detected text
    if result and result[0]:
        print("Detected text:")
        for line in result[0]:
            print(f"  Text: {line[1][0]}, Confidence: {line[1][1]}")
            print(f"  Bounding box: {line[0]}")
    else:
        print("No text detected in the image.")
