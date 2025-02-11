import cv2
import numpy as np
import easyocr

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Load image
image = cv2.imread('test/!test01.png')
original_image = image.copy()  # Keep a copy for final display

# Perform OCR
ocr_result = reader.readtext(image)

# Create a mask for the text regions
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Draw bounding boxes and text on the image to show detected areas
for (bbox, text, score) in ocr_result:
    x1, y1 = bbox[0]
    x2, y2 = bbox[2]
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red boxes
    cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text
    cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), (255), -1)  # Fill with white

# Apply the mask to the grayscale image (text regions become black)
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) # Use original image here

for (bbox, text, score) in ocr_result:
    x1, y1 = bbox[0]
    x2, y2 = bbox[2]
    cv2.rectangle(gray, (int(x1), int(y1)), (int(x2), int(y2)), (255), -1)  # Fill with white

# Apply Canny edge detection on the masked grayscale image
edges = cv2.Canny(gray, 100, 200)

# Apply HoughLinesP
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=20, maxLineGap=10)

# Draw lines on the ORIGINAL image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show the result
cv2.imshow('Image with Lines', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()