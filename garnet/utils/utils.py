import cv2
import numpy as np


def remove_circular_lines(
        image, 
        dp=1.2, 
        minDist=30, 
        param1=50, 
        param2=30, 
        minRadius=20, 
        maxRadius=100,
        inpaint_radius=3, 
        thickness=2,
        outside=True,
        background_color=(255, 255, 255)):
    """
    Detects and removes circular lines from an image using Hough Circle Transform and inpainting.
    
    :param image: Input image already read by OpenCV (BGR format).
    :param dp: Inverse ratio of the accumulator resolution to the image resolution.
    :param minDist: Minimum distance between the centers of detected circles.
    :param param1: Higher threshold for the Canny edge detector (the lower threshold is half).
    :param param2: Accumulator threshold for circle detection.
    :param minRadius: Minimum circle radius to detect.
    :param maxRadius: Maximum circle radius to detect.
    :param inpaint_radius: Radius of a circular neighborhood of each point inpainted.
    :param thichkness: The thickness of draw line.
    :param outside: Flag to keep outside circle, False to replace with background_color.
    :param background_color: The color to fill the outside area, default is white.
    :return: Image with circular lines removed.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve circle detection
    #blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # If circles are detected
    if circles is not None:
        # Convert circles to integer values
        circles = np.round(circles[0, :]).astype("int")
        
        if outside:
            result = image.copy()

            # Loop through each detected circle
            for (x, y, r) in circles:
                # Draw a filled circle (white) on the mask at the detected circle's position
                cv2.circle(result, (x, y), r, (255, 255, 255), thickness=thickness)

        else:
            # Create a mask with the same size as the image,
            # Initialized to zero (black)
            mask = np.zeros_like(image)

            x, y, r = circles[0]
        
            # Draw a filled white circle on the mask
            cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)

            # Apply the mask: keep the circle area, set outside to background color
            result = np.where(mask == 255, image, background_color).astype(np.uint8)

        return result
    else:
        print("No circles detected.")
        return image  # Return the original image if no circles are detected


def rotate_image(image, angle=90):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (nW, nH))
    return rotated
