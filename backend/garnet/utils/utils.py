try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import torch


def rotate_image(image, angle=90):
    if cv2 is None:
        raise ImportError("opencv-python is required for rotate_image()")
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

def define_torch_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU (slower but functional)")
        
    return device