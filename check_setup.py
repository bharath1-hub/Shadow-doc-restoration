import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

img = cv2.imread("data/raw/test1.jpg")

if img is None:
    print("Image could not be loaded. Check the file path.")
else:
    print("Image loaded successfully")
    print("Shape:", img.shape)
    print("Data type:", img.dtype)
