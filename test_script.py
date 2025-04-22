import numpy as np
import cv2

# Create 300x300 image, all zeros (black), single channel (grayscale)
img = np.zeros((300, 300), dtype=np.uint8)

# Save or display the image if needed
# cv2.imwrite('black.png', img)

# Find contours (in a black image, there should be none)
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Print what we get
print(f"Number of contours found: {len(contours)}")
print("Contours:", contours)
print("Hierarchy:", hierarchy)
