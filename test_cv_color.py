import cv2
import numpy as np
from typefly.janah_cv import janah_cv

# Test with a sample image
print("=== Clothing Color Detection Test ===\n")

# Create a test image (pink shirt simulation)
# Or load a real image: image = cv2.imread('test_person.jpg')
image = np.zeros((480, 640, 3), dtype=np.uint8)
# Draw a pink rectangle (simulating person with pink clothes)
cv2.rectangle(image, (200, 100), (400, 400), (203, 192, 255), -1)  # BGR pink

# Test bbox (normalized coordinates)
bbox = {
    'x': 0.5,      # Center x
    'y': 0.5,      # Center y
    'width': 0.3,  # 30% of image width
    'height': 0.6  # 60% of image height
}

# Detect color
color = janah_cv.detect_clothing_color(image, bbox)
print(f"✅ Detected color: {color}")
print(f"   Expected: pink")

if color == 'pink':
    print("\n✅ TEST PASSED!")
else:
    print(f"\n⚠️ TEST FAILED: Expected 'pink', got '{color}'")

# Test with real image (optional)
# Uncomment if you have a test image:
# real_image = cv2.imread('person_photo.jpg')
# if real_image is not None:
#     real_color = janah_cv.detect_clothing_color(real_image, bbox)
#     print(f"\n✅ Real image color: {real_color}")