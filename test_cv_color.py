import cv2
import numpy as np
from typefly.janah_cv import janah_cv

print("="*70)
print("=== Clothing Color Detection Test ===")
print("="*70)

# Test 1: Simulated pink clothing
print("\n[Test 1] Simulated Pink Clothing")
print("-" * 70)

# Create a white background image
image = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Draw a large pink rectangle (simulating person with pink clothes)
# BGR format: (B, G, R) - Pink is high R and G, medium B
cv2.rectangle(image, (50, 50), (590, 430), (203, 192, 255), -1)  # BGR pink

# Test bbox (normalized coordinates - covering most of the image)
bbox = {
    'x': 0.5,      # Center x
    'y': 0.5,      # Center y
    'width': 0.8,  # 80% of image width
    'height': 0.75 # 75% of image height
}

# Detect color
color = janah_cv.detect_clothing_color(image, bbox)
print(f"✅ Detected color: {color}")
print(f"   Expected: pink")

if color == 'pink':
    print("   ✅ TEST PASSED!")
else:
    print(f"   ⚠️ TEST FAILED: Expected 'pink', got '{color}'")

# Show the test image
cv2.imshow("Test Image - Pink Clothing", image)
print("\nPress any key to continue to next test...")
cv2.waitKey(0)

# Test 2: Blue clothing
print("\n[Test 2] Simulated Blue Clothing")
print("-" * 70)

image_blue = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.rectangle(image_blue, (50, 50), (590, 430), (255, 100, 100), -1)  # BGR blue

color_blue = janah_cv.detect_clothing_color(image_blue, bbox)
print(f"✅ Detected color: {color_blue}")
print(f"   Expected: blue")

if color_blue == 'blue':
    print("   ✅ TEST PASSED!")
else:
    print(f"   ⚠️ TEST FAILED: Expected 'blue', got '{color_blue}'")

cv2.imshow("Test Image - Blue Clothing", image_blue)
print("\nPress any key to continue to next test...")
cv2.waitKey(0)

# Test 3: Red clothing
print("\n[Test 3] Simulated Red Clothing")
print("-" * 70)

image_red = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.rectangle(image_red, (50, 50), (590, 430), (50, 50, 255), -1)  # BGR red

color_red = janah_cv.detect_clothing_color(image_red, bbox)
print(f"✅ Detected color: {color_red}")
print(f"   Expected: red")

if color_red == 'red':
    print("   ✅ TEST PASSED!")
else:
    print(f"   ⚠️ TEST FAILED: Expected 'red', got '{color_red}'")

cv2.imshow("Test Image - Red Clothing", image_red)
print("\nPress any key to continue...")
cv2.waitKey(0)

# Test 4: Green clothing
print("\n[Test 4] Simulated Green Clothing")
print("-" * 70)

image_green = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.rectangle(image_green, (50, 50), (590, 430), (100, 255, 100), -1)  # BGR green

color_green = janah_cv.detect_clothing_color(image_green, bbox)
print(f"✅ Detected color: {color_green}")
print(f"   Expected: green")

if color_green == 'green':
    print("   ✅ TEST PASSED!")
else:
    print(f"   ⚠️ TEST FAILED: Expected 'green', got '{color_green}'")

cv2.imshow("Test Image - Green Clothing", image_green)
print("\nPress any key to close...")
cv2.waitKey(0)

cv2.destroyAllWindows()

# Test 5: With real image (optional)
print("\n[Test 5] Real Image Test (Optional)")
print("-" * 70)
print("Place a test image at: test_photos/person.jpg")
print("The image should contain a person wearing colored clothing")

try:
    real_image = cv2.imread('test_photos/person.jpg')
    if real_image is not None:
        # Bbox covering center of image
        real_bbox = {
            'x': 0.5,
            'y': 0.5,
            'width': 0.6,
            'height': 0.8
        }
        
        real_color = janah_cv.detect_clothing_color(real_image, real_bbox)
        print(f"✅ Detected color from real image: {real_color}")
        
        # Draw bbox on image
        h, w = real_image.shape[:2]
        x1 = int((real_bbox['x'] - real_bbox['width']/2) * w)
        y1 = int((real_bbox['y'] - real_bbox['height']/2) * h)
        x2 = int((real_bbox['x'] + real_bbox['width']/2) * w)
        y2 = int((real_bbox['y'] + real_bbox['height']/2) * h)
        cv2.rectangle(real_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        cv2.imshow("Real Image Test", real_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("   ⚠️ No test image found - skipping real image test")
        print("   To test with real image:")
        print("   1. Create folder: mkdir test_photos")
        print("   2. Add image: test_photos/person.jpg")
except Exception as e:
    print(f"   ⚠️ Error loading real image: {e}")

print("\n" + "="*70)
print("=== All Tests Completed ===")
print("="*70)

# Age estimation test
print("\n[Bonus Test] Age Estimation from Bbox Size")
print("-" * 70)

test_cases = [
    {'height': 0.25, 'expected': '3-5'},
    {'height': 0.35, 'expected': '6-8'},
    {'height': 0.50, 'expected': '9-12'},
    {'height': 0.70, 'expected': '13+'}
]

for i, test in enumerate(test_cases, 1):
    bbox_age = {'x': 0.5, 'y': 0.5, 'width': 0.3, 'height': test['height']}
    age = janah_cv.estimate_age_from_size(bbox_age)
    status = "✅" if age == test['expected'] else "⚠️"
    print(f"{status} Height {test['height']}: Estimated age: {age} (Expected: {test['expected']})")

print("\n" + "="*70)
print("✅ All tests completed!")
print("="*70)