import cv2
import numpy as np
from typefly.janah_cv import janah_cv
import os

print("=" * 70)
print("🧪 Janah LBPH Face Recognition - Full Evaluation Test")
print("=" * 70)

# ==============================
# Images setup
# ==============================
reference_images = {
    "Net_Reference": "tests/images/reference.jpg",
    "Aboud_big": "tests/images/Same_big1.jpg",
    "Azzoz_no_glasses": "tests/images/Azzoz2.jpg"
}

all_test_images = {
    "Net_Reference": "tests/images/reference.jpg",
    "Aboud_big": "tests/images/Same_big1.jpg",
    "Aboud_small": "tests/images/Same_small1.jpg",
    "Azzoz_no_glasses": "tests/images/Azzoz2.jpg",
    "Azzoz_glasses": "tests/images/Azzoz2_glasses.jpg"
}

bbox = {'x': 0.5, 'y': 0.5, 'width': 0.8, 'height': 0.8}

# ==============================
# Face recognition tests
# ==============================
for ref_name, ref_path in reference_images.items():
    print("\n" + "=" * 70)
    print(f"[REFERENCE] Training with: {ref_name}")
    print("=" * 70)

    if not os.path.exists(ref_path):
        print(f"❌ Reference image not found: {ref_path}")
        continue

    # Train LBPH with selected reference
    janah_cv.set_reference_photo(ref_path)

    if not janah_cv.is_trained:
        print("❌ Training failed")
        continue

    print("✅ LBPH trained successfully!")

    print("\n[Test] Comparing all images against this reference")
    print("-" * 70)

    for test_name, test_path in all_test_images.items():
        img = cv2.imread(test_path)
        if img is None:
            print(f"⚠️ Image not found: {test_path}")
            continue

        score = janah_cv.face_match(img, bbox)
        print(f"{test_name:20s} → {score:3d}%")

# ==============================
# Color detection sanity test
# ==============================
print("\n" + "=" * 70)
print("[Color Detection Test]")
print("-" * 70)

img_pink = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.rectangle(img_pink, (50, 50), (590, 430), (203, 192, 255), -1)

bbox_test = {'x': 0.5, 'y': 0.5, 'width': 0.8, 'height': 0.75}
color = janah_cv.detect_clothing_color(img_pink, bbox_test)

print(f"Detected color: {color}")
if color == 'pink':
    print("✅ Color detection working!")
else:
    print("⚠️ Color detection issue")

print("\n" + "=" * 70)
print("✅ Full Evaluation Completed")
print("=" * 70)
