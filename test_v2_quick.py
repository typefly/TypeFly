"""
Quick Test for Janah CV v2 (FaceNet)
"""

import cv2
import os
from typefly.janah_cv_v2 import janah_cv_v2

print("=" * 70)
print("🧪 Janah CV v2 - Quick Performance Test")
print("=" * 70)

# Test images
tests = {
    "Net": "tests/images/reference.jpg",
    "Aboud_big": "tests/images/Same_big1.jpg",
    "Aboud_small": "tests/images/Same_small1.jpg",
    "Azzoz": "tests/images/Azzoz2.jpg",
    "Azzoz_glasses": "tests/images/Azzoz2_glasses.jpg"
}

bbox = {'x': 0.5, 'y': 0.5, 'width': 0.8, 'height': 0.8}

# Test 1: Aboud
print("\n" + "=" * 70)
print("[Test 1] Reference: Aboud_big")
print("=" * 70)

janah_cv_v2.set_reference_photo(tests["Aboud_big"])

print(f"\n{'Image':<20} {'Score':>8} {'Status':>10}")
print("-" * 70)

for name, path in tests.items():
    if not os.path.exists(path):
        continue
    
    img = cv2.imread(path)
    score = janah_cv_v2.face_match(img, bbox)
    
    is_same = "Aboud" in name
    
    if is_same:
        status = "✅ PASS" if score >= 80 else "❌ FAIL"
    else:
        status = "✅ PASS" if score < 60 else "⚠️ HIGH"
    
    print(f"{name:<20} {score:>7}% {status:>10}")

# Test 2: Azzoz with glasses
print("\n" + "=" * 70)
print("[Test 2] Reference: Azzoz (testing glasses robustness)")
print("=" * 70)

janah_cv_v2.set_reference_photo(tests["Azzoz"])

no_glasses = janah_cv_v2.face_match(cv2.imread(tests["Azzoz"]), bbox)
with_glasses = janah_cv_v2.face_match(cv2.imread(tests["Azzoz_glasses"]), bbox)

print(f"\nWithout glasses: {no_glasses}%")
print(f"With glasses:    {with_glasses}%")
print(f"Difference:      {abs(no_glasses - with_glasses)}%")

if abs(no_glasses - with_glasses) < 15:
    print("\n✅ EXCELLENT! Robust to glasses")
elif abs(no_glasses - with_glasses) < 25:
    print("\n✅ GOOD! Moderate robustness")
else:
    print("\n⚠️ Sensitive to accessories")

print("\n" + "=" * 70)
print("✅ Test Complete!")
print("=" * 70)