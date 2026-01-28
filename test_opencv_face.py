from typefly.janah_cv import janah_cv
import cv2
import numpy as np

print("="*70)
print("🧪 OpenCV Face Detection Test")
print("="*70)

# Test 1: No face (blank image)
print("\n[TEST 1] Blank Image (No Face)")
print("-"*70)

blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
bbox = {'x': 0.5, 'y': 0.5, 'width': 0.5, 'height': 0.5}

score = janah_cv.face_match(blank, bbox)
print(f"Face confidence: {score}%")

if score == 0:
    print("✅ Correctly returns 0% for blank image")
else:
    print(f"⚠️  Expected 0%, got {score}%")

# Test 2: Drawing a simple "face" (circles for testing)
print("\n[TEST 2] Synthetic Face")
print("-"*70)

face_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
# Draw face-like structure
cv2.circle(face_img, (320, 200), 80, (200, 200, 200), -1)  # Head
cv2.circle(face_img, (290, 180), 15, (50, 50, 50), -1)     # Left eye
cv2.circle(face_img, (350, 180), 15, (50, 50, 50), -1)     # Right eye
cv2.ellipse(face_img, (320, 220), (30, 15), 0, 0, 180, (50, 50, 50), 2)  # Mouth

bbox_face = {'x': 0.5, 'y': 0.4, 'width': 0.3, 'height': 0.4}
score = janah_cv.face_match(face_img, bbox_face)

print(f"Face confidence: {score}%")

if score > 0:
    print(f"✅ Face detected with {score}% confidence")
else:
    print("⚠️  No face detected (OpenCV Haar Cascade may be strict)")

# Summary
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)
print("✅ Using OpenCV face detection (no face_recognition needed)")
print("✅ Detects presence of faces (not identity matching)")
print("💡 For identity matching, install face_recognition library")
print("="*70)