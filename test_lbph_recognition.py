import cv2
import numpy as np
from typefly.janah_cv import janah_cv
import os

print("="*70)
print("🧪 Janah LBPH Face Recognition Test")
print("="*70)

# Test 1: Train with reference photo
print("\n[Test 1] Training LBPH with reference photo")
print("-"*70)

# Create test reference photo (or use real one)
test_ref_path = "test_photos/reference.jpg"

if not os.path.exists("test_photos"):
    os.makedirs("test_photos")

if not os.path.exists(test_ref_path):
    print("⚠️  No reference photo found.")
    print("   Please add a photo at: test_photos/reference.jpg")
    print("   The photo should contain a clear face.")
else:
    janah_cv.set_reference_photo(test_ref_path)
    
    if janah_cv.is_trained:
        print("✅ LBPH trained successfully!")
        
        # Test 2: Face matching
        print("\n[Test 2] Testing face recognition")
        print("-"*70)
        
        # Load same reference and test matching
        test_image = cv2.imread(test_ref_path)
        bbox = {'x': 0.5, 'y': 0.5, 'width': 0.8, 'height': 0.8}
        
        match_score = janah_cv.face_match(test_image, bbox)
        print(f"Match score (same image): {match_score}%")
        
        if match_score > 70:
            print("✅ HIGH confidence match!")
        elif match_score > 40:
            print("⚠️  MEDIUM confidence match")
        else:
            print("❌ LOW confidence")
    else:
        print("❌ Training failed")

# Test 3: Color detection still works
print("\n[Test 3] Color detection")
print("-"*70)

img_pink = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.rectangle(img_pink, (50, 50), (590, 430), (203, 192, 255), -1)
bbox_test = {'x': 0.5, 'y': 0.5, 'width': 0.8, 'height': 0.75}

color = janah_cv.detect_clothing_color(img_pink, bbox_test)
print(f"Detected color: {color}")
if color == 'pink':
    print("✅ Color detection working!")
else:
    print(f"⚠️  Expected 'pink', got '{color}'")

print("\n" + "="*70)
print("✅ LBPH Face Recognition Test Complete")
print("="*70)