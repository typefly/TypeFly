"""
اختبار النظام المتكامل بدون درون
"""

import cv2
from typefly.janah_cv_integrated import janah_cv_integrated

print("=" * 70)
print("🧪 Testing Integrated Janah SAR System")
print("=" * 70)

# 1. إعداد صورة مرجعية
print("\n[1] Setting up reference photo...")
person_info = {
    'name': 'عزوز',
    'age': 25,
    'clothing_color': 'blue'
}

success = janah_cv_integrated.setup_reference(
    "tests/images/Azzoz2.jpg",
    person_info
)

if not success:
    print("❌ Setup failed!")
    exit(1)

# 2. محاكاة كشف YOLO
print("\n[2] Simulating YOLO detection...")

# صورة اختبار
test_img = cv2.imread("tests/images/Azzoz2_glasses.jpg")

yolo_detections = [
    {
        'class': 'person',
        'confidence': 0.95,
        'bbox': {'x': 0.5, 'y': 0.5, 'width': 0.8, 'height': 0.8}
    }
]

# 3. معالجة الإطار
print("\n[3] Processing frame with face recognition...")
enriched = janah_cv_integrated.process_frame(test_img, yolo_detections)

# 4. عرض النتائج
print("\n" + "=" * 70)
print("📊 Results:")
print("=" * 70)

for det in enriched:
    if det.get('is_target'):
        print(f"🎯 TARGET FOUND!")
        print(f"   Name: {det['target_info']['name']}")
        print(f"   Match Score: {det['face_match_score']}%")
        print(f"   Confidence: {det['confidence']*100:.1f}%")
    else:
        print(f"❌ Not target (Match: {det.get('face_match_score', 0)}%)")

print("\n✅ Integration test complete!")