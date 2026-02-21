"""
فحص ما يفعله webui.py الحالي
"""

import ast
import os

print("=" * 80)
print("🔍 Checking Current WebUI Implementation")
print("=" * 80)

webui_path = "typefly/webui.py"

if not os.path.exists(webui_path):
    print("❌ webui.py not found!")
    exit(1)

with open(webui_path, 'r', encoding='utf-8') as f:
    content = f.read()

# فحص الاستيرادات
print("\n📦 Imports Check:")
print("-" * 80)

imports_to_check = {
    'janah_cv': '❌ Not imported - Face Recognition disabled',
    'janah_cv_v2': '❌ Not imported - Face Recognition v2 disabled',
    'janah_cv_integrated': '❌ Not imported - Integrated system disabled',
    'reference_manager': '❌ Not imported - No reference photo support',
    'janah_nlp': '✅ Imported - NLP enabled' if 'janah_nlp' in content else '❌ Not imported',
    'drone_controller': '✅ Imported - Drone control enabled' if 'drone_controller' in content else '❌ Not imported',
    'vision_encoder': '✅ Imported - YOLO enabled' if 'vision' in content.lower() else '❌ Not imported'
}

for module, status in imports_to_check.items():
    if module in content:
        print(f"✅ {module:<25} : Found")
    else:
        print(f"❌ {module:<25} : Not found")

# فحص الوظائف
print("\n🔧 Functions Check:")
print("-" * 80)

functions_to_check = [
    'set_reference',
    'face_match',
    'setup_reference',
    'process_frame',
    'handle_video'
]

for func in functions_to_check:
    if func in content:
        print(f"✅ Function '{func}' found")
    else:
        print(f"❌ Function '{func}' not found")

# فحص نصوص مهمة
print("\n📝 Feature Detection:")
print("-" * 80)

features = {
    'Face Recognition': ['face_match', 'face_recognition', 'janah_cv'],
    'Reference Photo': ['reference', 'set_reference', 'reference_manager'],
    'YOLO Detection': ['yolo', 'detect', 'vision'],
    'NLP': ['nlp', 'natural_language', 'janah_nlp'],
    'Drone Control': ['drone', 'tello', 'move']
}

for feature, keywords in features.items():
    found = any(keyword.lower() in content.lower() for keyword in keywords)
    status = "✅ Enabled" if found else "❌ Disabled"
    print(f"{status:<15} {feature}")

print("\n" + "=" * 80)
print("💡 CONCLUSION:")
print("=" * 80)

if 'janah_cv' in content or 'face_match' in content:
    print("✅ Face Recognition is integrated in current WebUI")
else:
    print("❌ Face Recognition is NOT integrated in current WebUI")
    print("\n📋 To enable Face Recognition, you need to:")
    print("   1. Import janah_cv_integrated")
    print("   2. Add reference photo upload UI")
    print("   3. Process frames with face matching")
    print("   4. Show alerts when target is found")
    print("\n💡 Use webui_enhanced.py for full integration")