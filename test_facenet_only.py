"""
FaceNet Only Evaluation Test - No LBPH Dependency
"""

import cv2
import numpy as np
import os
import time

# Import only the improved version
from typefly.improved_janah_cv import improved_janah_cv

print("=" * 80)
print("🧪 Improved FaceNet Face Recognition - Full Evaluation")
print("=" * 80)

# Test images
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

# Store results
all_results = {}

for ref_name, ref_path in reference_images.items():
    print("\n" + "=" * 80)
    print(f"📸 REFERENCE: {ref_name}")
    print("=" * 80)
    
    if not os.path.exists(ref_path):
        print(f"❌ File not found: {ref_path}")
        continue
    
    all_results[ref_name] = {}
    
    # Train with reference
    print("\n[Training FaceNet Model...]")
    print("-" * 80)
    
    start = time.time()
    success = improved_janah_cv.set_reference_photo(ref_path)
    train_time = time.time() - start
    
    if not success:
        print("❌ Training failed!")
        continue
    
    print(f"✅ Training complete in {train_time:.2f}s")
    
    # Test all images
    print("\n[Testing all images...]")
    print("-" * 80)
    print(f"{'Test Image':<20} | {'Score':^8} | {'Time':^10} | {'Status':^8}")
    print("-" * 80)
    
    for test_name, test_path in all_test_images.items():
        img = cv2.imread(test_path)
        if img is None:
            print(f"⚠️  {test_name:<20} | File not found")
            continue
        
        start = time.time()
        score = improved_janah_cv.face_match(img, bbox)
        inference_time = (time.time() - start) * 1000  # ms
        
        all_results[ref_name][test_name] = {
            'score': score,
            'time': inference_time
        }
        
        # Determine if result is correct
        is_same_person = (
            test_name == ref_name or
            (ref_name == "Aboud_big" and "Aboud" in test_name) or
            (ref_name == "Azzoz_no_glasses" and "Azzoz" in test_name)
        )
        
        if is_same_person:
            status = "✅ PASS" if score >= 80 else "❌ FAIL"
        else:
            status = "✅ PASS" if score < 60 else "⚠️ WARN"
        
        print(f"{test_name:<20} | {score:^8}% | {inference_time:^9.1f}ms | {status:^8}")

# ============================================================
# Overall Statistics
# ============================================================
print("\n" + "=" * 80)
print("📊 OVERALL PERFORMANCE STATISTICS")
print("=" * 80)

same_person_scores = []
different_person_scores = []

for ref_name, results in all_results.items():
    for test_name, data in results.items():
        score = data['score']
        
        is_same = (
            test_name == ref_name or
            (ref_name == "Aboud_big" and "Aboud" in test_name) or
            (ref_name == "Azzoz_no_glasses" and "Azzoz" in test_name)
        )
        
        if is_same:
            same_person_scores.append(score)
        else:
            different_person_scores.append(score)

print(f"\n🎯 Same Person Detection (Should be HIGH):")
print(f"   Average Score:  {np.mean(same_person_scores):.1f}%")
print(f"   Min Score:      {np.min(same_person_scores):.1f}%")
print(f"   Max Score:      {np.max(same_person_scores):.1f}%")
print(f"   Std Dev:        {np.std(same_person_scores):.1f}%")

print(f"\n🚫 Different Person Rejection (Should be LOW):")
print(f"   Average Score:  {np.mean(different_person_scores):.1f}%")
print(f"   Min Score:      {np.min(different_person_scores):.1f}%")
print(f"   Max Score:      {np.max(different_person_scores):.1f}%")

print(f"\n📈 Separation Gap: {np.mean(same_person_scores) - np.mean(different_person_scores):.1f}%")
print(f"   (Higher is better - indicates clear distinction)")

# ============================================================
# Detailed Analysis
# ============================================================
print("\n" + "=" * 80)
print("🔍 DETAILED ANALYSIS")
print("=" * 80)

# Check glasses impact
if "Azzoz_no_glasses" in all_results:
    azzoz_results = all_results["Azzoz_no_glasses"]
    if "Azzoz_no_glasses" in azzoz_results and "Azzoz_glasses" in azzoz_results:
        no_glasses = azzoz_results["Azzoz_no_glasses"]['score']
        with_glasses = azzoz_results["Azzoz_glasses"]['score']
        diff = no_glasses - with_glasses
        
        print(f"\n👓 Glasses Impact Test:")
        print(f"   Without Glasses: {no_glasses}%")
        print(f"   With Glasses:    {with_glasses}%")
        print(f"   Impact:          {diff:+.1f}%")
        
        if abs(diff) < 15:
            print(f"   Status:          ✅ Robust to glasses!")
        else:
            print(f"   Status:          ⚠️ Sensitive to glasses")

# Check size variation impact
if "Aboud_big" in all_results:
    aboud_results = all_results["Aboud_big"]
    if "Aboud_big" in aboud_results and "Aboud_small" in aboud_results:
        big = aboud_results["Aboud_big"]['score']
        small = aboud_results["Aboud_small"]['score']
        diff = big - small
        
        print(f"\n📏 Size Variation Test:")
        print(f"   Large Image:  {big}%")
        print(f"   Small Image:  {small}%")
        print(f"   Impact:       {diff:+.1f}%")
        
        if abs(diff) < 10:
            print(f"   Status:       ✅ Robust to size changes!")
        else:
            print(f"   Status:       ⚠️ Sensitive to size")

print("\n" + "=" * 80)
print("✅ Full Evaluation Completed!")
print("=" * 80)

# Color detection test
print("\n" + "=" * 80)
print("🎨 Bonus: Color Detection Test")
print("=" * 80)

img_pink = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.rectangle(img_pink, (50, 50), (590, 430), (203, 192, 255), -1)
bbox_test = {'x': 0.5, 'y': 0.5, 'width': 0.8, 'height': 0.75}

color = improved_janah_cv.detect_clothing_color(img_pink, bbox_test)
print(f"Detected color: {color}")
print("✅ Color detection working!" if color == 'pink' else "⚠️ Color detection issue")