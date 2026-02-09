"""
🔬 Complete Cross-Reference Evaluation Test
"""

import cv2
import os
import numpy as np
from typefly.janah_cv_v2 import janah_cv_v2
import time

print("=" * 80)
print("🔬 COMPLETE CROSS-REFERENCE EVALUATION")
print("=" * 80)

# All test images
all_images = {
    "Net_Reference": "tests/images/reference.jpg",
    "Aboud_big": "tests/images/Same_big1.jpg",
    "Aboud_small": "tests/images/Same_small1.jpg",
    "Azzoz_no_glasses": "tests/images/Azzoz2.jpg",
    "Azzoz_glasses": "tests/images/Azzoz2_glasses.jpg"
}

# Define ground truth (who matches who)
ground_truth = {
    "Net_Reference": ["Net_Reference"],
    "Aboud_big": ["Aboud_big", "Aboud_small"],
    "Aboud_small": ["Aboud_big", "Aboud_small"],
    "Azzoz_no_glasses": ["Azzoz_no_glasses", "Azzoz_glasses"],
    "Azzoz_glasses": ["Azzoz_no_glasses", "Azzoz_glasses"]
}

bbox = {'x': 0.5, 'y': 0.5, 'width': 0.8, 'height': 0.8}

# Results matrix
results_matrix = {}
training_times = {}
inference_times = {}

# Statistics
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Cross-reference evaluation
for ref_name, ref_path in all_images.items():
    print(f"\n{'='*80}")
    print(f"📸 REFERENCE: {ref_name}")
    print('='*80)
    
    if not os.path.exists(ref_path):
        print(f"❌ Not found: {ref_path}")
        continue
    
    # Train with this reference
    start = time.time()
    success = janah_cv_v2.set_reference_photo(ref_path)
    training_times[ref_name] = time.time() - start
    
    if not success:
        print("❌ Training failed!")
        continue
    
    results_matrix[ref_name] = {}
    
    print(f"\n{'Test Image':<20} | {'Score':^8} | {'Expected':^10} | {'Result':^10} | {'Status':^10}")
    print('-'*80)
    
    # Test against all images
    for test_name, test_path in all_images.items():
        img = cv2.imread(test_path)
        if img is None:
            continue
        
        # Inference
        start = time.time()
        score = janah_cv_v2.face_match(img, bbox)
        inf_time = (time.time() - start) * 1000  # ms
        
        # Store results
        results_matrix[ref_name][test_name] = score
        if test_name not in inference_times:
            inference_times[test_name] = []
        inference_times[test_name].append(inf_time)
        
        # Determine expected result
        is_same_person = test_name in ground_truth[ref_name]
        expected = "MATCH" if is_same_person else "DIFFERENT"
        
        # Classify result
        threshold = 75  # Score threshold for match
        predicted_match = score >= threshold
        
        # Confusion matrix
        if is_same_person and predicted_match:
            true_positives += 1
            result = "TP"
            status = "✅ PASS"
        elif is_same_person and not predicted_match:
            false_negatives += 1
            result = "FN"
            status = "❌ FAIL"
        elif not is_same_person and not predicted_match:
            true_negatives += 1
            result = "TN"
            status = "✅ PASS"
        else:  # not same but predicted match
            false_positives += 1
            result = "FP"
            status = "⚠️ WARN"
        
        print(f"{test_name:<20} | {score:^8}% | {expected:^10} | {result:^10} | {status:^10}")

# ============================================================
# Comprehensive Statistics
# ============================================================
print("\n" + "=" * 80)
print("📊 COMPREHENSIVE PERFORMANCE STATISTICS")
print("=" * 80)

total_tests = true_positives + false_positives + true_negatives + false_negatives

print(f"\n🎯 Confusion Matrix:")
print(f"   True Positives (TP):  {true_positives:2d}  (Same person, correctly matched)")
print(f"   False Negatives (FN): {false_negatives:2d}  (Same person, missed)")
print(f"   True Negatives (TN):  {true_negatives:2d}  (Different person, correctly rejected)")
print(f"   False Positives (FP): {false_positives:2d}  (Different person, wrongly matched)")

# Calculate metrics
accuracy = (true_positives + true_negatives) / total_tests * 100
precision = true_positives / (true_positives + false_positives) * 100 if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📈 Performance Metrics:")
print(f"   Accuracy:  {accuracy:.1f}%  {'⭐'*int(accuracy/20)}")
print(f"   Precision: {precision:.1f}%  {'⭐'*int(precision/20)}")
print(f"   Recall:    {recall:.1f}%  {'⭐'*int(recall/20)}")
print(f"   F1-Score:  {f1_score:.1f}%  {'⭐'*int(f1_score/20)}")

# Analyze specific challenges
print(f"\n🔍 Challenge Analysis:")

# Glasses robustness
azzoz_results = results_matrix.get("Azzoz_no_glasses", {})
if "Azzoz_glasses" in azzoz_results:
    no_glass = results_matrix["Azzoz_no_glasses"]["Azzoz_no_glasses"]
    with_glass = results_matrix["Azzoz_no_glasses"]["Azzoz_glasses"]
    diff = abs(no_glass - with_glass)
    
    print(f"\n   👓 Glasses Robustness:")
    print(f"      Without glasses: {no_glass}%")
    print(f"      With glasses:    {with_glass}%")
    print(f"      Impact:          {diff}%")
    
    if diff < 10:
        print(f"      Rating:          ⭐⭐⭐⭐⭐ Excellent!")
    elif diff < 20:
        print(f"      Rating:          ⭐⭐⭐⭐ Good")
    elif diff < 30:
        print(f"      Rating:          ⭐⭐⭐ Moderate")
    else:
        print(f"      Rating:          ⭐⭐ Needs improvement")

# Size variation robustness
aboud_results = results_matrix.get("Aboud_big", {})
if "Aboud_small" in aboud_results:
    big = results_matrix["Aboud_big"]["Aboud_big"]
    small = results_matrix["Aboud_big"]["Aboud_small"]
    diff = abs(big - small)
    
    print(f"\n   📏 Size Variation Robustness:")
    print(f"      Large image:  {big}%")
    print(f"      Small image:  {small}%")
    print(f"      Impact:       {diff}%")
    
    if diff < 10:
        print(f"      Rating:       ⭐⭐⭐⭐⭐ Excellent!")
    elif diff < 15:
        print(f"      Rating:       ⭐⭐⭐⭐ Good")
    else:
        print(f"      Rating:       ⭐⭐⭐ Moderate")

# Performance timing
print(f"\n⏱️  Performance Timing:")
print(f"   Avg Training Time:  {np.mean(list(training_times.values())):.2f}s")
print(f"   Avg Inference Time: {np.mean([np.mean(times) for times in inference_times.values()]):.1f}ms")

# Results matrix heatmap
print(f"\n🔥 Score Heatmap (Reference → Test):")
print(f"\n{'Reference \\ Test':<20}", end='')
for test_name in all_images.keys():
    print(f" {test_name[:12]:^13}", end='')
print()
print('-' * 100)

for ref_name, test_results in results_matrix.items():
    print(f"{ref_name:<20}", end='')
    for test_name in all_images.keys():
        score = test_results.get(test_name, 0)
        
        # Color coding
        if score >= 85:
            emoji = "🟢"
        elif score >= 70:
            emoji = "🟡"
        elif score >= 50:
            emoji = "🟠"
        else:
            emoji = "⚪"
        
        print(f" {emoji} {score:3d}%     ", end='')
    print()

print("\n" + "=" * 80)
print("✅ Complete Evaluation Finished!")
print("=" * 80)

# Final recommendation
print("\n💡 RECOMMENDATIONS:")
if accuracy >= 95:
    print("   ⭐⭐⭐⭐⭐ Model is production-ready!")
elif accuracy >= 90:
    print("   ⭐⭐⭐⭐ Model is very good, minor improvements possible")
elif accuracy >= 85:
    print("   ⭐⭐⭐ Model is good, consider fine-tuning")
else:
    print("   ⭐⭐ Model needs improvement, check configurations")

if false_positives > 2:
    print("   ⚠️  High false positives - consider raising threshold or more training data")
if false_negatives > 2:
    print("   ⚠️  High false negatives - consider lowering threshold or better augmentations")