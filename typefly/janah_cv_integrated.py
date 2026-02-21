# -*- coding: utf-8 -*-
"""
Integrated Janah CV - YOLO + Face Recognition + COLOR VERIFICATION
TELLO OPTIMIZED - Your Strategy Implementation
"""

from typefly.janah_cv_v2 import janah_cv_v2
from typefly.reference_manager import reference_manager
import cv2
import numpy as np
import time

class JanahCVIntegrated:
    """
    دمج YOLO مع Face Recognition + Color Verification
    
    YOUR STRATEGY:
    1. YOLO detects person
    2. COLOR CHECK FIRST (fast filter)
    3. FACE CHECK only if color matches
    4. Alert at 70%+ (realistic for Tello)
    """

    def __init__(self):
        self.face_recognizer = janah_cv_v2
        self.is_face_trained = False
        self.reference_info = None
        
        # Alert cooldown to prevent spam
        self.last_alert_time = 0
        self.alert_cooldown = 15.0  # seconds
        
        # Statistics
        self.stats = {
            'persons_scanned': 0,
            'color_filtered': 0,
            'face_checked': 0,
            'targets_found': 0
        }

    def setup_reference(self, image_path, person_info=None):
        """
        إعداد صورة مرجعية للشخص المفقود

        Args:
            image_path: مسار صورة الشخص المفقود
            person_info: معلومات (اسم، عمر، ملابس، إلخ)
        """
        print(f"[Janah SAR] Setting up reference for missing person...")

        # حفظ الصورة المرجعية
        saved_path = reference_manager.set_reference(image_path, person_info)

        # تدريب FaceNet
        success = self.face_recognizer.set_reference_photo(str(saved_path))

        if success:
            self.is_face_trained = True
            self.reference_info = person_info
            
            # Normalize target color
            if person_info and 'clothing_color' in person_info:
                self.reference_info['clothing_color'] = person_info['clothing_color'].lower().strip()
            
            print(f"[Janah SAR] ✅ Reference setup complete!")
            print(f"[Janah SAR] 🎯 Target: {person_info.get('name', 'Unknown')}")
            print(f"[Janah SAR] 👕 Color: {self.reference_info.get('clothing_color', 'unknown')}")
        else:
            print(f"[Janah SAR] ❌ Failed to train face recognizer")

        return success

    def process_frame(self, frame, yolo_detections):
        """
        معالجة الإطار مع YOLO و Face Recognition
        
        YOUR STRATEGY IMPLEMENTED:
        1. YOLO gives us persons
        2. Check COLOR FIRST (fast filter)
        3. Check FACE only if color matches
        4. Alert at 70%+ (realistic for Tello)

        Args:
            frame: الصورة الحالية
            yolo_detections: نتائج YOLO (قائمة الأشخاص المكتشفين)

        Returns:
            list: قائمة بالأشخاص مع نسبة المطابقة
        """
        if not self.is_face_trained:
            return yolo_detections

        enriched_detections = []
        target_color = self.reference_info.get('clothing_color', '').lower()

        for detection in yolo_detections:
            # إذا كان الكشف عن شخص
            if detection.get('class') == 'person':
                self.stats['persons_scanned'] += 1
                bbox = detection.get('bbox', {})
                
                # ===== STEP 1: COLOR CHECK FIRST (YOUR IDEA!) =====
                detected_color, color_confidence = self.face_recognizer.detect_clothing_color(frame, bbox)
                
                detection['detected_color'] = detected_color
                detection['color_confidence'] = color_confidence
                
                # Color filter
                color_matches = False
                if target_color and target_color != 'unknown':
                    color_matches = self._check_color_match(
                        detected_color, target_color, color_confidence
                    )
                    
                    if not color_matches:
                        # ❌ Wrong color → Skip face check! (SAVES PROCESSING)
                        self.stats['color_filtered'] += 1
                        detection['is_target'] = False
                        detection['alert_type'] = 'WRONG_COLOR'
                        detection['face_match_score'] = 0
                        detection['skip_reason'] = f"Color {detected_color} ≠ {target_color}"
                        
                        print(f"[Janah SAR] ⏭️  Skipped: {detected_color} ≠ {target_color}")
                        
                        enriched_detections.append(detection)
                        continue  # Skip this person!
                else:
                    # No target color specified
                    color_matches = True
                
                # ===== STEP 2: FACE CHECK (only if color matched!) =====
                self.stats['face_checked'] += 1
                face_score = self.face_recognizer.face_match(frame, bbox)
                detection['face_match_score'] = face_score
                
                # ===== STEP 3: DECISION (70% threshold) =====
                if face_score >= 70:
                    self.stats['targets_found'] += 1
                    
                    # Determine alert type
                    if face_score >= 85:
                        alert_type = 'HIGH_CONFIDENCE'
                        priority = 'URGENT'
                    else:  # 70-85%
                        alert_type = 'NEEDS_REVIEW'
                        priority = 'HIGH'
                    
                    detection['is_target'] = True
                    detection['alert_type'] = alert_type
                    detection['priority'] = priority
                    detection['target_info'] = self.reference_info
                    
                    # Cooldown check
                    if self._check_cooldown():
                        detection['send_alert'] = True
                        self._log_detection(detection, face_score, detected_color, color_confidence)
                    else:
                        detection['send_alert'] = False
                        print(f"[Janah SAR] 🔇 Alert suppressed (cooldown)")
                    
                    print(f"[Janah SAR] 🎯 TARGET FOUND! Face: {face_score}%, Color: {detected_color}")
                
                else:
                    # Face score < 70%
                    detection['is_target'] = False
                    detection['alert_type'] = 'NO_ALERT'
                    detection['skip_reason'] = f"Face {face_score}% < 70%"

            enriched_detections.append(detection)

        return enriched_detections
    
    def _check_color_match(self, detected: str, target: str, confidence: float) -> bool:
        """
        Check if colors match (with tolerance)
        
        Based on research: HSV color detection has ~77-81% accuracy
        So we allow similar colors if confidence is low
        """
        if confidence < 0.5:  # Low confidence → give benefit of doubt
            return True
        
        detected = detected.lower()
        target = target.lower()
        
        # Exact match
        if detected == target:
            return True
        
        # Similar colors (from research papers)
        similar_colors = {
            'black': ['gray'],
            'gray': ['black'],
            'blue': ['purple'],
            'purple': ['blue', 'pink'],
            'red': ['orange'],
            'orange': ['red', 'yellow'],
            'yellow': ['orange'],
            'white': ['gray']
        }
        
        if target in similar_colors.get(detected, []):
            return True
        
        return False
    
    def _check_cooldown(self) -> bool:
        """Prevent alert spam"""
        current_time = time.time()
        if current_time - self.last_alert_time >= self.alert_cooldown:
            self.last_alert_time = current_time
            return True
        return False
    
    def _log_detection(self, detection, face_score, color, color_conf):
        """Log detection details"""
        print(f"\n{'🚨'*20}")
        print(f"[Janah SAR] TARGET DETECTION")
        print(f"{'='*50}")
        print(f"🎯 Name: {self.reference_info.get('name')}")
        print(f"👤 Face Match: {face_score}%")
        print(f"👕 Detected Color: {color} ({color_conf:.0%} confidence)")
        print(f"✓ Expected Color: {self.reference_info.get('clothing_color')}")
        print(f"🔔 Alert Type: {detection['alert_type']}")
        print(f"📍 Priority: {detection.get('priority', 'UNKNOWN')}")
        print(f"{'='*50}\n")
    
    def get_statistics(self):
        """Get detection statistics"""
        efficiency = 0
        if self.stats['persons_scanned'] > 0:
            efficiency = (self.stats['color_filtered'] / self.stats['persons_scanned']) * 100
        
        return {
            **self.stats,
            'efficiency': f"{efficiency:.1f}% processing saved by color filter"
        }

    def get_color_detection(self, frame, bbox):
        """كشف لون الملابس (for backward compatibility)"""
        color, confidence = self.face_recognizer.detect_clothing_color(frame, bbox)
        return color


# Global instance
janah_cv_integrated = JanahCVIntegrated()