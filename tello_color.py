# tello_color_FINAL.py
# -*- coding: utf-8 -*-
"""
Tello Color Detection - CORRECT VERSION
========================================

CRITICAL FIX:
djitellopy returns RGB (NOT BGR!)
We must use COLOR_RGB2HSV (NOT COLOR_BGR2HSV!)

Source: DJITelloPy GitHub
https://github.com/damiafuentes/DJITelloPy
"""

import cv2
import time
import numpy as np
from collections import deque, Counter

try:
    from djitellopy import Tello
except:
    print("❌ djitellopy not installed!")
    exit(1)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False


class TelloColorDetector:
    """
    Color detector for Tello
    
    CRITICAL: Tello frame is RGB (from djitellopy)
    Use cv2.COLOR_RGB2HSV!
    """
    
    def __init__(self):
        # HSV ranges - for RGB → HSV conversion
        self.hsv_ranges = {
            'black': {
                'ranges': [(0, 180, 0, 255, 0, 70)],
            },
            'gray': {
                'ranges': [(0, 180, 0, 50, 70, 180)],
            },
            'white': {
                'ranges': [(0, 180, 0, 50, 180, 255)],
            },
            'red': {
                'ranges': [(0, 10, 40, 255, 40, 255), (170, 180, 40, 255, 40, 255)],
            },
            'orange': {
                'ranges': [(11, 25, 40, 255, 40, 255)],
            },
            'yellow': {
                'ranges': [(26, 40, 40, 255, 40, 255)],
            },
            'green': {
                'ranges': [(41, 85, 35, 255, 35, 255)],
            },
            'blue': {
                'ranges': [(86, 130, 40, 255, 40, 255)],
            },
            'purple': {
                'ranges': [(131, 165, 40, 255, 40, 255)],
            }
        }
        
        self.history_window = deque(maxlen=3)
        self.min_confidence = 0.15  # Minimum confidence threshold
    
    def adaptive_clahe(self, img_rgb):
        """
        CLAHE on RGB image
        
        Args:
            img_rgb: RGB image (from Tello)
        """
        # RGB → Lab
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
        l, a, b = cv2.split(lab)
        
        mean_brightness = np.mean(l)
        
        if mean_brightness < 80:
            clip_limit = 3.5
        elif mean_brightness > 180:
            clip_limit = 1.5
        else:
            clip_limit = 2.5
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        lab_clahe = cv2.merge([l_clahe, a, b])
        
        # Lab → RGB
        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_Lab2RGB)
        
        return rgb_clahe
    
    def detect_color_simple(self, clothing_rgb):
        """
        Detect color from RGB image
        
        CRITICAL:
        - Input: RGB image (from Tello)
        - Use COLOR_RGB2HSV (NOT COLOR_BGR2HSV!)
        
        Args:
            clothing_rgb: RGB image
        """
        if clothing_rgb is None or clothing_rgb.size == 0:
            return 'unknown', 0.0
        
        # CLAHE on RGB
        corrected = self.adaptive_clahe(clothing_rgb)
        
        # RGB → HSV (CRITICAL!)
        hsv = cv2.cvtColor(corrected, cv2.COLOR_RGB2HSV)
        
        # Check ranges
        color_scores = {}
        total_pixels = hsv.shape[0] * hsv.shape[1]
        
        for color_name, color_info in self.hsv_ranges.items():
            matching_pixels = 0
            
            for hsv_range in color_info['ranges']:
                h_min, h_max, s_min, s_max, v_min, v_max = hsv_range
                
                lower = np.array([h_min, s_min, v_min])
                upper = np.array([h_max, s_max, v_max])
                
                mask = cv2.inRange(hsv, lower, upper)
                matching_pixels += np.sum(mask > 0)
            
            color_scores[color_name] = matching_pixels / total_pixels
        
        if not color_scores:
            return 'unknown', 0.0
        
        best_color = max(color_scores.items(), key=lambda x: x[1])
        color_name, raw_confidence = best_color
        
        # Apply minimum confidence threshold
        if raw_confidence < self.min_confidence:
            return 'unknown', raw_confidence
        
        # Temporal smoothing
        self.history_window.append(color_name)
        
        if len(self.history_window) >= 2:
            color_counts = Counter(self.history_window)
            most_common = color_counts.most_common(1)[0]
            consistent_color = most_common[0]
            consistency_ratio = most_common[1] / len(self.history_window)
            
            if consistent_color == color_name:
                final_confidence = raw_confidence * (0.7 + 0.3 * consistency_ratio)
            else:
                final_confidence = raw_confidence * 0.5
        else:
            final_confidence = raw_confidence
        
        return color_name, min(final_confidence, 1.0)


class TelloColorTest:
    """Tello Color Test - FINAL CORRECT VERSION"""
    
    def __init__(self, target_color='red'):
        self.tello = Tello()
        self.detector = TelloColorDetector()
        self.target_color = target_color.lower().strip()
        
        # YOLO
        self.use_yolo = YOLO_AVAILABLE
        if self.use_yolo:
            try:
                self.yolo = YOLO('yolov8n.pt')
                print("✅ YOLO loaded")
            except:
                self.use_yolo = False
        
        self.frame_count = 0
        self.frames_with_person = 0
        self.fps_history = deque(maxlen=30)
        self.detections = {'correct': 0, 'wrong': 0}
        self.second_results = {}
    
    def connect(self):
        print("\n" + "="*60)
        print("🚁 TELLO CONNECTION")
        print("="*60)
        input("Press ENTER when ready...")
        
        try:
            self.tello.connect()
            battery = self.tello.get_battery()
            print(f"✅ Battery: {battery}%")
            
            if battery < 10:
                return False
            
            self.tello.streamon()
            time.sleep(2)
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def detect_persons(self, frame):
        """NO fallback!"""
        if not self.use_yolo:
            return []
        
        try:
            results = self.yolo(frame, verbose=False)
            persons = []
            h, w = frame.shape[:2]
            
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        bbox_width = (x2 - x1) / w
                        bbox_height = (y2 - y1) / h
                        
                        if bbox_width > 0.1 and bbox_height > 0.2:
                            persons.append({
                                'x': float((x1+x2)/2/w),
                                'y': float((y1+y2)/2/h),
                                'width': float((x2-x1)/w),
                                'height': float((y2-y1)/h)
                            })
            
            return persons
        except:
            return []
    
    def extract_clothing(self, frame, bbox):
        """Better ROI"""
        h, w = frame.shape[:2]
        
        x1 = max(0, int((bbox['x'] - bbox['width']/2) * w))
        x2 = min(w, int((bbox['x'] + bbox['width']/2) * w))
        y1 = max(0, int((bbox['y'] - bbox['height']/2) * h))
        y2 = min(h, int((bbox['y'] + bbox['height']/2) * h))
        
        person = frame[y1:y2, x1:x2]
        
        if person.size == 0:
            return None, None
        
        person_h, person_w = person.shape[:2]
        
        cy1 = int(person_h * 0.35)
        cy2 = int(person_h * 0.65)
        cx1 = int(person_w * 0.25)
        cx2 = int(person_w * 0.75)
        
        clothing = person[cy1:cy2, cx1:cx2]
        
        if clothing.size < 1500:
            return None, None
        
        return clothing, (x1, y1, x2, y2)
    
    def run(self, duration=60):
        print("\n" + "="*60)
        print("🎯 TELLO COLOR TEST - FINAL")
        print("="*60)
        print(f"Target: {self.target_color.upper()}")
        print(f"YOLO: {'ON' if self.use_yolo else 'OFF'}")
        print()
        print("CRITICAL FIX:")
        print("  ✅ Tello frame is RGB (from djitellopy)")
        print("  ✅ Using COLOR_RGB2HSV (NOT BGR2HSV!)")
        print("  ✅ Minimum confidence threshold (15%)")
        print()
        print("Press 'q' to quit")
        print("="*60)
        print()
        
        start_time = time.time()
        
        try:
            while True:
                frame_start = time.time()
                current_second = int(time.time() - start_time)
                
                # Get RGB frame from Tello
                frame_rgb = self.tello.get_frame_read().frame
                
                if frame_rgb is None:
                    time.sleep(0.1)
                    continue
                
                if current_second not in self.second_results:
                    self.second_results[current_second] = [0, 0]
                
                # Detect persons
                persons = self.detect_persons(frame_rgb)
                
                # Convert RGB → BGR for display (cv2.imshow expects BGR)
                display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                if not persons:
                    cv2.putText(display, "No person detected", (10, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    self.frames_with_person += 1
                    
                    for person in persons:
                        clothing_rgb, bbox_coords = self.extract_clothing(frame_rgb, person)
                        
                        if clothing_rgb is None:
                            continue
                        
                        # Detect color (expects RGB)
                        color, conf = self.detector.detect_color_simple(clothing_rgb)
                        
                        if color != 'unknown':
                            self.second_results[current_second][1] += 1
                            is_correct = (color == self.target_color)
                            if is_correct:
                                self.second_results[current_second][0] += 1
                                self.detections['correct'] += 1
                            else:
                                self.detections['wrong'] += 1
                            
                            x1, y1, x2, y2 = bbox_coords
                            bbox_color = (0, 255, 0) if is_correct else (0, 0, 255)
                            
                            cv2.rectangle(display, (x1, y1), (x2, y2), bbox_color, 3)
                            
                            person_h = y2 - y1
                            cy1 = y1 + int(person_h * 0.35)
                            cy2 = y1 + int(person_h * 0.65)
                            cv2.rectangle(display, (x1, cy1), (x2, cy2), (255, 255, 0), 2)
                            
                            cv2.putText(display, f"{color.upper()} ({conf:.0%})", (x1, y1-40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, bbox_color, 2)
                            cv2.putText(display, "OK" if is_correct else "WRONG", (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
                            
                            # Show clothing (convert RGB → BGR for imshow)
                            clothing_bgr = cv2.cvtColor(clothing_rgb, cv2.COLOR_RGB2BGR)
                            clothing_display = cv2.resize(clothing_bgr, (200, 300))
                            cv2.imshow('Clothing', clothing_display)
                
                fps = 1.0 / (time.time() - frame_start)
                self.fps_history.append(fps)
                avg_fps = np.mean(self.fps_history)
                
                seconds_with_data = [s for s in self.second_results.values() if s[1] > 0]
                if seconds_with_data:
                    correct_seconds = sum(1 for s in seconds_with_data if s[0]/s[1] >= 0.5)
                    total_seconds = len(seconds_with_data)
                    second_accuracy = (correct_seconds / total_seconds) * 100
                else:
                    second_accuracy = 0.0
                
                battery = self.tello.get_battery()
                
                cv2.putText(display, f"Bat: {battery}%", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(display, f"FPS: {avg_fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(display, f"Target: {self.target_color.upper()}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(display, f"Acc: {second_accuracy:.1f}%", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                cv2.imshow('Tello FINAL', display)
                
                self.frame_count += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if time.time() - start_time >= duration:
                    break
        
        except KeyboardInterrupt:
            pass
        
        finally:
            cv2.destroyAllWindows()
            self.disconnect()
            
            # Final stats
            total = self.detections['correct'] + self.detections['wrong']
            accuracy = (self.detections['correct'] / total * 100) if total > 0 else 0
            print(f"\n✅ Final: {self.detections}, Accuracy: {accuracy:.1f}%")
    
    def disconnect(self):
        try:
            self.tello.streamoff()
            self.tello.end()
        except:
            pass


if __name__ == "__main__":
    print("\n🚁 Tello Color - FINAL CORRECT VERSION")
    print("Source: djitellopy returns RGB frames\n")
    
    target = input("Target (red/blue/black/white/green/yellow): ").lower().strip()
    
    if target not in ['black', 'red', 'yellow', 'blue', 'green', 'white', 'gray']:
        target = 'red'
    
    tester = TelloColorTest(target_color=target)
    
    if tester.connect():
        tester.run(duration=60)