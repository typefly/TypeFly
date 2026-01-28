"""
Janah SAR Computer Vision Module
Clothing color detection and face matching for missing child identification
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import os

class JanahCV:
    """Computer Vision utilities for Janah SAR system"""
    
    # Color ranges in HSV for detection
    COLOR_RANGES = {
        'red': [(0, 100, 100), (10, 255, 255), (170, 100, 100), (180, 255, 255)],  # Red wraps around
        'pink': [(145, 30, 100), (175, 255, 255)],  # Wider range + lower saturation threshold
        'blue': [(100, 100, 100), (130, 255, 255)],
        'green': [(40, 50, 50), (80, 255, 255)],
        'yellow': [(20, 100, 100), (30, 255, 255)],
        'orange': [(10, 100, 100), (20, 255, 255)],
        'purple': [(130, 50, 50), (160, 255, 255)],
        'white': [(0, 0, 200), (180, 30, 255)],
        'black': [(0, 0, 0), (180, 255, 50)],
    }
    
    def __init__(self):
        """Initialize Janah CV module"""
        self.reference_face_encoding = None
        self.reference_photo_path = None
    
    def detect_clothing_color(self, image: np.ndarray, bbox: dict) -> str:
        """
        Detect dominant clothing color from person bounding box
        
        Args:
            image: Full image (numpy array, BGR format)
            bbox: Dictionary with keys 'x', 'y', 'width', 'height' (normalized 0-1)
        
        Returns:
            Color name as string (e.g., 'pink', 'blue', 'red')
        """
        try:
            # Convert normalized bbox to pixel coordinates
            h, w = image.shape[:2]
            x1 = int((bbox['x'] - bbox['width']/2) * w)
            y1 = int((bbox['y'] - bbox['height']/2) * h)
            x2 = int((bbox['x'] + bbox['width']/2) * w)
            y2 = int((bbox['y'] + bbox['height']/2) * h)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Crop person region
            person_crop = image[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return 'unknown'
            
            # Focus on torso region (middle 60% vertically, full width)
            torso_h = person_crop.shape[0]
            torso_y1 = int(torso_h * 0.2)
            torso_y2 = int(torso_h * 0.8)
            torso = person_crop[torso_y1:torso_y2, :]
            
            if torso.size == 0:
                return 'unknown'
            
            # Convert to HSV
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            
            # Count pixels for each color
            color_scores = {}
            for color_name, ranges in self.COLOR_RANGES.items():
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                
                # Handle colors with multiple ranges (like red)
                if color_name == 'red':
                    # Red has two ranges (wraps around hue)
                    lower1, upper1, lower2, upper2 = ranges
                    mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
                    mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    lower, upper = ranges
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                
                # Count matching pixels
                color_scores[color_name] = np.sum(mask > 0)
            
            # Get dominant color
            if max(color_scores.values()) < 100:  # Threshold for minimum pixels
                return 'unknown'
            
            dominant_color = max(color_scores, key=color_scores.get)
            return dominant_color
            
        except Exception as e:
            print(f"[Janah CV] Error detecting clothing color: {e}")
            return 'unknown'
    
    def set_reference_photo(self, photo_path: str):
        """
        Set reference photo for face matching
        
        Args:
            photo_path: Path to child's reference photo
        """
        try:
            # Try to import face_recognition
            try:
                import face_recognition
            except ImportError:
                print("[Janah CV] Warning: face_recognition not installed. Face matching disabled.")
                print("[Janah CV] Install with: pip install face_recognition")
                return
            
            if not os.path.exists(photo_path):
                print(f"[Janah CV] Reference photo not found: {photo_path}")
                return
            
            # Load reference photo
            reference_image = face_recognition.load_image_file(photo_path)
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(reference_image)
            
            if len(face_encodings) == 0:
                print("[Janah CV] No face found in reference photo")
                return
            
            self.reference_face_encoding = face_encodings[0]
            self.reference_photo_path = photo_path
            print(f"[Janah CV] Reference photo loaded: {photo_path}")
            
        except Exception as e:
            print(f"[Janah CV] Error loading reference photo: {e}")
    
    def face_match(self, image: np.ndarray, bbox: dict) -> int:
        """
        Calculate face match score with reference photo
        
        Args:
            image: Full image (numpy array, BGR format)
            bbox: Person bounding box
        
        Returns:
            Match score 0-100 (percentage)
        """
        try:
            # Check if reference photo is set
            if self.reference_face_encoding is None:
                return 0
            
            # Try to import face_recognition
            try:
                import face_recognition
            except ImportError:
                return 0
            
            # Convert normalized bbox to pixel coordinates
            h, w = image.shape[:2]
            x1 = int((bbox['x'] - bbox['width']/2) * w)
            y1 = int((bbox['y'] - bbox['height']/2) * h)
            x2 = int((bbox['x'] + bbox['width']/2) * w)
            y2 = int((bbox['y'] + bbox['height']/2) * h)
            
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Crop person region
            person_crop = image[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return 0
            
            # Convert BGR to RGB for face_recognition
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            
            # Detect faces in cropped region
            face_encodings = face_recognition.face_encodings(person_rgb)
            
            if len(face_encodings) == 0:
                return 0  # No face detected
            
            # Compare with reference
            face_encoding = face_encodings[0]
            face_distance = face_recognition.face_distance([self.reference_face_encoding], face_encoding)[0]
            
            # Convert distance (0=identical, 1=very different) to similarity percentage
            # Typical threshold is 0.6, so we map:
            # 0.0 -> 100%, 0.6 -> 0%, >0.6 -> 0%
            similarity = max(0, (1 - face_distance / 0.6) * 100)
            similarity = min(100, similarity)  # Cap at 100
            
            return int(similarity)
            
        except Exception as e:
            print(f"[Janah CV] Error in face matching: {e}")
            return 0
    
    def estimate_age_from_size(self, bbox: dict) -> str:
        """
        Rough age estimation based on person size in image
        
        Args:
            bbox: Person bounding box (normalized)
        
        Returns:
            Age range string (e.g., "3-5", "6-8")
        """
        # Very rough heuristic: smaller bbox height = younger child
        height = bbox['height']
        
        if height < 0.3:
            return "3-5"
        elif height < 0.45:
            return "6-8"
        elif height < 0.6:
            return "9-12"
        else:
            return "13+"

# Global instance
janah_cv = JanahCV()