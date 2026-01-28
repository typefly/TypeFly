"""
Janah SAR Computer Vision Module
LBPH Face Recognition + Color Detection + Age Estimation
Optimized for Intel Iris Xe Graphics (CPU/iGPU)
"""

import cv2
import numpy as np
from typing import Optional
import os

class JanahCV:
    """Computer Vision utilities for Janah SAR system"""
    
    # Color ranges in HSV for detection
    COLOR_RANGES = {
        'red': [(0, 100, 100), (10, 255, 255), (170, 100, 100), (180, 255, 255)],
        'pink': [(145, 30, 100), (175, 255, 255)],
        'blue': [(100, 100, 100), (130, 255, 255)],
        'green': [(40, 50, 50), (80, 255, 255)],
        'yellow': [(20, 100, 100), (30, 255, 255)],
        'orange': [(10, 100, 100), (20, 255, 255)],
        'purple': [(130, 50, 50), (160, 255, 255)],
        'white': [(0, 0, 200), (180, 30, 255)],
        'black': [(0, 0, 0), (180, 255, 50)],
    }
    
    def __init__(self):
        """Initialize Janah CV module with LBPH face recognizer"""
        self.reference_photo_path = None
        
        # Initialize face detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize LBPH Face Recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        
        self.is_trained = False
        
        print("[Janah CV] Initialized with LBPH Face Recognition")
        print("[Janah CV] Optimized for Intel Iris Xe Graphics")
        print("[Janah CV] Features: Color ✅, Face Recognition ✅, Age ✅")
    
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
            if max(color_scores.values()) < 100:
                return 'unknown'
            
            dominant_color = max(color_scores, key=color_scores.get)
            return dominant_color
            
        except Exception as e:
            print(f"[Janah CV] Error detecting clothing color: {e}")
            return 'unknown'
    
    def set_reference_photo(self, photo_path: str):
        """
        Train LBPH recognizer with reference photo
        
        Args:
            photo_path: Path to child's reference photo
        """
        try:
            if not os.path.exists(photo_path):
                print(f"[Janah CV] Reference photo not found: {photo_path}")
                return
            
            # Load reference photo
            reference_image = cv2.imread(photo_path)
            if reference_image is None:
                print(f"[Janah CV] Failed to load photo: {photo_path}")
                return
            
            # Convert to grayscale
            gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
            
            # Detect face in reference photo
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            
            if len(faces) == 0:
                print("[Janah CV] No face detected in reference photo")
                return
            
            # Extract face
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size for LBPH
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Train LBPH with multiple variations for robustness
            training_faces = []
            training_labels = []
            
            # Original
            training_faces.append(face_roi)
            training_labels.append(1)
            
            # Slight rotations for robustness
            for angle in [-10, -5, 5, 10]:
                M = cv2.getRotationMatrix2D((100, 100), angle, 1.0)
                rotated = cv2.warpAffine(face_roi, M, (200, 200))
                training_faces.append(rotated)
                training_labels.append(1)
            
            # Brightness variations
            for alpha in [0.8, 0.9, 1.1, 1.2]:
                brightened = cv2.convertScaleAbs(face_roi, alpha=alpha, beta=0)
                training_faces.append(brightened)
                training_labels.append(1)
            
            # Train recognizer
            self.face_recognizer.train(training_faces, np.array(training_labels))
            self.is_trained = True
            self.reference_photo_path = photo_path
            
            print(f"[Janah CV] LBPH trained with {len(training_faces)} face variations")
            print(f"[Janah CV] Reference: {photo_path}")
            
        except Exception as e:
            print(f"[Janah CV] Error training LBPH: {e}")
            import traceback
            traceback.print_exc()
    
    def face_match(self, image: np.ndarray, bbox: dict) -> int:
        """
        Calculate face match score using LBPH recognizer
        
        Args:
            image: Full image (numpy array, BGR format)
            bbox: Person bounding box
        
        Returns:
            Match confidence 0-100 (percentage)
        """
        try:
            # Check if trained
            if not self.is_trained:
                # Fallback to basic face detection
                return self._basic_face_detection(image, bbox)
            
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
            
            # Convert to grayscale
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            
            if len(faces) == 0:
                return 0  # No face detected
            
            # Get largest face
            face_x, face_y, face_w, face_h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[face_y:face_y+face_h, face_x:face_x+face_w]
            
            # Resize to match training size
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Predict using LBPH
            label, confidence_distance = self.face_recognizer.predict(face_roi)
            
            # LBPH confidence: lower is better
            # Typical range: 0-100 (0 = perfect match, 100+ = no match)
            # Convert to percentage: lower distance = higher confidence
            if confidence_distance < 50:
                # Very good match
                match_confidence = int(100 - confidence_distance)
            elif confidence_distance < 80:
                # Possible match
                match_confidence = int(max(0, 50 - (confidence_distance - 50) * 0.5))
            else:
                # No match
                match_confidence = 0
            
            # Clamp to 0-100
            match_confidence = max(0, min(100, match_confidence))
            
            return match_confidence
            
        except Exception as e:
            print(f"[Janah CV] Error in LBPH face recognition: {e}")
            return 0
    
    def _basic_face_detection(self, image: np.ndarray, bbox: dict) -> int:
        """
        Basic face detection without recognition (fallback)
        
        Args:
            image: Full image
            bbox: Person bounding box
        
        Returns:
            Confidence 0-100 based on face presence only
        """
        try:
            h, w = image.shape[:2]
            x1 = int((bbox['x'] - bbox['width']/2) * w)
            y1 = int((bbox['y'] - bbox['height']/2) * h)
            x2 = int((bbox['x'] + bbox['width']/2) * w)
            y2 = int((bbox['y'] + bbox['height']/2) * h)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            person_crop = image[y1:y2, x1:x2]
            if person_crop.size == 0:
                return 0
            
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            
            if len(faces) > 0:
                # Face detected but no recognition
                # Return moderate confidence
                face_x, face_y, face_w, face_h = faces[0]
                face_ratio = (face_w * face_h) / (person_crop.shape[0] * person_crop.shape[1])
                return min(60, int(face_ratio * 200))  # Max 60% without recognition
            else:
                return 0
                
        except Exception as e:
            print(f"[Janah CV] Face detection error: {e}")
            return 0
    
    def estimate_age_from_size(self, bbox: dict) -> str:
        """
        Rough age estimation based on person size in image
        
        Args:
            bbox: Person bounding box (normalized)
        
        Returns:
            Age range string (e.g., "3-5", "6-8")
        """
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