"""
Integrated Janah CV - YOLO + Face Recognition
"""

from typefly.janah_cv_v2 import janah_cv_v2
from typefly.reference_manager import reference_manager
import cv2
import numpy as np

class JanahCVIntegrated:
    """ط¯ظ…ط¬ YOLO ظ…ط¹ Face Recognition"""
    
    def __init__(self):
        self.face_recognizer = janah_cv_v2
        self.is_face_trained = False
        self.reference_info = None
    
    def setup_reference(self, image_path, person_info=None):
        """
        ط¥ط¹ط¯ط§ط¯ طµظˆط±ط© ظ…ط±ط¬ط¹ظٹط© ظ„ظ„ط´ط®طµ ط§ظ„ظ…ظپظ‚ظˆط¯
        
        Args:
            image_path: ظ…ط³ط§ط± طµظˆط±ط© ط§ظ„ط´ط®طµ ط§ظ„ظ…ظپظ‚ظˆط¯
            person_info: ظ…ط¹ظ„ظˆظ…ط§طھ (ط§ط³ظ…طŒ ط¹ظ…ط±طŒ ظ…ظ„ط§ط¨ط³طŒ ط¥ظ„ط®)
        """
        print(f"[Janah SAR] Setting up reference for missing person...")
        
        # ط­ظپط¸ ط§ظ„طµظˆط±ط© ط§ظ„ظ…ط±ط¬ط¹ظٹط©
        saved_path = reference_manager.set_reference(image_path, person_info)
        
        # طھط¯ط±ظٹط¨ FaceNet
        success = self.face_recognizer.set_reference_photo(str(saved_path))
        
        if success:
            self.is_face_trained = True
            self.reference_info = person_info
            print(f"[Janah SAR] âœ… Reference setup complete!")
            print(f"[Janah SAR] Person: {person_info.get('name', 'Unknown')}")
        else:
            print(f"[Janah SAR] â‌Œ Failed to train face recognizer")
        
        return success
    
    def process_frame(self, frame, yolo_detections):
        """
        ظ…ط¹ط§ظ„ط¬ط© ط§ظ„ط¥ط·ط§ط± ظ…ط¹ YOLO ظˆ Face Recognition
        
        Args:
            frame: ط§ظ„طµظˆط±ط© ط§ظ„ط­ط§ظ„ظٹط©
            yolo_detections: ظ†طھط§ط¦ط¬ YOLO (ظ‚ط§ط¦ظ…ط© ط§ظ„ط£ط´ط®ط§طµ ط§ظ„ظ…ظƒطھط´ظپظٹظ†)
        
        Returns:
            list: ظ‚ط§ط¦ظ…ط© ط¨ط§ظ„ط£ط´ط®ط§طµ ظ…ط¹ ظ†ط³ط¨ط© ط§ظ„ظ…ط·ط§ط¨ظ‚ط©
        """
        if not self.is_face_trained:
            return yolo_detections
        
        enriched_detections = []
        
        for detection in yolo_detections:
            # ط¥ط°ط§ ظƒط§ظ† ط§ظ„ظƒط´ظپ ط¹ظ† ط´ط®طµ
            if detection.get('class') == 'person':
                bbox = detection.get('bbox', {})
                
                # Face matching
                match_score = self.face_recognizer.face_match(frame, bbox)
                
                # ط¥ط¶ط§ظپط© ط§ظ„ظ…ط¹ظ„ظˆظ…ط§طھ
                detection['face_match_score'] = match_score
                detection['is_target'] = match_score >= 70  # threshold
                
                # ط¥ط°ط§ ظƒط§ظ† ط§ظ„ظ‡ط¯ظپ
                if detection['is_target']:
                    detection['target_info'] = self.reference_info
                    print(f"[Janah SAR] ًںژ¯ TARGET FOUND! Match: {match_score}%")
            
            enriched_detections.append(detection)
        
        return enriched_detections
    
    def get_color_detection(self, frame, bbox):
        """ظƒط´ظپ ظ„ظˆظ† ط§ظ„ظ…ظ„ط§ط¨ط³"""
        return self.face_recognizer.detect_clothing_color(frame, bbox)

# Instance ط¹ط§ظ…
janah_cv_integrated = JanahCVIntegrated()

