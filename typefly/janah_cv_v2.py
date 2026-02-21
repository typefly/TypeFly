# -*- coding: utf-8 -*-
"""
Janah CV v2.0 - FaceNet Enhanced Face Recognition
FIXED: bbox crop before embedding + cleaner pipeline
TELLO OPTIMIZED: Improved color detection with histogram + adaptive lighting
"""

import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import warnings
warnings.filterwarnings('ignore')

class JanahCVv2:
    def __init__(self):
        print("[Janah CV v2] 🚀 Initializing FaceNet model...")
        try:
            self.mtcnn = MTCNN(
                keep_all=False,
                device='cpu',
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7]
            )
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
            self.reference_embeddings = []
            self.is_trained = False
            print("[Janah CV v2] ✅ Model loaded!")
        except Exception as e:
            print(f"[Janah CV v2] ❌ Error: {e}")
            raise

    def set_reference_photo(self, image_path: str) -> bool:
        """Train with reference photo + augmented variations"""
        print(f"[Janah CV v2] 📸 Loading reference: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print("[Janah CV v2] ❌ Failed to load image")
            return False

        variations = self._generate_variations(img, 80)
        self.reference_embeddings = []

        for var_img in variations:
            emb = self._extract_embedding_from_image(var_img)
            if emb is not None:
                self.reference_embeddings.append(emb)

        self.is_trained = len(self.reference_embeddings) > 0
        if self.is_trained:
            print(f"[Janah CV v2] ✅ Trained with {len(self.reference_embeddings)} embeddings")
        else:
            print("[Janah CV v2] ❌ Training failed - no faces detected")
        return self.is_trained

    def _generate_variations(self, image: np.ndarray, count: int = 80) -> list:
        """Generate augmented variations for robust training"""
        variations = [image.copy()]
        h, w = image.shape[:2]
        occlusion_count = int(count * 0.4)

        for i in range(count - 1):
            var = image.copy()
            if i < occlusion_count:
                if i % 4 == 0:
                    y1 = h // 4 + np.random.randint(-h//15, h//15)
                    y2 = min(y1 + h // 8, h)
                    var[y1:y2, :] = (var[y1:y2, :] * 0.6).astype(np.uint8)
                elif i % 4 == 1:
                    y1 = h // 4
                    y2 = min(y1 + h // 7, h)
                    var[y1:y2, :] = (var[y1:y2, :] * 0.4).astype(np.uint8)
                elif i % 4 == 2:
                    y1 = max(0, h // 4 - h // 20)
                    y2 = min(y1 + h // 6, h)
                    var[y1:y2, :] = (var[y1:y2, :] * 0.5).astype(np.uint8)
                else:
                    for _ in range(2):
                        x1 = np.random.randint(0, w // 3)
                        x2 = min(x1 + w // 4, w)
                        y1 = np.random.randint(h // 5, h // 2)
                        y2 = min(y1 + h // 10, h)
                        var[y1:y2, x1:x2] = (var[y1:y2, x1:x2] * 0.65).astype(np.uint8)
            else:
                vtype = i % 4
                if vtype == 0:
                    factor = np.random.uniform(0.6, 1.4)
                    var = np.clip(var * factor, 0, 255).astype(np.uint8)
                elif vtype == 1:
                    angle = np.random.uniform(-20, 20)
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                    var = cv2.warpAffine(var, M, (w, h))
                elif vtype == 2:
                    var = cv2.flip(var, 1)
                else:
                    noise = np.random.normal(0, 15, var.shape).astype(np.int16)
                    var = np.clip(var.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            variations.append(var)
        return variations

    def _extract_embedding_from_image(self, image: np.ndarray):
        """Extract embedding from a full image (for training)"""
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_tensor = self.mtcnn(rgb)
            if face_tensor is None:
                return None
            with torch.no_grad():
                return self.facenet(face_tensor.unsqueeze(0))
        except Exception:
            return None

    def _crop_by_bbox(self, frame: np.ndarray, bbox: dict) -> np.ndarray:
        """
        ✅ FIX #9: Crop frame by bbox before embedding extraction
        bbox format: {'x': 0-1, 'y': 0-1, 'width': 0-1, 'height': 0-1}
        """
        h, w = frame.shape[:2]
        x1 = max(0, int((bbox['x'] - bbox['width'] / 2) * w))
        y1 = max(0, int((bbox['y'] - bbox['height'] / 2) * h))
        x2 = min(w, int((bbox['x'] + bbox['width'] / 2) * w))
        y2 = min(h, int((bbox['y'] + bbox['height'] / 2) * h))

        if x2 <= x1 or y2 <= y1:
            return frame  # fallback to full frame if bbox invalid

        return frame[y1:y2, x1:x2]

    def face_match(self, frame: np.ndarray, bbox: dict) -> int:
        """
        ✅ FIX #9: Crop by bbox first, then extract FaceNet embedding
        Returns match score 0-100
        """
        if not self.is_trained:
            return 0

        # Crop the person region first
        person_crop = self._crop_by_bbox(frame, bbox)

        current_emb = self._extract_embedding_from_image(person_crop)
        if current_emb is None:
            return 0

        similarities = []
        for ref_emb in self.reference_embeddings:
            sim = torch.nn.functional.cosine_similarity(
                current_emb, ref_emb, dim=1
            ).item()
            similarities.append(sim)

        top_n = min(15, len(similarities))
        top_sims = sorted(similarities, reverse=True)[:top_n]
        avg_sim = np.mean(top_sims)

        if avg_sim < 0.3:
            score = 0
        elif avg_sim > 0.7:
            score = int(75 + (avg_sim - 0.7) * 83.3)
        else:
            score = int((avg_sim - 0.3) * 187.5)

        return max(0, min(100, score))

    def detect_clothing_color(self, frame: np.ndarray, bbox: dict) -> tuple:
        """
        Detect dominant clothing color - TELLO OPTIMIZED
        
        Based on research:
        - HSV color space (best for color detection)
        - Histogram-based (better than simple average)
        - Adaptive CLAHE for outdoor lighting
        - Wider ranges for outdoor conditions
        
        Returns:
            (color_name, confidence)
        """
        h, w = frame.shape[:2]
        
        # Extract person region
        x1 = max(0, int((bbox['x'] - bbox['width'] / 2) * w))
        x2 = min(w, int((bbox['x'] + bbox['width'] / 2) * w))
        y1 = max(0, int((bbox['y'] - bbox['height'] / 2) * h))
        y2 = min(h, int((bbox['y'] + bbox['height'] / 2) * h))
        
        person_region = frame[y1:y2, x1:x2]
        
        if person_region.size == 0:
            return "unknown", 0.0
        
        # Extract clothing region (chest area: 25-65% from top)
        # Avoids head and legs
        person_h = person_region.shape[0]
        clothing_y1 = int(person_h * 0.25)
        clothing_y2 = int(person_h * 0.65)
        
        clothing = person_region[clothing_y1:clothing_y2, :]
        
        if clothing.size == 0:
            return "unknown", 0.0
        
        # Convert to HSV
        hsv = cv2.cvtColor(clothing, cv2.COLOR_BGR2HSV)
        
        # Adaptive CLAHE for outdoor lighting (from research)
        h_chan, s_chan, v_chan = cv2.split(hsv)
        mean_v = np.mean(v_chan)
        
        # Adjust CLAHE based on brightness
        if mean_v < 80:  # Dark (shadow)
            clip_limit = 3.0
        elif mean_v > 180:  # Very bright (direct sun)
            clip_limit = 1.5
        else:  # Normal lighting
            clip_limit = 2.0
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
        v_normalized = clahe.apply(v_chan)
        hsv_normalized = cv2.merge([h_chan, s_chan, v_normalized])
        
        # Get HSV statistics
        h_mean = np.mean(hsv_normalized[:, :, 0])
        s_mean = np.mean(hsv_normalized[:, :, 1])
        v_mean = np.mean(hsv_normalized[:, :, 2])
        
        # Histogram-based detection (more accurate than mean)
        h_hist = cv2.calcHist([hsv_normalized], [0], None, [180], [0, 180])
        h_peak = np.argmax(h_hist)
        
        # Color classification with WIDER ranges for outdoor
        # Based on research: 77-81% accuracy with proper HSV ranges
        if s_mean < 40:  # Low saturation (neutral colors)
            if v_mean < 60:
                color = 'black'
                confidence = 1.0 - (v_mean / 60.0)
            elif v_mean > 180:
                color = 'white'
                confidence = (v_mean - 180) / 75.0
            else:
                color = 'gray'
                confidence = 0.6
        else:  # Saturated colors
            # Use histogram peak for better accuracy
            if h_peak < 15 or h_peak > 165:
                color = 'red'
            elif h_peak < 30:
                color = 'orange'
            elif h_peak < 45:
                color = 'yellow'
            elif h_peak < 85:
                color = 'green'
            elif h_peak < 135:
                color = 'blue'
            elif h_peak < 160:
                color = 'purple'
            else:
                color = 'pink'
            
            # Confidence based on saturation strength
            confidence = min(s_mean / 100.0, 1.0)
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return color, confidence


# Global instance
janah_cv_v2 = JanahCVv2()