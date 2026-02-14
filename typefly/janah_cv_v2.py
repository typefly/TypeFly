# -*- coding: utf-8 -*-
"""
Janah CV v2.0 - FaceNet Enhanced Face Recognition
FIXED: bbox crop before embedding + cleaner pipeline
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

    def detect_clothing_color(self, frame: np.ndarray, bbox: dict) -> str:
        """Detect dominant clothing color from bbox region"""
        h, w = frame.shape[:2]
        x1 = max(0, int((bbox['x'] - bbox['width'] / 2) * w))
        y1 = max(0, int((bbox['y'] + bbox['height'] / 4) * h))
        x2 = min(w, int((bbox['x'] + bbox['width'] / 2) * w))
        y2 = min(h, int((bbox['y'] + bbox['height'] / 2) * h))

        clothing = frame[y1:y2, x1:x2]
        if clothing.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(clothing, cv2.COLOR_BGR2HSV)
        h_val, s, v = np.mean(hsv, axis=(0, 1))

        if s < 30:
            return 'white' if v > 200 else 'gray' if v > 100 else 'black'
        if h_val < 15 or h_val > 165: return 'red'
        elif h_val < 30: return 'orange'
        elif h_val < 45: return 'yellow'
        elif h_val < 80: return 'green'
        elif h_val < 130: return 'blue'
        elif h_val < 150: return 'purple'
        else: return 'pink'


# Global instance
janah_cv_v2 = JanahCVv2()