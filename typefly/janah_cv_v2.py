"""
Janah CV v2.0 - FaceNet Enhanced Face Recognition
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
            
            print("[Janah CV v2] ✅ Model loaded successfully!")
            print("[Janah CV v2] Features: FaceNet ✅, Deep Learning ✅")
            
        except Exception as e:
            print(f"[Janah CV v2] ❌ Error: {e}")
            raise
    
    def set_reference_photo(self, image_path):
        """Train model with reference photo"""
        print(f"[Janah CV v2] 📸 Loading reference: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            print("[Janah CV v2] ❌ Failed to load image")
            return False
        
        # Generate variations
        print("[Janah CV v2] 🔄 Generating 50 variations...")
        variations = self._generate_variations(img, 50)
        
        # Extract embeddings
        print("[Janah CV v2] 🧠 Extracting face embeddings...")
        self.reference_embeddings = []
        
        for var_img in variations:
            emb = self._extract_embedding(var_img)
            if emb is not None:
                self.reference_embeddings.append(emb)
        
        self.is_trained = len(self.reference_embeddings) > 0
        
        if self.is_trained:
            print(f"[Janah CV v2] ✅ Trained with {len(self.reference_embeddings)} embeddings")
        else:
            print("[Janah CV v2] ❌ Training failed - no faces detected")
        
        return self.is_trained
    
    def _generate_variations(self, image, count=50):
        """Generate image variations using OpenCV"""
        variations = [image.copy()]
        h, w = image.shape[:2]
        
        for i in range(count - 1):
            var = image.copy()
            
            # Brightness adjustment
            if i % 5 == 0:
                factor = np.random.uniform(0.7, 1.3)
                var = np.clip(var * factor, 0, 255).astype(np.uint8)
            
            # Rotation
            elif i % 5 == 1:
                angle = np.random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                var = cv2.warpAffine(var, M, (w, h))
            
            # Horizontal flip
            elif i % 5 == 2:
                var = cv2.flip(var, 1)
            
            # Add noise
            elif i % 5 == 3:
                noise = np.random.normal(0, 10, var.shape).astype(np.int16)
                var = np.clip(var.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Occlusion (simulate glasses)
            else:
                y1 = h // 4 + np.random.randint(-h//10, h//10)
                y2 = y1 + h // 8
                var[y1:y2, :] = (var[y1:y2, :] * 0.6).astype(np.uint8)
            
            variations.append(var)
        
        return variations
    
    def _extract_embedding(self, image):
        """Extract 512-d face embedding"""
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_tensor = self.mtcnn(rgb)
            
            if face_tensor is None:
                return None
            
            with torch.no_grad():
                embedding = self.facenet(face_tensor.unsqueeze(0))
            
            return embedding
        
        except Exception as e:
            return None
    
    def face_match(self, frame, bbox):
        """Compare face with reference (0-100 score)"""
        if not self.is_trained:
            return 0
        
        current_emb = self._extract_embedding(frame)
        
        if current_emb is None:
            return 0
        
        # Calculate cosine similarity with all references
        similarities = []
        for ref_emb in self.reference_embeddings:
            sim = torch.nn.functional.cosine_similarity(
                current_emb, 
                ref_emb,
                dim=1
            ).item()
            similarities.append(sim)
        
        # Average of top 15 matches
        top_n = min(15, len(similarities))
        top_sims = sorted(similarities, reverse=True)[:top_n]
        avg_sim = np.mean(top_sims)
        
        # Convert similarity to 0-100 score
        if avg_sim < 0.3:
            score = 0
        elif avg_sim > 0.7:
            score = int(75 + (avg_sim - 0.7) * 83.3)
        else:
            score = int((avg_sim - 0.3) * 187.5)
        
        return max(0, min(100, score))
    
    def detect_clothing_color(self, frame, bbox):
        """Detect dominant clothing color"""
        h, w = frame.shape[:2]
        
        x1 = int((bbox['x'] - bbox['width']/2) * w)
        y1 = int((bbox['y'] + bbox['height']/4) * h)
        x2 = int((bbox['x'] + bbox['width']/2) * w)
        y2 = int((bbox['y'] + bbox['height']/2) * h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        clothing = frame[y1:y2, x1:x2]
        
        if clothing.size == 0:
            return "unknown"
        
        hsv = cv2.cvtColor(clothing, cv2.COLOR_BGR2HSV)
        avg_color = np.mean(hsv, axis=(0, 1))
        
        h_val, s, v = avg_color
        
        if s < 30:
            return 'white' if v > 200 else 'gray' if v > 100 else 'black'
        
        if h_val < 15 or h_val > 165:
            return 'red'
        elif h_val < 30:
            return 'orange'
        elif h_val < 45:
            return 'yellow'
        elif h_val < 80:
            return 'green'
        elif h_val < 130:
            return 'blue'
        elif h_val < 150:
            return 'purple'
        else:
            return 'pink'

# Create global instance
janah_cv_v2 = JanahCVv2()