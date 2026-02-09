"""
Improved Janah CV - FaceNet Based Face Recognition
Replaces LBPH with deep learning model for better accuracy
"""

import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from albumentations import Compose, HorizontalFlip, Rotate, RandomBrightnessContrast, GaussNoise
import warnings
warnings.filterwarnings('ignore')

class ImprovedJanahCV:
    def __init__(self):
        print("[Improved Janah CV] 🚀 Initializing FaceNet model...")
        
        # Face detection model
        self.mtcnn = MTCNN(
            keep_all=False,
            device='cpu',
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7]
        )
        
        # Face recognition model (pretrained on VGGFace2)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        
        # Storage
        self.reference_embeddings = []
        self.is_trained = False
        self.reference_path = None
        
        # Augmentation pipeline for robust training
        self.augment = Compose([
            HorizontalFlip(p=0.5),
            Rotate(limit=15, p=0.7),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            GaussNoise(var_limit=(5, 20), p=0.3),
        ])
        
        print("[Improved Janah CV] ✅ Model loaded successfully!")
        print("[Improved Janah CV] 🎯 Features: FaceNet ✅, Augmentation ✅, Robust ✅")
    
    def set_reference_photo(self, image_path):
        """Train the model with reference photo + augmented variations"""
        print(f"[Improved Janah CV] 📸 Loading reference: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Improved Janah CV] ❌ Failed to load image: {image_path}")
            return False
        
        self.reference_path = image_path
        
        # Generate 60 variations (original + 50 augmented + 9 occlusions)
        print("[Improved Janah CV] 🔄 Generating augmented variations...")
        variations = self._generate_variations(img, num_augmented=50, num_occlusions=9)
        
        # Extract embeddings from all variations
        print("[Improved Janah CV] 🧠 Extracting face embeddings...")
        self.reference_embeddings = []
        
        for i, var_img in enumerate(variations):
            embedding = self._extract_embedding(var_img)
            if embedding is not None:
                self.reference_embeddings.append(embedding)
        
        self.is_trained = len(self.reference_embeddings) > 0
        
        if self.is_trained:
            print(f"[Improved Janah CV] ✅ Training complete with {len(self.reference_embeddings)} embeddings")
            print(f"[Improved Janah CV] 📊 Reference: {image_path}")
        else:
            print("[Improved Janah CV] ❌ Training failed - no faces detected")
        
        return self.is_trained
    
    def _generate_variations(self, image, num_augmented=50, num_occlusions=9):
        """Generate diverse training variations"""
        variations = [image.copy()]  # Original image
        
        # Standard augmentations
        for _ in range(num_augmented):
            try:
                augmented = self.augment(image=image)['image']
                variations.append(augmented)
            except:
                variations.append(image.copy())
        
        # Occlusion variations (simulate glasses, masks, etc.)
        variations.extend(self._add_occlusions(image, num_occlusions))
        
        return variations
    
    def _add_occlusions(self, image, count=9):
        """Simulate glasses, accessories, and partial occlusions"""
        h, w = image.shape[:2]
        variations = []
        
        for i in range(count):
            occluded = image.copy()
            
            # Random horizontal bar (simulates glasses)
            if i < 3:
                y1 = np.random.randint(h//4, h//2)
                y2 = y1 + np.random.randint(h//12, h//8)
                alpha = 0.6
                occluded[y1:y2, :] = (occluded[y1:y2, :] * alpha).astype(np.uint8)
            
            # Random vertical bar (simulates hair/shadow)
            elif i < 6:
                x1 = np.random.randint(0, w//4)
                x2 = x1 + w//6
                alpha = 0.7
                occluded[:, x1:x2] = (occluded[:, x1:x2] * alpha).astype(np.uint8)
            
            # Random shadow/lighting
            else:
                mask = np.random.rand(h, w) > 0.5
                occluded = np.where(mask[:, :, None], occluded * 0.8, occluded).astype(np.uint8)
            
            variations.append(occluded)
        
        return variations
    
    def _extract_embedding(self, image):
        """Extract 512-dimensional face embedding"""
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face and align
            face_tensor = self.mtcnn(rgb)
            
            if face_tensor is None:
                return None
            
            # Extract embedding (512-d vector)
            with torch.no_grad():
                embedding = self.facenet(face_tensor.unsqueeze(0))
            
            return embedding
        
        except Exception as e:
            return None
    
    def face_match(self, frame, bbox):
        """
        Compare current face with reference embeddings
        Returns: confidence score 0-100
        """
        if not self.is_trained:
            return 0
        
        # Extract embedding from current frame
        current_embedding = self._extract_embedding(frame)
        
        if current_embedding is None:
            return 0
        
        # Calculate cosine similarity with all reference embeddings
        similarities = []
        for ref_emb in self.reference_embeddings:
            sim = torch.nn.functional.cosine_similarity(
                current_embedding, 
                ref_emb,
                dim=1
            ).item()
            similarities.append(sim)
        
        # Take average of top 15 best matches (more robust)
        top_n = min(15, len(similarities))
        top_similarities = sorted(similarities, reverse=True)[:top_n]
        avg_similarity = np.mean(top_similarities)
        
        # Convert from [-1, 1] to [0, 100]
        # Apply sigmoid-like transformation for better separation
        score = self._similarity_to_score(avg_similarity)
        
        return max(0, min(100, score))
    
    def _similarity_to_score(self, similarity):
        """Convert similarity to 0-100 score with better separation"""
        # FaceNet similarities typically range from 0.3 (different) to 0.95 (same)
        # We want to map this to 0-100 with good separation
        
        if similarity < 0.3:
            return 0
        elif similarity > 0.7:
            # Same person - map [0.7, 1.0] -> [75, 100]
            return int(75 + (similarity - 0.7) * 83.3)
        else:
            # Different person - map [0.3, 0.7] -> [0, 75]
            return int((similarity - 0.3) * 187.5)
    
    def detect_clothing_color(self, frame, bbox):
        """Detect dominant clothing color (same as original)"""
        h, w = frame.shape[:2]
        
        # Extract clothing region
        x1 = int((bbox['x'] - bbox['width']/2) * w)
        y1 = int((bbox['y'] + bbox['height']/4) * h)
        x2 = int((bbox['x'] + bbox['width']/2) * w)
        y2 = int((bbox['y'] + bbox['height']/2) * h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        clothing_region = frame[y1:y2, x1:x2]
        
        if clothing_region.size == 0:
            return "unknown"
        
        # Convert to HSV
        hsv = cv2.cvtColor(clothing_region, cv2.COLOR_BGR2HSV)
        avg_color = np.mean(hsv, axis=(0, 1))
        
        h, s, v = avg_color
        
        # Color classification
        if s < 30:
            return 'white' if v > 200 else 'gray' if v > 100 else 'black'
        
        if h < 15 or h > 165:
            return 'red'
        elif h < 30:
            return 'orange'
        elif h < 45:
            return 'yellow'
        elif h < 80:
            return 'green'
        elif h < 130:
            return 'blue'
        elif h < 150:
            return 'purple'
        else:
            return 'pink'

# Create global instance
improved_janah_cv = ImprovedJanahCV()