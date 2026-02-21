# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║          Janah Face Recognition  -  Standalone Module           ║
║          التعرف على الوجه  -  مستقل                             ║
╠══════════════════════════════════════════════════════════════════╣
║  INPUT FORMAT:  BGR  (OpenCV standard)                          ║
║  ─────────────────────────────────────────────────────────────  ║
║  Laptop camera → cv2.VideoCapture → BGR  ✅  (ready to use)     ║
║  Tello drone   → djitellopy → RGB                               ║
║    → convert:  frame_bgr = cv2.cvtColor(frame, COLOR_RGB2BGR)  ║
║  ─────────────────────────────────────────────────────────────  ║
║  Pipeline:                                                      ║
║    Training:                                                    ║
║      cv2.imread(path) → BGR → 80 augmented variations           ║
║      → BGR→RGB → MTCNN detect face → FaceNet embedding          ║
║    Inference:                                                   ║
║      BGR frame → crop by bbox → BGR→RGB → MTCNN → FaceNet       ║
║      → cosine similarity vs stored embeddings → 0-100 score     ║
║  ─────────────────────────────────────────────────────────────  ║
║  Score thresholds:                                              ║
║    85-100 : HIGH_CONFIDENCE – very likely the missing child     ║
║    70-84  : NEEDS_REVIEW    – possible match, human review      ║
║    0-69   : NO_MATCH                                            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("[JanahFace] FAIL - facenet_pytorch not installed! Run: pip install facenet-pytorch")


class JanahFace:
    """
    FaceNet-based face recognizer for SAR missing-child detection.

    Workflow:
        # 1. Train on reference photo (once per mission)
        janah_face.set_reference_photo("reference.jpg")

        # 2. Match faces in live frames
        score = janah_face.face_match(frame_bgr, bbox)
        if score >= 70:
            print(f"Possible match: {score}%")

    ⚠️ All frames must be BGR (OpenCV standard).
       For Tello (RGB): frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    bbox format: {'x': 0-1, 'y': 0-1, 'width': 0-1, 'height': 0-1}
    """

    def __init__(self):
        if not FACENET_AVAILABLE:
            self.is_ready = False
            return

        print("[JanahFace] Loading FaceNet (InceptionResnetV1 / vggface2)...")
        try:
            self.mtcnn = MTCNN(
                keep_all=False,
                device='cpu',
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7]  # P-Net / R-Net / O-Net
            )
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

            self.reference_embeddings = []  # List of torch tensors
            self.is_trained  = False
            self.is_ready    = True

            print("[JanahFace] Model loaded OK!")
        except Exception as e:
            print(f"[JanahFace] FAIL - model load error: {e}")
            self.is_ready = False

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def set_reference_photo(self, image_path: str) -> bool:
        """
        Train the recognizer on a reference photo.

        Creates 80 augmented variations (occlusion + brightness + rotation etc.)
        to make the recognizer robust to real-world changes.

        Args:
            image_path: Path to the reference image (any standard format).
                        Loaded as BGR by cv2.imread().

        Returns:
            True if at least one embedding was extracted successfully.
        """
        if not self.is_ready:
            print("[JanahFace] FAIL - model not loaded")
            return False

        print(f"[JanahFace] Loading reference photo: {image_path}")
        img_bgr = cv2.imread(image_path)

        if img_bgr is None:
            print(f"[JanahFace] FAIL - could not read image: {image_path}")
            return False

        # Generate 80 variations
        variations = self._generate_variations(img_bgr, count=80)

        self.reference_embeddings = []
        for var in variations:
            emb = self._embed_bgr(var)
            if emb is not None:
                self.reference_embeddings.append(emb)

        self.is_trained = len(self.reference_embeddings) > 0

        if self.is_trained:
            print(f"[JanahFace] Trained OK: {len(self.reference_embeddings)}/80 embeddings extracted")
        else:
            print("[JanahFace] FAIL - Training failed: no face detected in reference image")
            print("[JanahFace] TIP: Make sure the reference photo shows a clear, frontal face")

        return self.is_trained

    def face_match(self, frame_bgr: np.ndarray, bbox: dict) -> int:
        """
        Match the face in the bounding box against the stored reference.

        Args:
            frame_bgr : BGR frame (OpenCV standard).
                        ⚠️ For Tello: cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            bbox      : {'x': 0-1, 'y': 0-1, 'width': 0-1, 'height': 0-1}

        Returns:
            Match score 0-100
              0-69  → No match
              70-84 → Possible match (needs human review)
              85+   → High confidence match
        """
        if not self.is_trained:
            return 0

        # 1. Crop to person region first (prevents face confusion from background)
        person_bgr = self._crop_bbox(frame_bgr, bbox)

        # 2. Extract embedding (BGR→RGB→FaceNet)
        current_emb = self._embed_bgr(person_bgr)
        if current_emb is None:
            return 0  # No face detected in crop

        # 3. Cosine similarity against all reference embeddings
        similarities = []
        for ref_emb in self.reference_embeddings:
            sim = torch.nn.functional.cosine_similarity(
                current_emb, ref_emb, dim=1
            ).item()
            similarities.append(sim)

        # 4. Top-15 average (more stable than single best match)
        top_n   = min(15, len(similarities))
        top_sims = sorted(similarities, reverse=True)[:top_n]
        avg_sim  = float(np.mean(top_sims))

        # 5. Non-linear score: 0.3 → 0%, 0.7 → 75%, 1.0 → 100%
        if avg_sim < 0.3:
            score = 0
        elif avg_sim > 0.7:
            score = int(75 + (avg_sim - 0.7) * 83.3)
        else:
            score = int((avg_sim - 0.3) * 187.5)

        return max(0, min(100, score))

    def is_high_confidence(self, score: int) -> bool:
        """Returns True if score ≥ 85 (high confidence match)."""
        return score >= 85

    def is_possible_match(self, score: int) -> bool:
        """Returns True if score ≥ 70 (possible match, needs review)."""
        return score >= 70

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _embed_bgr(self, frame_bgr: np.ndarray):
        """
        Extract 512-dim FaceNet embedding from BGR frame.
        Converts BGR→RGB internally (FaceNet expects RGB).
        Returns None if no face is detected.
        """
        try:
            # ⚠️ FaceNet expects RGB – always convert from BGR
            rgb        = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            face_tensor = self.mtcnn(rgb)   # Returns (3, 160, 160) or None
            if face_tensor is None:
                return None
            with torch.no_grad():
                return self.facenet(face_tensor.unsqueeze(0))
        except Exception:
            return None

    def _crop_bbox(self, frame_bgr: np.ndarray, bbox: dict) -> np.ndarray:
        """
        Crop the frame to the person bounding box.
        Falls back to the full frame if the bbox is invalid.
        """
        h, w = frame_bgr.shape[:2]

        x1 = max(0, int((bbox['x'] - bbox['width']  / 2) * w))
        y1 = max(0, int((bbox['y'] - bbox['height'] / 2) * h))
        x2 = min(w, int((bbox['x'] + bbox['width']  / 2) * w))
        y2 = min(h, int((bbox['y'] + bbox['height'] / 2) * h))

        if x2 <= x1 or y2 <= y1:
            return frame_bgr  # Invalid bbox → use full frame

        return frame_bgr[y1:y2, x1:x2]

    def _generate_variations(self, image_bgr: np.ndarray, count: int = 80) -> list:
        """
        Generate 'count' augmented variations of the reference image.

        40% occlusion simulations  (partial shadow / blocked face)
        60% standard augmentations (brightness, rotation, flip, noise)

        All variations remain in BGR format.
        """
        variations = [image_bgr.copy()]
        h, w       = image_bgr.shape[:2]
        occ_count  = int(count * 0.4)

        for i in range(count - 1):
            var = image_bgr.copy()

            # ── Occlusion simulations ─────────────────────────
            if i < occ_count:
                mode = i % 4

                if mode == 0:   # Top-of-face shadow
                    y1 = h // 4 + np.random.randint(-h // 15, h // 15)
                    y2 = min(y1 + h // 8, h)
                    var[y1:y2, :] = (var[y1:y2, :] * 0.6).astype(np.uint8)

                elif mode == 1:  # Mid-face shadow
                    y1 = h // 4
                    y2 = min(y1 + h // 7, h)
                    var[y1:y2, :] = (var[y1:y2, :] * 0.4).astype(np.uint8)

                elif mode == 2:  # Forehead shadow
                    y1 = max(0, h // 4 - h // 20)
                    y2 = min(y1 + h // 6, h)
                    var[y1:y2, :] = (var[y1:y2, :] * 0.5).astype(np.uint8)

                else:            # Random patches
                    for _ in range(2):
                        rx1 = np.random.randint(0, w // 3)
                        rx2 = min(rx1 + w // 4, w)
                        ry1 = np.random.randint(h // 5, h // 2)
                        ry2 = min(ry1 + h // 10, h)
                        var[ry1:ry2, rx1:rx2] = (var[ry1:ry2, rx1:rx2] * 0.65).astype(np.uint8)

            # ── Standard augmentations ────────────────────────
            else:
                mode = i % 4

                if mode == 0:    # Brightness variation
                    factor = np.random.uniform(0.6, 1.4)
                    var    = np.clip(var.astype(np.float32) * factor, 0, 255).astype(np.uint8)

                elif mode == 1:  # Slight rotation (±20°)
                    angle = np.random.uniform(-20, 20)
                    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                    var   = cv2.warpAffine(var, M, (w, h))

                elif mode == 2:  # Horizontal flip
                    var = cv2.flip(var, 1)

                else:            # Gaussian noise
                    noise = np.random.normal(0, 15, var.shape).astype(np.int16)
                    var   = np.clip(var.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            variations.append(var)

        return variations


# ─────────────────────────────────────────────────────────────────────
# Global instance  (import and use directly)
# ─────────────────────────────────────────────────────────────────────
janah_face = JanahFace()
