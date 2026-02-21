# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║          Janah Color Detection  -  Standalone Module            ║
║          كشف لون الملابس  -  مستقل                              ║
╠══════════════════════════════════════════════════════════════════╣
║  INPUT FORMAT:  BGR  (OpenCV standard)                          ║
║  ─────────────────────────────────────────────────────────────  ║
║  Laptop camera → cv2.VideoCapture → BGR  ✅  (ready to use)     ║
║  Tello drone   → djitellopy → RGB                               ║
║    → convert:  frame_bgr = cv2.cvtColor(frame, COLOR_RGB2BGR)  ║
║  ─────────────────────────────────────────────────────────────  ║
║  Method:                                                        ║
║    1. Extract person crop from bbox                             ║
║    2. Extract clothing region (35-65% from top)                 ║
║    3. Adaptive CLAHE via Lab colorspace (handles outdoor light) ║
║    4. BGR→HSV conversion                                        ║
║    5. Dual detection: pixel counting + histogram peak           ║
║    6. Temporal smoothing (3-frame history)                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from collections import deque, Counter


class JanahColor:
    """
    Clothing color detector for SAR child search operations.

    Usage (laptop camera - BGR):
        color, conf = janah_color.detect_clothing_color(frame_bgr, bbox)

    Usage (Tello drone - RGB to BGR):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        color, conf = janah_color.detect_clothing_color(frame_bgr, bbox)

    bbox format: {'x': 0-1, 'y': 0-1, 'width': 0-1, 'height': 0-1}
                  (normalized YOLO-style bounding box)
    """

    # Supported colors
    COLORS = ['black', 'gray', 'white', 'red', 'orange',
              'yellow', 'green', 'blue', 'purple', 'pink']

    def __init__(self):
        # HSV color ranges (for BGR→HSV conversion)
        # Format per range: (h_min, h_max, s_min, s_max, v_min, v_max)
        self.hsv_ranges = {
            'black':  {'ranges': [(0,   180,  0,  255,  0,   60)]},
            'gray':   {'ranges': [(0,   180,  0,   50,  60, 180)]},
            'white':  {'ranges': [(0,   180,  0,   40, 200, 255)]},
            'red':    {'ranges': [(0,    10, 50,  255,  50, 255),
                                  (170, 180, 50,  255,  50, 255)]},
            'orange': {'ranges': [(11,   25, 50,  255,  50, 255)]},
            'yellow': {'ranges': [(26,   40, 50,  255,  50, 255)]},
            'green':  {'ranges': [(41,   85, 40,  255,  40, 255)]},
            'blue':   {'ranges': [(86,  130, 50,  255,  50, 255)]},
            'purple': {'ranges': [(131, 160, 40,  255,  40, 255)]},
            'pink':   {'ranges': [(161, 175, 30,  255, 120, 255),
                                  (0,    10, 20,  120, 150, 255)]},
        }

        # Temporal smoothing (3-frame window for stability)
        self.history = deque(maxlen=3)
        self.min_confidence = 0.15  # Minimum threshold to report a color

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def detect_clothing_color(self, frame_bgr: np.ndarray, bbox: dict) -> tuple:
        """
        Detect dominant clothing color from a person bounding box.

        Args:
            frame_bgr : BGR frame  (OpenCV standard)
                        ⚠️ For Tello RGB frames, convert first:
                            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            bbox      : {'x': 0-1, 'y': 0-1, 'width': 0-1, 'height': 0-1}

        Returns:
            (color_name: str, confidence: float 0-1)
            color_name is 'unknown' when no reliable color found.
        """
        clothing_bgr = self._extract_clothing(frame_bgr, bbox)
        if clothing_bgr is None:
            return 'unknown', 0.0

        # Step 1: Adaptive CLAHE for lighting correction
        enhanced = self._apply_clahe(clothing_bgr)

        # Step 2: BGR → HSV
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        s_mean = float(np.mean(hsv[:, :, 1]))
        v_mean = float(np.mean(hsv[:, :, 2]))

        # Step 3: Neutral colors (low saturation → black / gray / white)
        if s_mean < 40:
            color, confidence = self._classify_neutral(v_mean)

        # Step 4: Saturated colors – dual method
        else:
            color, confidence = self._classify_saturated(hsv, s_mean)

        # Apply minimum confidence gate
        if confidence < self.min_confidence:
            return 'unknown', confidence

        # Step 5: Temporal smoothing
        color, confidence = self._smooth(color, confidence)

        return color, max(0.0, min(1.0, confidence))

    def reset_history(self):
        """Reset temporal smoothing (call between different persons/scenes)."""
        self.history.clear()

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _extract_clothing(self, frame_bgr: np.ndarray, bbox: dict):
        """
        Crop clothing region (chest area 35-65% vertically, 25-75% horizontally).
        Returns None if region is too small.
        """
        h, w = frame_bgr.shape[:2]

        x1 = max(0, int((bbox['x'] - bbox['width']  / 2) * w))
        x2 = min(w, int((bbox['x'] + bbox['width']  / 2) * w))
        y1 = max(0, int((bbox['y'] - bbox['height'] / 2) * h))
        y2 = min(h, int((bbox['y'] + bbox['height'] / 2) * h))

        if x2 <= x1 or y2 <= y1:
            return None

        person = frame_bgr[y1:y2, x1:x2]
        if person.size < 100:
            return None

        ph, pw = person.shape[:2]

        # Chest region: rows 35-65%, cols 25-75%
        cy1, cy2 = int(ph * 0.35), int(ph * 0.65)
        cx1, cx2 = int(pw * 0.25), int(pw * 0.75)

        clothing = person[cy1:cy2, cx1:cx2]
        if clothing.size < 1500:  # Too small → unreliable
            return None

        return clothing

    def _apply_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Adaptive CLAHE using L*a*b* colorspace.
        Handles shadows (clip=3.5) and direct sunlight (clip=1.5).
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)

        mean_l = float(np.mean(l))

        if mean_l < 80:
            clip_limit = 3.5     # Dark / shadow
        elif mean_l > 180:
            clip_limit = 1.5     # Overexposed / direct sun
        else:
            clip_limit = 2.5     # Normal lighting

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)

        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)

    def _classify_neutral(self, v_mean: float) -> tuple:
        """Classify black / gray / white from V channel mean."""
        if v_mean < 70:
            return 'black', min(1.0, (70 - v_mean) / 70.0)
        elif v_mean > 200:
            return 'white', min(1.0, (v_mean - 180) / 75.0)
        else:
            return 'gray', 0.65

    def _classify_saturated(self, hsv: np.ndarray, s_mean: float) -> tuple:
        """
        Classify saturated colors using two methods:
          - Method A: Pixel counting (each color range vs total pixels)
          - Method B: Histogram peak of hue channel
        Combines both for higher accuracy.
        """
        total_pixels = hsv.shape[0] * hsv.shape[1]

        # ── Method A: Pixel counting ──────────────────────────
        pixel_scores = {}
        for name, info in self.hsv_ranges.items():
            matched = 0
            for r in info['ranges']:
                lo = np.array([r[0], r[2], r[4]])
                hi = np.array([r[1], r[3], r[5]])
                mask = cv2.inRange(hsv, lo, hi)
                matched += int(np.sum(mask > 0))
            pixel_scores[name] = matched / total_pixels

        pixel_color = max(pixel_scores, key=pixel_scores.get)
        pixel_conf  = pixel_scores[pixel_color]

        # ── Method B: Hue histogram peak ──────────────────────
        h_hist  = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        h_peak  = int(np.argmax(h_hist))

        if   h_peak < 10 or h_peak > 170: hist_color = 'red'
        elif h_peak < 25:                  hist_color = 'orange'
        elif h_peak < 40:                  hist_color = 'yellow'
        elif h_peak < 85:                  hist_color = 'green'
        elif h_peak < 130:                 hist_color = 'blue'
        elif h_peak < 155:                 hist_color = 'purple'
        elif h_peak < 175:                 hist_color = 'pink'
        else:                              hist_color = 'red'

        # ── Combine both methods ───────────────────────────────
        if hist_color == pixel_color:
            # Both agree → high confidence
            color      = hist_color
            confidence = min(1.0, pixel_conf + 0.10)
        elif pixel_conf > 0.30:
            # Pixel counting has strong signal → trust it
            color      = pixel_color
            confidence = pixel_conf
        else:
            # Histogram peak wins (good for mid-saturation)
            color      = hist_color
            confidence = min(s_mean / 128.0, 1.0)

        return color, confidence

    def _smooth(self, color: str, confidence: float) -> tuple:
        """3-frame temporal smoothing to reduce flickering."""
        self.history.append(color)

        if len(self.history) < 2:
            return color, confidence

        counts           = Counter(self.history)
        common_color, n  = counts.most_common(1)[0]
        consistency      = n / len(self.history)

        if common_color == color:
            smoothed_conf = confidence * (0.70 + 0.30 * consistency)
            return color, smoothed_conf
        else:
            # History says different color → reduce confidence
            return common_color, confidence * 0.50


# ─────────────────────────────────────────────────────────────────────
# Global instance  (import and use directly)
# ─────────────────────────────────────────────────────────────────────
janah_color = JanahColor()
