# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║         Janah CV Integrated  -  YOLO + Color + Face             ║
║         الرؤية الحاسوبية المتكاملة لنظام جناح                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Uses:                                                          ║
║    janah_color.py  → clothing color detection                   ║
║    janah_face.py   → FaceNet face recognition                   ║
║  ─────────────────────────────────────────────────────────────  ║
║  INPUT FRAME FORMATS:                                           ║
║    Laptop / OpenCV  → BGR   (default, no conversion needed)     ║
║    Tello drone      → RGB   → pass  is_tello=True  to           ║
║                                process_frame()                  ║
║  ─────────────────────────────────────────────────────────────  ║
║  Pipeline (YOUR strategy):                                      ║
║    YOLO detects person                                          ║
║        ↓                                                        ║
║    1. COLOR CHECK  (fast filter – skip wrong colors)            ║
║        ↓  color matches?                                        ║
║    2. FACE CHECK   (FaceNet – only if color matched)            ║
║        ↓  score ≥ 70%?                                          ║
║    3. ALERT  (85%+ HIGH / 70-84% NEEDS_REVIEW)                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import time

from typefly.janah_color import janah_color     # JanahColor instance
from typefly.janah_face  import janah_face      # JanahFace  instance
from typefly.reference_manager import reference_manager


class JanahCVIntegrated:
    """
    Unified CV pipeline: YOLO detections → color filter → face match → alert.

    Quick start (laptop camera / BGR frames):
        cv_pipeline.setup_reference("child.jpg", {"name": "Sara", "clothing_color": "pink"})
        results = cv_pipeline.process_frame(frame_bgr, yolo_detections)

    Quick start (Tello / RGB frames):
        results = cv_pipeline.process_frame(frame_rgb, yolo_detections, is_tello=True)
    """

    # Alert thresholds
    THRESHOLD_HIGH   = 85   # HIGH_CONFIDENCE
    THRESHOLD_MATCH  = 70   # NEEDS_REVIEW

    def __init__(self):
        self.color_detector  = janah_color
        self.face_recognizer = janah_face

        self.is_face_trained = False
        self.reference_info  = {}

        self.last_alert_time = 0.0
        self.alert_cooldown  = 15.0  # seconds between alerts

        self.stats = {
            'persons_scanned' : 0,
            'color_filtered'  : 0,
            'face_checked'    : 0,
            'targets_found'   : 0,
        }

    # ─────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────

    def setup_reference(self, image_path: str, person_info: dict = None) -> bool:
        """
        Load reference photo and train the face recognizer.

        Args:
            image_path  : Path to the missing child's photo.
            person_info : {
                'name'           : 'Sara',
                'age'            : 5,
                'clothing_color' : 'pink',   # ← used for fast color filter
                'description'    : '...'
              }
        Returns:
            True on success.
        """
        print("[Janah SAR] ⚙️  Setting up reference for missing person...")

        # Persist the photo via reference manager
        saved_path = reference_manager.set_reference(image_path, person_info)

        # Train FaceNet
        success = self.face_recognizer.set_reference_photo(str(saved_path))

        if success:
            self.is_face_trained = True
            self.reference_info  = dict(person_info or {})

            # Normalise clothing color string
            raw_color = self.reference_info.get('clothing_color', '')
            self.reference_info['clothing_color'] = raw_color.lower().strip()

            print(f"[Janah SAR] ✅ Reference ready!")
            print(f"[Janah SAR] 🧒 Target : {self.reference_info.get('name', 'Unknown')}")
            print(f"[Janah SAR] 👕 Color  : {self.reference_info['clothing_color']}")
        else:
            print("[Janah SAR] ❌ Face training failed – check reference photo quality")

        return success

    # ─────────────────────────────────────────────────────────────────
    # Main pipeline
    # ─────────────────────────────────────────────────────────────────

    def process_frame(self, frame, yolo_detections: list, is_tello: bool = False) -> list:
        """
        Run the full pipeline on one camera frame.

        Args:
            frame           : Camera frame.
                                BGR (laptop / OpenCV)  → is_tello=False  (default)
                                RGB (Tello drone)      → is_tello=True
            yolo_detections : List of dicts from YOLO, each must have:
                                {'class': 'person', 'bbox': {'x', 'y', 'width', 'height'}}
            is_tello        : Set True when frame comes from djitellopy (RGB).

        Returns:
            Enriched copy of yolo_detections with added keys:
              detected_color, color_confidence,
              face_match_score, is_target, alert_type, [send_alert]
        """
        if not self.is_face_trained:
            return yolo_detections  # pass-through if not set up yet

        # ── Convert Tello RGB → BGR once (all later code uses BGR) ──
        if is_tello:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame  # already BGR

        target_color  = self.reference_info.get('clothing_color', '').lower()
        enriched      = []

        for det in yolo_detections:
            if det.get('class') != 'person':
                enriched.append(det)
                continue

            self.stats['persons_scanned'] += 1
            bbox = det.get('bbox', {})
            det  = dict(det)   # work on a copy

            # ════════════════════════════════════════════════
            # STEP 1 – COLOR CHECK  (fast filter)
            # ════════════════════════════════════════════════
            color, color_conf = self.color_detector.detect_clothing_color(frame_bgr, bbox)
            det['detected_color']    = color
            det['color_confidence']  = color_conf

            color_ok = self._color_matches(color, target_color, color_conf)

            if not color_ok:
                self.stats['color_filtered'] += 1
                det.update({
                    'is_target'        : False,
                    'alert_type'       : 'WRONG_COLOR',
                    'face_match_score' : 0,
                    'skip_reason'      : f"color {color} ≠ target {target_color}",
                })
                print(f"[Janah SAR] ⏭️  Skip (color): {color} ≠ {target_color}")
                enriched.append(det)
                continue

            # ════════════════════════════════════════════════
            # STEP 2 – FACE CHECK  (FaceNet)
            # ════════════════════════════════════════════════
            self.stats['face_checked'] += 1
            score = self.face_recognizer.face_match(frame_bgr, bbox)
            det['face_match_score'] = score

            # ════════════════════════════════════════════════
            # STEP 3 – DECISION
            # ════════════════════════════════════════════════
            if score >= self.THRESHOLD_MATCH:
                self.stats['targets_found'] += 1

                alert_type = ('HIGH_CONFIDENCE' if score >= self.THRESHOLD_HIGH
                              else 'NEEDS_REVIEW')
                priority   = 'URGENT' if score >= self.THRESHOLD_HIGH else 'HIGH'

                det.update({
                    'is_target'   : True,
                    'alert_type'  : alert_type,
                    'priority'    : priority,
                    'target_info' : self.reference_info,
                    'send_alert'  : self._check_cooldown(),
                })

                if det['send_alert']:
                    self._log_match(det, score, color, color_conf)
                else:
                    print(f"[Janah SAR] 🔇 Alert suppressed (cooldown active)")

                print(f"[Janah SAR] 🎯 TARGET  face={score}%  color={color}")

            else:
                det.update({
                    'is_target'  : False,
                    'alert_type' : 'NO_ALERT',
                    'skip_reason': f"face {score}% < {self.THRESHOLD_MATCH}%",
                })

            enriched.append(det)

        return enriched

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _color_matches(self, detected: str, target: str, confidence: float) -> bool:
        """
        True if detected color matches target (with tolerance for low confidence
        and adjacent colors).
        """
        if not target or target == 'unknown':
            return True  # No target color specified → skip color filter

        if confidence < 0.50:
            return True  # Low confidence → give benefit of the doubt

        if detected == target:
            return True

        # Adjacent / similar colors (colour-theory neighbours)
        adjacent = {
            'black'  : ['gray'],
            'gray'   : ['black', 'white'],
            'white'  : ['gray'],
            'blue'   : ['purple'],
            'purple' : ['blue', 'pink'],
            'pink'   : ['purple', 'red'],
            'red'    : ['orange', 'pink'],
            'orange' : ['red', 'yellow'],
            'yellow' : ['orange', 'green'],
            'green'  : ['yellow'],
        }
        return detected in adjacent.get(target, [])

    def _check_cooldown(self) -> bool:
        """Return True if enough time has passed since last alert."""
        now = time.time()
        if now - self.last_alert_time >= self.alert_cooldown:
            self.last_alert_time = now
            return True
        return False

    def _log_match(self, det: dict, score: int, color: str, color_conf: float):
        """Print formatted match log."""
        print(f"\n{'🚨' * 15}")
        print("[Janah SAR] ⚡ TARGET DETECTED")
        print("=" * 50)
        print(f"  🧒 Name         : {self.reference_info.get('name', '?')}")
        print(f"  👤 Face match   : {score}%")
        print(f"  👕 Color found  : {color}  ({color_conf:.0%} conf)")
        print(f"  ✅ Color target : {self.reference_info.get('clothing_color', '?')}")
        print(f"  🔔 Alert type   : {det['alert_type']}")
        print(f"  📍 Priority     : {det.get('priority', '?')}")
        print("=" * 50 + "\n")

    # ─────────────────────────────────────────────────────────────────
    # Statistics & utility
    # ─────────────────────────────────────────────────────────────────

    def get_statistics(self) -> dict:
        """Return processing statistics."""
        scanned = self.stats['persons_scanned']
        saved   = self.stats['color_filtered']
        pct     = (saved / scanned * 100) if scanned else 0
        return {
            **self.stats,
            'efficiency': f"{pct:.1f}% processing saved by color filter",
        }

    def reset_statistics(self):
        """Reset all statistics counters."""
        for k in self.stats:
            self.stats[k] = 0

    def get_color_only(self, frame, bbox, is_tello: bool = False) -> str:
        """Quick helper: just return the detected color string."""
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if is_tello else frame
        color, _  = self.color_detector.detect_clothing_color(frame_bgr, bbox)
        return color


# ─────────────────────────────────────────────────────────────────────
# Global instance
# ─────────────────────────────────────────────────────────────────────
janah_cv_integrated = JanahCVIntegrated()
