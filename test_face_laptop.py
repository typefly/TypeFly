# test_face_laptop.py
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║      Test: Face Recognition  –  Laptop Webcam                  ║
║      اختبار: التعرف على الوجه على كاميرا اللاب                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Pipeline:  YOLO person → Color filter → FaceNet match          ║
║  Source   : janah_color.py + janah_face.py                      ║
║  Input    : BGR frames from cv2.VideoCapture                    ║
║                                                                 ║
║  Controls:                                                      ║
║    q  →  quit                                                   ║
║    r  →  reset stats                                            ║
║    s  →  save current frame as screenshot                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import os
import sys
import time
import numpy as np
from collections import deque

sys.path.insert(0, '.')

# ── Imports ───────────────────────────────────────────────────────
try:
    from typefly.janah_color import JanahColor
    from typefly.janah_face  import JanahFace
    print("✅ janah_color + janah_face imported")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO imported")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available – using full-frame fallback")

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
CAMERA_INDEX    = 0
WINDOW_TITLE    = "Janah – Face + Color Test (Laptop)"
SCORE_HIGH      = 85    # ← HIGH_CONFIDENCE threshold
SCORE_MATCH     = 70    # ← NEEDS_REVIEW threshold


class FaceLaptopTest:
    """
    Real-time pipeline test on laptop webcam.

    Usage:
        python test_face_laptop.py

    You will be asked for:
      - Path to reference image
      - Target clothing color (optional, '' to skip color filter)
    """

    def __init__(self, reference_path: str, target_color: str = ''):
        self.reference_path = reference_path
        self.target_color   = target_color.lower().strip()

        # ── Models ────────────────────────────────────────────
        self.color_det  = JanahColor()
        self.face_rec   = JanahFace()

        if not self.face_rec.is_ready:
            print("❌ FaceNet not ready. Is facenet_pytorch installed?")
            sys.exit(1)

        print("\n[Pipeline] Training face recognizer …")
        ok = self.face_rec.set_reference_photo(reference_path)
        if not ok:
            print("❌ Could not train on reference image")
            sys.exit(1)

        # ── YOLO ──────────────────────────────────────────────
        self.yolo = None
        if YOLO_AVAILABLE:
            try:
                self.yolo = YOLO('yolov8n.pt')
                print("✅ YOLO model ready")
            except Exception as e:
                print(f"⚠️  YOLO load failed: {e}")

        # ── Stats ─────────────────────────────────────────────
        self.fps_hist    = deque(maxlen=30)
        self.score_hist  = deque(maxlen=60)
        self.detections  = {'high': 0, 'review': 0, 'no_match': 0}
        self.frame_count = 0

    # ──────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {CAMERA_INDEX}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        print("\n" + "=" * 65)
        print(f"  📷 Camera        : {CAMERA_INDEX}")
        print(f"  🖼  Reference     : {self.reference_path}")
        print(f"  👕 Target color  : {self.target_color or '(any – color filter OFF)'}")
        print(f"  YOLO             : {'ON' if self.yolo else 'OFF'}")
        print("=" * 65)
        print("  q=quit   r=reset stats   s=screenshot")
        print("=" * 65 + "\n")

        screenshot_n = 0

        while True:
            t0 = time.time()
            ret, frame_bgr = cap.read()
            if not ret:
                break

            self.frame_count += 1
            bboxes  = self._get_bboxes(frame_bgr)
            display = frame_bgr.copy()

            if not bboxes:
                self._text(display, "No person detected", (20, 200),
                           (0, 0, 200), scale=0.9)
            else:
                for bbox in bboxes:
                    self._process_person(frame_bgr, display, bbox)

            # ── HUD ──────────────────────────────────────────
            fps = 1.0 / max(time.time() - t0, 1e-6)
            self.fps_hist.append(fps)
            self._draw_hud(display)

            cv2.imshow(WINDOW_TITLE, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.detections = {'high': 0, 'review': 0, 'no_match': 0}
                self.score_hist.clear()
                self.color_det.reset_history()
                print("🔄 Stats reset")
            elif key == ord('s'):
                screenshot_n += 1
                fname = f"screenshot_{screenshot_n:03d}.jpg"
                cv2.imwrite(fname, display)
                print(f"📸 Saved: {fname}")

        cap.release()
        cv2.destroyAllWindows()
        self._print_summary()

    # ──────────────────────────────────────────────────────────────
    def _process_person(self, frame_bgr, display, bbox):
        h, w = frame_bgr.shape[:2]
        x1 = max(0, int((bbox['x'] - bbox['width']  / 2) * w))
        x2 = min(w, int((bbox['x'] + bbox['width']  / 2) * w))
        y1 = max(0, int((bbox['y'] - bbox['height'] / 2) * h))
        y2 = min(h, int((bbox['y'] + bbox['height'] / 2) * h))

        # ── Step 1: Color check ───────────────────────────────
        color, conf = self.color_det.detect_clothing_color(frame_bgr, bbox)
        color_ok    = self._color_ok(color, conf)

        if not color_ok:
            # Wrong color → draw gray, skip face
            cv2.rectangle(display, (x1, y1), (x2, y2), (120, 120, 120), 2)
            self._text(display, f"⏭ {color} ≠ {self.target_color}",
                       (x1, y1 - 10), (120, 120, 120), scale=0.6)
            return

        # ── Step 2: Face check ────────────────────────────────
        score = self.face_rec.face_match(frame_bgr, bbox)
        self.score_hist.append(score)

        # ── Step 3: Colour + result ───────────────────────────
        if score >= SCORE_HIGH:
            box_color = (0, 50, 255)     # Red-ish
            label     = f"HIGH CONFIDENCE  {score}%"
            self.detections['high'] += 1
            # Flash the border
            cv2.rectangle(display, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 255), 5)

        elif score >= SCORE_MATCH:
            box_color = (0, 200, 255)    # Orange
            label     = f"NEEDS REVIEW  {score}%"
            self.detections['review'] += 1

        else:
            box_color = (80, 200, 80)    # Green = safe (not the target)
            label     = f"No match  {score}%"
            self.detections['no_match'] += 1

        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3)

        # Clothing region
        ph = y2 - y1
        cy1d = y1 + int(ph * 0.35)
        cy2d = y1 + int(ph * 0.65)
        cv2.rectangle(display, (x1, cy1d), (x2, cy2d), (255, 255, 0), 2)

        self._text(display, label,                   (x1, y1 - 45), box_color)
        self._text(display, f"Color: {color} ({conf:.0%})", (x1, y1 - 20), (200, 200, 200), scale=0.6)

        # Score bar
        bar_w = x2 - x1
        fill  = int(bar_w * score / 100)
        cv2.rectangle(display, (x1, y2 + 5), (x2, y2 + 18), (60, 60, 60), -1)
        cv2.rectangle(display, (x1, y2 + 5), (x1 + fill, y2 + 18), box_color, -1)
        self._text(display, f"{score}%", (x2 + 5, y2 + 15), box_color, scale=0.55)

        # Show clothing crop
        person = frame_bgr[y1:y2, x1:x2]
        if person.size > 0:
            ph2 = person.shape[0]
            crop = person[int(ph2*0.35):int(ph2*0.65), :]
            if crop.size > 0:
                cv2.imshow("Clothing Crop", cv2.resize(crop, (200, 200)))

    def _color_ok(self, detected: str, confidence: float) -> bool:
        """True if no target color set, or detected matches target."""
        if not self.target_color:
            return True
        if confidence < 0.5:
            return True   # Low confidence → proceed to face check anyway
        if detected == self.target_color:
            return True
        adjacent = {
            'black':['gray'],'gray':['black','white'],'white':['gray'],
            'blue':['purple'],'purple':['blue','pink'],'pink':['purple','red'],
            'red':['orange','pink'],'orange':['red','yellow'],'yellow':['orange','green'],
            'green':['yellow'],
        }
        return detected in adjacent.get(self.target_color, [])

    def _get_bboxes(self, frame_bgr):
        if self.yolo is None:
            return [{'x': 0.5, 'y': 0.5, 'width': 0.9, 'height': 0.9}]

        results = self.yolo(frame_bgr, verbose=False)
        bboxes  = []
        h, w    = frame_bgr.shape[:2]

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                if bw > 0.08 and bh > 0.15:
                    bboxes.append({
                        'x':float((x1+x2)/2/w), 'y':float((y1+y2)/2/h),
                        'width':float(bw), 'height':float(bh)
                    })
        return bboxes

    def _draw_hud(self, display):
        total  = sum(self.detections.values())
        avg_sc = float(np.mean(self.score_hist)) if self.score_hist else 0.0

        cv2.rectangle(display, (0, 0), (380, 130), (20, 20, 20), -1)
        cv2.rectangle(display, (0, 0), (380, 130), (60, 60, 60),  1)

        self._text(display, f"FPS       : {np.mean(self.fps_hist):.1f}",
                   (10, 24), (180, 255, 180), scale=0.60)
        self._text(display, f"Color     : {self.target_color or 'any'}",
                   (10, 46), (255, 255, 255), scale=0.60)
        self._text(display, f"Avg score : {avg_sc:.1f}%",
                   (10, 68), (100, 200, 255), scale=0.60)
        self._text(display, f"HIGH      : {self.detections['high']}",
                   (10, 90), (0,  80, 255),   scale=0.60)
        self._text(display, f"Review    : {self.detections['review']}",
                   (10,112), (0, 200, 255),   scale=0.60)

    @staticmethod
    def _text(img, text, pos, color, scale=0.65, thick=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    def _print_summary(self):
        total  = sum(self.detections.values())
        avg_sc = float(np.mean(self.score_hist)) if self.score_hist else 0.0
        print("\n" + "=" * 65)
        print("  📊 FINAL RESULTS")
        print("=" * 65)
        print(f"  Reference     : {self.reference_path}")
        print(f"  Target color  : {self.target_color or '(any)'}")
        print(f"  Total persons : {total}")
        print(f"  HIGH (≥{SCORE_HIGH}%): {self.detections['high']}")
        print(f"  Review (≥{SCORE_MATCH}%): {self.detections['review']}")
        print(f"  No match      : {self.detections['no_match']}")
        print(f"  Avg face score: {avg_sc:.1f}%")
        print("=" * 65)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  👤 Janah Face Recognition Test  –  Laptop Webcam")
    print("  Frame format: BGR (OpenCV standard)")
    print("=" * 65 + "\n")

    # ── Reference image ───────────────────────────────────────
    default_ref = "tests/images/reference.jpg"
    ref_path = input(f"Reference image path (default: {default_ref}): ").strip()
    if not ref_path:
        ref_path = default_ref

    if not os.path.exists(ref_path):
        print(f"❌ File not found: {ref_path}")
        print("   Create a folder  tests/images/  and put your reference image there.")
        sys.exit(1)

    # ── Target clothing color ─────────────────────────────────
    COLORS = ['black','gray','white','red','orange','yellow','green','blue','purple','pink']
    print(f"\nAvailable colors: {', '.join(COLORS)}")
    print("(Press ENTER to skip color filter)")
    target_col = input("Target clothing color: ").strip().lower()

    if target_col and target_col not in COLORS:
        print(f"⚠️  '{target_col}' not recognised → color filter disabled")
        target_col = ''

    tester = FaceLaptopTest(reference_path=ref_path, target_color=target_col)
    tester.run()
