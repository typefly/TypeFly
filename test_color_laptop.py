# test_color_laptop.py
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║      Test: Color Detection  –  Laptop Webcam                    ║
║      اختبار: كشف لون الملابس على كاميرا اللاب                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Source : janah_color.py  (JanahColor)                          ║
║  Input  : BGR frames from cv2.VideoCapture                      ║
║                                                                 ║
║  Controls:                                                      ║
║    q  →  quit                                                   ║
║    r  →  reset accuracy stats                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import sys
import time
import numpy as np
from collections import deque

# ── Add project root to path ──────────────────────────────────────
sys.path.insert(0, '.')

# ── Imports ───────────────────────────────────────────────────────
try:
    from typefly.janah_color import JanahColor
    print("✅ janah_color imported")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO imported")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available – bounding box will cover full frame")

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0          # 0 = default webcam
WINDOW_TITLE = "Janah – Color Test (Laptop)"
COLORS       = ['black','gray','white','red','orange','yellow','green','blue','purple','pink']
BGR_PALETTE  = {          # For drawing bounding boxes
    'black' : (30,  30,  30),
    'gray'  : (150,150,150),
    'white' : (240,240,240),
    'red'   : (0,   0, 220),
    'orange': (0, 140, 255),
    'yellow': (0, 220, 220),
    'green' : (0, 200,  80),
    'blue'  : (220, 80,  0),
    'purple': (180,  0, 180),
    'pink'  : (180, 80, 220),
}

# ─────────────────────────────────────────────────────────────────
# Main test class
# ─────────────────────────────────────────────────────────────────
class ColorLaptopTest:
    def __init__(self, target_color: str = 'red'):
        self.target  = target_color.lower().strip()
        self.detector = JanahColor()

        # YOLO
        self.yolo     = None
        if YOLO_AVAILABLE:
            try:
                self.yolo = YOLO('yolov8n.pt')
                print(f"✅ YOLO model ready")
            except Exception as e:
                print(f"⚠️  YOLO load failed: {e}")

        # Stats
        self.correct     = 0
        self.wrong       = 0
        self.fps_hist    = deque(maxlen=30)
        self.second_data = {}   # {second: [correct, total]}

    # ──────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"❌ Cannot open camera index {CAMERA_INDEX}")
            return

        # Try to set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        print("\n" + "=" * 60)
        print(f"  🎯 Target color : {self.target.upper()}")
        print(f"  📷 Camera index : {CAMERA_INDEX}")
        print(f"  YOLO            : {'ON' if self.yolo else 'OFF'}")
        print("=" * 60)
        print("  q = quit    r = reset stats")
        print("=" * 60 + "\n")

        start_time = time.time()

        while True:
            t0  = time.time()
            sec = int(t0 - start_time)

            ret, frame_bgr = cap.read()
            if not ret:
                print("⚠️  Frame read failed")
                break

            # ── Detect persons (YOLO or full-frame fallback) ──
            bboxes = self._get_bboxes(frame_bgr)

            display = frame_bgr.copy()

            if not bboxes:
                self._text(display, "No person detected", (20, 200),
                           (0, 0, 200), scale=0.9)
            else:
                if sec not in self.second_data:
                    self.second_data[sec] = [0, 0]

                for bbox in bboxes:
                    color, conf = self.detector.detect_clothing_color(frame_bgr, bbox)

                    if color == 'unknown':
                        continue

                    self.second_data[sec][1] += 1
                    is_correct = (color == self.target)
                    if is_correct:
                        self.correct += 1
                        self.second_data[sec][0] += 1
                    else:
                        self.wrong += 1

                    # Draw
                    h, w = frame_bgr.shape[:2]
                    x1 = max(0, int((bbox['x'] - bbox['width']  / 2) * w))
                    x2 = min(w, int((bbox['x'] + bbox['width']  / 2) * w))
                    y1 = max(0, int((bbox['y'] - bbox['height'] / 2) * h))
                    y2 = min(h, int((bbox['y'] + bbox['height'] / 2) * h))

                    box_bgr = BGR_PALETTE.get(color, (200, 200, 200))
                    cv2.rectangle(display, (x1, y1), (x2, y2), box_bgr, 3)

                    # Clothing region highlight
                    ph = y2 - y1
                    cy1 = y1 + int(ph * 0.35)
                    cy2 = y1 + int(ph * 0.65)
                    cv2.rectangle(display, (x1, cy1), (x2, cy2), (255, 255, 0), 2)

                    label = f"{color.upper()}  ({conf:.0%})"
                    ok    = "✓ CORRECT" if is_correct else "✗ WRONG"
                    self._text(display, label, (x1, y1 - 45), box_bgr)
                    self._text(display, ok,    (x1, y1 - 18),
                               (0, 200, 0) if is_correct else (0, 0, 200))

                    # Show clothing crop
                    self._show_clothing(frame_bgr, bbox)

            # ── HUD ──────────────────────────────────────────
            fps = 1.0 / max(time.time() - t0, 1e-6)
            self.fps_hist.append(fps)

            total   = self.correct + self.wrong
            acc     = (self.correct / total * 100) if total else 0.0
            sec_acc = self._second_accuracy()

            cv2.rectangle(display, (0, 0), (340, 110), (30, 30, 30), -1)
            self._text(display, f"Target : {self.target.upper()}",  (10, 25),  (255,255,255), scale=0.65)
            self._text(display, f"FPS    : {np.mean(self.fps_hist):.1f}", (10, 48), (180,255,180), scale=0.65)
            self._text(display, f"Acc(frame): {acc:.1f}%",          (10, 71),  (100,220,255), scale=0.65)
            self._text(display, f"Acc(sec)  : {sec_acc:.1f}%",      (10, 94),  (100,255,200), scale=0.65)

            cv2.imshow(WINDOW_TITLE, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.correct = self.wrong = 0
                self.second_data.clear()
                self.detector.reset_history()
                print("🔄  Stats reset")

        cap.release()
        cv2.destroyAllWindows()
        self._print_summary()

    # ──────────────────────────────────────────────────────────────
    def _get_bboxes(self, frame_bgr):
        """Return list of normalised bboxes. Fallback: full-frame bbox."""
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
                        'x'      : float((x1 + x2) / 2 / w),
                        'y'      : float((y1 + y2) / 2 / h),
                        'width'  : float(bw),
                        'height' : float(bh),
                    })
        return bboxes

    def _show_clothing(self, frame_bgr, bbox):
        """Pop-up window showing the extracted clothing patch."""
        h, w = frame_bgr.shape[:2]
        x1 = max(0, int((bbox['x'] - bbox['width']  / 2) * w))
        x2 = min(w, int((bbox['x'] + bbox['width']  / 2) * w))
        y1 = max(0, int((bbox['y'] - bbox['height'] / 2) * h))
        y2 = min(h, int((bbox['y'] + bbox['height'] / 2) * h))

        person = frame_bgr[y1:y2, x1:x2]
        if person.size == 0:
            return

        ph, pw = person.shape[:2]
        cy1, cy2 = int(ph * 0.35), int(ph * 0.65)
        cx1, cx2 = int(pw * 0.25), int(pw * 0.75)
        clothing  = person[cy1:cy2, cx1:cx2]

        if clothing.size < 100:
            return

        disp = cv2.resize(clothing, (200, 200))
        cv2.imshow("Clothing Crop", disp)

    def _second_accuracy(self) -> float:
        """Accuracy computed per second (majority-wins per second)."""
        secs = [s for s in self.second_data.values() if s[1] > 0]
        if not secs:
            return 0.0
        correct_secs = sum(1 for s in secs if s[0] / s[1] >= 0.5)
        return correct_secs / len(secs) * 100

    @staticmethod
    def _text(img, text, pos, color, scale=0.7, thick=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    def _print_summary(self):
        total = self.correct + self.wrong
        acc   = (self.correct / total * 100) if total else 0.0
        print("\n" + "=" * 60)
        print("  📊 FINAL RESULTS")
        print("=" * 60)
        print(f"  Target color   : {self.target.upper()}")
        print(f"  Total detections: {total}")
        print(f"  Correct         : {self.correct}")
        print(f"  Wrong           : {self.wrong}")
        print(f"  Accuracy        : {acc:.1f}%")
        print(f"  Per-second acc  : {self._second_accuracy():.1f}%")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🎨 Janah Color Test  –  Laptop Webcam")
    print("  Input format: BGR (OpenCV standard)")
    print("=" * 60)

    print("\nAvailable colors:")
    for i, c in enumerate(COLORS, 1):
        print(f"  {i:2}. {c}")

    target = input("\nEnter target color (default=red): ").strip().lower() or 'red'
    if target not in COLORS:
        print(f"⚠️  '{target}' not recognised → using 'red'")
        target = 'red'

    tester = ColorLaptopTest(target_color=target)
    tester.run()
