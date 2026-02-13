"""
Janah Tello Wrapper - للسيمولاتور والدرون الحقيقي
=======================================================
يعمل مع TypeFly بالكامل.
للتبديل بين السيمولاتور والدرون:
  robot_info.json → "robot_type": "tello_sim"   ← سيمولاتور
  robot_info.json → "robot_type": "tello"        ← درون حقيقي
"""

import time
import threading
from typing import Any
from PIL import Image
import numpy as np
import cv2
from overrides import overrides

from ..robot_wrapper import RobotWrapper, RobotObservation
from ..yolo_client import YoloClient
from ..robot_info import RobotInfo

import logging

MOVEMENT_MIN = 20
MOVEMENT_MAX = 300
EXECUTION_DELAY = 0.8


# ==========================================
# Observation: مصدر الصور
# ==========================================

class TelloSimObservation(RobotObservation):
    """
    Observation يستخدم السيمولاتور أو الدرون الحقيقي
    """
    
    def __init__(self, drone, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        self.drone = drone
        self.yolo_client = YoloClient(robot_info)
        
        def _capture_spin():
            frame_reader = self.drone.get_frame_read()
            while self.running:
                frame = None
                if frame_reader:
                    frame = frame_reader.frame
                if frame is not None:
                    # frame من الدرون يكون RGB، من السيمولاتور BGR
                    if hasattr(self.drone, '_sim_mode'):
                        self._image = Image.fromarray(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        )
                    else:
                        self._image = Image.fromarray(frame)
                time.sleep(0.033)  # ~30fps
                
        self.capture_thread = threading.Thread(target=_capture_spin, daemon=True)

    @overrides
    def _start(self):
        self.drone.streamon()
        self.capture_thread.start()

    @overrides
    def _stop(self):
        self.running = False
        self.capture_thread.join(timeout=2)
        self.drone.streamoff()

    @overrides
    async def process_image(self, image: Image.Image):
        await self.yolo_client.detect(image)

    @overrides
    def fetch_processed_result(self) -> dict[str, Any]:
        _, object_list = self.yolo_client.latest_result
        return {"yolo": object_list}


# ==========================================
# الـ Wrapper الرئيسي
# ==========================================

class TelloSimWrapper(RobotWrapper):
    """
    Wrapper كامل يدعم:
    - السيمولاتور (كاميرا اللابتوب)
    - الدرون الحقيقي Tello
    
    TypeFly يستخدم نفس الـ interface في الحالتين ✅
    """
    
    def __init__(self, robot_info: RobotInfo):
        # اختر الدرون المناسب
        self.sim_mode = robot_info.robot_type == "tello_sim"
        
        if self.sim_mode:
            from .janah_simulator import JanahSimulator
            camera_index = int(robot_info.extra.get("capture", 0))
            self.drone = JanahSimulator(camera_index)
            self.drone._sim_mode = True  # علامة للـ observation
            print(f"🎮 [Janah] وضع السيمولاتور نشط")
        else:
            try:
                from djitellopy import Tello
                import logging
                Tello.LOGGER.setLevel(logging.WARNING)
                self.drone = Tello()
                print(f"🚁 [Janah] وضع الدرون الحقيقي")
            except ImportError:
                raise ImportError("djitellopy غير مثبت: pip install djitellopy")
        
        super().__init__(robot_info, TelloSimObservation(self.drone, robot_info))
        
        # أضف مهارة الارتفاع
        self.skillset.add_skill(self.lift, "ارتفع أو انزل بمسافة معينة")
        
        self.last_command_time = time.time()
        self.running = True
        
        # Keep-alive thread (مهم للدرون الحقيقي)
        if not self.sim_mode:
            self.keep_alive_thread = threading.Thread(
                target=self._keep_alive, daemon=True
            )

    # ==========================================
    # الأوامر الأساسية
    # ==========================================

    @overrides
    def start(self) -> bool:
        """إقلاع"""
        if not self.sim_mode:
            self.drone.connect()
            battery = self.drone.query_battery()
            print(f"🔋 البطارية: {battery}%")
            if battery < 10:
                print("⚠️ البطارية منخفضة جداً!")
                return False
            self.keep_alive_thread.start()
        
        self.drone.takeoff()
        self.obs.start()
        return True

    @overrides
    def stop(self) -> bool:
        """هبوط"""
        self.drone.land()
        self.obs.stop()
        self.running = False
        return True

    @overrides
    def _move(self, dx: float, dy: float):
        """
        حركة أفقية
        dx: أمام/خلف (متر، موجب = أمام)
        dy: يمين/يسار (متر، موجب = يسار)
        """
        print(f"-> Move forward by {dx} m")
        dx_cm = int(dx * 100)
        dy_cm = int(dy * 100)
        
        if dx_cm > 0:
            self.drone.move_forward(self._cap(dx_cm))
        elif dx_cm < 0:
            self.drone.move_back(self._cap(-dx_cm))
        
        time.sleep(EXECUTION_DELAY)
        
        if dy_cm > 0:
            self.drone.move_left(self._cap(dy_cm))
        elif dy_cm < 0:
            self.drone.move_right(self._cap(-dy_cm))
        
        self.last_command_time = time.time()
        time.sleep(EXECUTION_DELAY)

    @overrides
    def _rotate(self, deg: float):
        """دوران"""
        print(f"-> Rotate left by {deg} degrees")
        if deg > 0:
            self.drone.rotate_counter_clockwise(int(deg))
        else:
            self.drone.rotate_clockwise(int(-deg))
        self.last_command_time = time.time()
        time.sleep(EXECUTION_DELAY)

    def lift(self, dist: float):
        """ارتفع أو انزل"""
        print(f"-> Lift by {dist} cm")
        dist_cm = self._cap(abs(int(dist)))
        if dist > 0:
            self.drone.move_up(dist_cm)
        else:
            self.drone.move_down(dist_cm)
        self.last_command_time = time.time()
        time.sleep(EXECUTION_DELAY)

    # ==========================================
    # Helpers
    # ==========================================

    def _cap(self, dist: int) -> int:
        return max(MOVEMENT_MIN, min(MOVEMENT_MAX, dist))

    def _keep_alive(self):
        """يبعث أمر كل 4 ثواني لمنع الدرون من الهبوط"""
        while self.running:
            if time.time() - self.last_command_time > 4:
                self.drone.send_control_command("command")
                self.last_command_time = time.time()
            time.sleep(0.5)