import numpy as np
import cv2
import time
import threading
from multiprocessing import Process, Queue
from typing import Any
from PIL import Image
from overrides import overrides
from ..robot_wrapper import RobotWrapper, RobotObservation
from ..yolo_client import YoloClient
from ..robot_info import RobotInfo
from .airsim_worker import airsim_worker


class AirSimObservation(RobotObservation):
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        self.yolo_client = YoloClient(robot_info)
        self.vehicle_name = robot_info.extra.get("airsim_vehicle", "Drone1")
        self.camera_name = robot_info.extra.get("airsim_camera", "front_center")
        self.host = robot_info.extra.get("airsim_host", "127.0.0.1")
        self.cmd_queue = Queue(maxsize=10)
        self.img_queue = Queue(maxsize=5)
        self._worker_process = None

    @overrides
    def _start(self):
        self._worker_process = Process(
            target=airsim_worker,
            args=(self.cmd_queue, self.img_queue,
                  self.vehicle_name, self.camera_name, self.host),
            daemon=True
        )
        self._worker_process.start()
        time.sleep(3)
        threading.Thread(target=self._image_spin, daemon=True).start()

    def _image_spin(self):
        while self.running:
            try:
                while not self.img_queue.empty():
                    frame = self.img_queue.get_nowait()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._image = Image.fromarray(frame_rgb)
            except Exception as e:
                print(f"[AirSim] image spin error: {e}")
            time.sleep(1.0 / 30.0)

    @overrides
    def _stop(self):
        try:
            self.cmd_queue.put_nowait({'type': 'stop'})
        except Exception:
            pass
        if self._worker_process:
            self._worker_process.terminate()

    @overrides
    async def process_image(self, image: Image.Image):
        await self.yolo_client.detect(image)

    @overrides
    def fetch_processed_result(self) -> dict[str, Any]:
        _, object_list = self.yolo_client.latest_result
        return {"yolo": object_list}


class AirSimDronePlatform(RobotWrapper):
    def __init__(self, robot_info: RobotInfo):
        super().__init__(robot_info, AirSimObservation(robot_info))
        self.vehicle_name = robot_info.extra.get("airsim_vehicle", "Drone1")

    @property
    def _cmd_queue(self):
        return self.obs.cmd_queue

    @overrides
    def start(self) -> bool:
        self.obs.start()
        time.sleep(3)
        print("[AirSim] ✅ جاهز!")
        return True

    @overrides
    def stop(self) -> bool:
        self.obs.stop()
        return True

    @overrides
    def _move(self, dx: float, dy: float):
        duration = max(0.5, abs(dx) + abs(dy))
        try:
            self._cmd_queue.put({'type': 'move', 'dx': dx, 'dy': -dy, 'duration': duration}, timeout=2)
            time.sleep(duration)
        except Exception as e:
            print(f"[AirSim] حركة فشلت: {e}")

    @overrides
    def _rotate(self, deg: float):
        try:
            self._cmd_queue.put({'type': 'rotate', 'deg': deg}, timeout=2)
            time.sleep(1.5)
        except Exception as e:
            print(f"[AirSim] دوران فشل: {e}")