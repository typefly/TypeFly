import time, cv2
from typing import Any
import numpy as np
from PIL import Image
import threading
from overrides import overrides

from podtp import Podtp, sensor

from ..robot_wrapper import RobotWrapper, RobotObservation
from ..yolo_client import YoloClient
from ..robot_info import RobotInfo
from ..utils import undistort_image

MOVEMENT_MIN = 20
MOVEMENT_MAX = 100
EXECUTION_DELAY = 0.8

POD_CAM_K = np.array([[454.19405878,   0.,         617.24234876],
                  [  0.,         452.65234296, 299.6066995 ],
                  [  0.,           0.,           1.        ]])
    
POD_CAM_D = np.array([[ 0.47264424],
                [ 0.96219725],
                [-2.22589356],
                [ 1.31717773]])

class PodObservation(RobotObservation):
    def __init__(self, sensor: sensor.Sensor, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        self.sensor = sensor
        self.yolo_client = YoloClient(robot_info)

        def _capture_spin():
            while self.running:
                frame = sensor.frame
                # Convert the frame to RGB and store it in self._image
                if frame is not None:
                    undistorted_frame = undistort_image(frame, POD_CAM_K, POD_CAM_D)
                    self._image = Image.fromarray(undistorted_frame)
                time.sleep(0.1)
        self.capture_thread = threading.Thread(target=_capture_spin)
    
    @overrides
    def _start(self):
        self.capture_thread.start()
    
    @overrides
    def _stop(self):
        self.capture_thread.join()

    @overrides
    async def process_image(self, image: Image.Image):
        await self.yolo_client.detect(image)
    
    @overrides
    def fetch_processed_result(self) -> dict[str, Any]:
        _, object_list = self.yolo_client.latest_result
        return {
            "yolo": object_list
        }

class PodWrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo):
        self.podtp = Podtp(robot_info.extra)
        super().__init__(robot_info, PodObservation(self.podtp.sensor_data, robot_info))

        self.height = 0.7
        self.xy_speed = 0.3
        self.flying = False

        # extra movement skills
        self.skillset.add_skill(self.lift, "Move up/down by a distance")
        self.skillset.add_skill(self.land, "Land the drone")

    def _cap_dist(self, dist):
        if abs(dist) < MOVEMENT_MIN:
            return MOVEMENT_MIN if dist > 0 else -MOVEMENT_MIN
        elif abs(dist) > MOVEMENT_MAX:
            return MOVEMENT_MAX if dist > 0 else -MOVEMENT_MAX
        return dist

    @overrides
    def start(self) -> bool:
        if not self.podtp.connect():
            print("Failed to connect to the drone")
            return False
        self.podtp.start_stream()
        self.obs.start()
        self._take_off()
        return True
    
    def _take_off(self):
        if not self.podtp.ctrl_lock(False):
            print("Failed to unlock control")
            return False
        else:
            self.flying = True
            self._take_off_from_dog()
            print("Drone started")
    
    def _take_off_from_dog(self):
        # dog is around 40cm high
        self.podtp.reset_estimator(40)
        count = 0
        while count < 15:
            self.podtp.command_hover(0, 0, 0, self.height)
            time.sleep(0.2)
            count += 1
        # self.podtp.command_position(0.6, 0, 0, 0)
        self._move(0.6, 0.0)

    @overrides
    def stop(self) -> bool:
        self.obs.stop()
        if self.flying:
            self.podtp.command_land()
        self.podtp.disconnect()
        return True

    @overrides
    def _move(self, dx: float, dy: float):
        if not self.flying:
            self._take_off()

        print(f"-> Move by ({dx}, {dy}) m")
        if dx != 0:
            # self.podtp.command_position(self._cap_dist(dx) / 100.0, 0, 0, 0)
            for i in range(int(abs(dx) * 5 / self.xy_speed)):
                speed = self.xy_speed if dx > 0 else -self.xy_speed
                self.podtp.command_hover(speed, 0, 0, self.height)
                time.sleep(0.2)
            self.podtp.command_hover(0, 0, 0, self.height)
        time.sleep(EXECUTION_DELAY)

        if dy != 0:
            # self.podtp.command_position(0, self._cap_dist(dy) / 100.0, 0, 0)
            for i in range(int(abs(dy) * 5 / self.xy_speed)):
                speed = self.xy_speed if dy > 0 else -self.xy_speed
                self.podtp.command_hover(0, speed, 0, self.height)
                time.sleep(0.2)
            self.podtp.command_hover(0, 0, 0, self.height)
        time.sleep(EXECUTION_DELAY)

    @overrides
    def _rotate(self, deg: float):
        if not self.flying:
            self._take_off()
        print(f"-> Rotate by {deg} degrees")
        self.podtp.command_position(0, 0, 0, deg)
        time.sleep(abs(deg) / 360.0 * 4)
        self.podtp.command_hover(0, 0, 0, self.height)
    
    def lift(self, dist: float):
        if not self.flying:
            self._take_off()
        print(f"-> Lift for {dist} cm")
        self.height += dist / 100.0
        self.podtp.command_position(0, 0, self._cap_dist(dist) / 100.0, 0)
        time.sleep(EXECUTION_DELAY)
        self.podtp.command_hover(0, 0, 0, self.height)
    
    def land(self):
        if not self.flying:
            return
        
        self.flying = False
        print("-> Land")
        self.podtp.command_land()
        time.sleep(EXECUTION_DELAY)