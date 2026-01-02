import time
from typing import Any
from djitellopy import Tello
from PIL import Image
import threading
from overrides import overrides

from ..robot_wrapper import RobotWrapper, RobotObservation
from ..yolo_client import YoloClient
from ..robot_info import RobotInfo

import logging
Tello.LOGGER.setLevel(logging.WARNING)

MOVEMENT_MIN = 20
MOVEMENT_MAX = 300

SCENE_CHANGE_DIST = 300
SCENE_CHANGE_ANGLE = 360

EXECUTION_DELAY = 0.8

class TelloObservation(RobotObservation):
    def __init__(self, drone: Tello, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        self.drone = drone
        self.yolo_client = YoloClient(robot_info)

        def _capture_spin():
            frame_reader = self.drone.get_frame_read()
            while self.running:
                frame = None
                if frame_reader:
                    frame = frame_reader.frame
                # Convert the frame to RGB and store it in self._image
                if frame is not None:
                    self._image = Image.fromarray(frame)
                time.sleep(0.1)
        self.capture_thread = threading.Thread(target=_capture_spin)
    
    @overrides
    def _start(self):
        self.drone.streamon()
        self.capture_thread.start()
    
    @overrides
    def _stop(self):
        self.capture_thread.join()
        self.drone.streamoff()

    @overrides
    async def process_image(self, image: Image.Image):
        await self.yolo_client.detect(image)
    
    @overrides
    def fetch_processed_result(self) -> dict[str, Any]:
        _, object_list = self.yolo_client.latest_result
        return {
            "yolo": object_list
        }

class TelloWrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo):
        self.drone = Tello()
        super().__init__(robot_info, TelloObservation(self.drone, robot_info))

        # extra movement skills
        self.skillset.add_skill(self.lift, "Move up/down by a distance")


        self.last_command_time = time.time()
        self.keep_alive_thread = threading.Thread(target=self.keep_alive)
        self.running = True

    def keep_alive(self):
        while self.running:
            if time.time() - self.last_command_time > 4:
                self.drone.send_control_command("command")
                self.last_command_time = time.time()
            time.sleep(0.5)

    def _cap_dist(self, dist):
        if dist < MOVEMENT_MIN:
            return MOVEMENT_MIN
        elif dist > MOVEMENT_MAX:
            return MOVEMENT_MAX
        return dist

    @overrides
    def start(self) -> bool:
        self.drone.connect()
        if not self._is_battery_good():
            self.log("Battery is too low")
            return False
        else:
            self.drone.takeoff()
        # self.move_up(25)
        self.obs.start()
        self.keep_alive_thread.start()
        self.running = True
        return True

    @overrides
    def stop(self) -> bool:
        self.drone.land()
        self.obs.stop()
        self.running = False
        self.keep_alive_thread.join()
        return True

    @overrides
    def _move(self, dx: float, dy: float):
        print(f"-> Move by ({dx}, {dy}) m")
        dx = int(dx * 100)
        dy = int(dy * 100)
        if dx > 0:
            self.drone.move_forward(self._cap_dist(dx))
        elif dx < 0:
            self.drone.move_back(self._cap_dist(-dx))
        time.sleep(EXECUTION_DELAY)

        if dy > 0:
            self.drone.move_left(self._cap_dist(dy))
        elif dy < 0:
            self.drone.move_right(self._cap_dist(-dy))
        self.last_command_time = time.time()
        time.sleep(EXECUTION_DELAY)

    @overrides
    def _rotate(self, deg: float):
        print(f"-> Rotate by {deg} degrees")
        self.drone.rotate_counter_clockwise(deg) if deg > 0 else self.drone.rotate_clockwise(-deg)
        self.last_command_time = time.time()
        time.sleep(EXECUTION_DELAY)
    
    def lift(self, dist: float):
        print(f"-> Lift for {dist} cm")
        self.drone.move_up(self._cap_dist(dist)) if dist > 0 else self.drone.move_down(self._cap_dist(-dist))
        self.last_command_time = time.time()
        time.sleep(EXECUTION_DELAY)
    
    def _is_battery_good(self) -> bool:
        self.battery = self.drone.query_battery()
        print(f"> Battery level: {self.battery}% ", end='')
        if self.battery < 10:
            print('is too low [WARNING]')
        else:
            print('[OK]')
            return True
        return False