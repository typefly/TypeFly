import time, cv2
import numpy as np
from djitellopy import Tello
from PIL import Image
import threading
from overrides import overrides

from ..robot_wrapper import RobotWrapper, RobotObservation
from ..skillset import SkillSet, SkillArg, SkillSetLevel
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
        self.alive_count = 0

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

    def keep_alive(self):
        self.alive_count += 1
        if self.alive_count > 15:
            self.drone.send_control_command("command")
            self.alive_count = 0
    
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
    def fetch_processed_result(self) -> tuple[Image.Image, list]:
        return self.yolo_client.latest_result

class TelloWrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo, system_skill_func: list[callable]):
        self.drone = Tello()
        super().__init__(robot_info, TelloObservation(self.drone, robot_info), system_skill_func)

        # extra movement skills
        self.ll_skillset.add_low_level_skill("lift", self.lift, "Move up/down by a distance", args=[SkillArg("dist", float)])
        
        high_level_skills = [
            {
                "name": "scan",
                "definition": "{8{?is_visible($1){->True}rotate(45)}->False}",
                "description": "Rotate to find a specific object $1 when it's *not* in current view",
            },
            {
                "name": "scan_description",
                "definition": "{8{_1=probe($1);?_1!=False{->_1}rotate(45)}->False}",
                "description": "Rotate to find an abstract object $1 when it's *not* in current view",
            },
            {
                "name": "orienting",
                "definition": "4{_1=ox($1);?_1>0.6{rotate(-15)}:?_1<0.4{rotate(15)}:{->True}}->False",
                "description": "Rotate to align with object $1",
            },
            {
                "name": "goto",
                "definition": "?orienting($1){move(80, 0)}",
                "description": "Move to object $1 in the view (orienting then go forward)"
            }
        ]

        self.hl_skillset = SkillSet(SkillSetLevel.HIGH, self.ll_skillset)
        for skill in high_level_skills:
            self.hl_skillset.add_high_level_skill(skill['name'], skill['definition'], skill['description'])

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
            return False
        else:
            self.drone.takeoff()
        # self.move_up(25)
        self.observation.start()
        return True

    @overrides
    def stop(self):
        self.drone.land()
        self.observation.stop()

    @overrides
    def move(self, dx: float, dy: float) -> tuple[bool, bool]:
        print(f"-> Move by ({dx}, {dy}) cm")
        if dx > 0:
            self.drone.move_forward(self._cap_dist(dx))
        elif dx < 0:
            self.drone.move_back(self._cap_dist(-dx))
        time.sleep(EXECUTION_DELAY)

        if dy > 0:
            self.drone.move_left(self._cap_dist(dy))
        elif dy < 0:
            self.drone.move_right(self._cap_dist(-dy))
        time.sleep(EXECUTION_DELAY)
        return True, False

    @overrides
    def rotate(self, deg: float) -> tuple[bool, bool]:
        print(f"-> Rotate by {deg} degrees")
        self.drone.rotate_counter_clockwise(deg) if deg > 0 else self.drone.rotate_clockwise(-deg)
        time.sleep(EXECUTION_DELAY)
        return True, False
    
    def lift(self, dist: float) -> tuple[bool, bool]:
        print(f"-> Lift for {dist} cm")
        self.drone.move_up(self._cap_dist(dist)) if dist > 0 else self.drone.move_down(self._cap_dist(-dist))
        time.sleep(EXECUTION_DELAY)
        return True, False
    
    def _is_battery_good(self) -> bool:
        self.battery = self.drone.query_battery()
        print(f"> Battery level: {self.battery}% ", end='')
        if self.battery < 10:
            print('is too low [WARNING]')
        else:
            print('[OK]')
            return True
        return False