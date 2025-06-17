import time, cv2
import numpy as np
from PIL import Image
import threading
from overrides import overrides

from podtp import Podtp, sensor

from ..robot_wrapper import RobotWrapper
from ..skillset import SkillSet, SkillSetLevel
from ..robot_info import RobotInfo
from .pod_wrapper import PodObservation

EXECUTION_DELAY = 0.8

class GearWrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo, system_skill_func: list[callable]):
        self.podtp = Podtp(robot_info.extra)
        super().__init__(robot_info, PodObservation(self.podtp.sensor_data, robot_info), system_skill_func)
        self.xy_speed = 0.3
        
        high_level_skills = [
            {
                "name": "scan",
                "definition": "{8{?is_visible($1){->True}rotate(-45)}->False}",
                "description": "Rotate to find a specific object $1 when it's *not* in current view",
            },
            {
                "name": "scan_description",
                "definition": "{8{_1=probe($1);?_1!=False{->_1}rotate(-45)}->False}",
                "description": "Rotate to find an abstract object $1 when it's *not* in current view",
            },
            {
                "name": "orienting",
                "definition": "{_1=ox($1);rotate((0.5-_1)*80)}",
                "description": "Rotate to align with object $1",
            },
            {
                "name": "goto",
                "definition": "2{orienting($1);move(0.8, 0)}",
                "description": "Move to object $1 in the view (orienting then go forward)"
            }
        ]

        self.hl_skillset = SkillSet(SkillSetLevel.HIGH, self.ll_skillset)
        for skill in high_level_skills:
            self.hl_skillset.add_high_level_skill(skill['name'], skill['definition'], skill['description'])

    @overrides
    def start(self) -> bool:
        if not self.podtp.connect():
            print("Failed to connect to the car")
            return False
        self.podtp.ctrl_lock(False)
        self.podtp.start_stream()
        self.observation.start()
        return True

    @overrides
    def stop(self):
        self.observation.stop()
        self.podtp.disconnect()

    @overrides
    def move(self, dx: float, dy: float) -> tuple[bool, bool]:
        print(f"-> Move by ({dx}, {dy}) cm")
        if dx != 0:
            # self.podtp.command_position(self._cap_dist(dx) / 100.0, 0, 0, 0)
            for i in range(int(abs(dx) / 20 / self.xy_speed)):
                speed = self.xy_speed if dx > 0 else -self.xy_speed
                self.podtp.command_hover(speed, 0, 0, 0)
                time.sleep(0.2)
            self.podtp.command_hover(0, 0, 0, 0)
        time.sleep(EXECUTION_DELAY)

        if dy != 0:
            # self.podtp.command_position(0, self._cap_dist(dy) / 100.0, 0, 0)
            for i in range(int(abs(dy) / 20 / self.xy_speed)):
                speed = self.xy_speed if dy > 0 else -self.xy_speed
                self.podtp.command_hover(0, -speed, 0, 0)
                time.sleep(0.2)
            self.podtp.command_hover(0, 0, 0, 0)
        time.sleep(EXECUTION_DELAY)
        return True, False

    @overrides
    def rotate(self, deg: float) -> tuple[bool, bool]:
        print(f"-> Rotate by {deg} degrees")
        self.podtp.command_position(0, 0, 0, deg)
        time.sleep(abs(deg) / 360.0 * 4)
        self.podtp.command_hover(0, 0, 0, 0)
        return True, False
