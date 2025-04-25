import time, cv2
import numpy as np
from PIL import Image
import threading
from overrides import overrides

from podtp import Podtp, sensor

from ..robot_wrapper import RobotWrapper, RobotObservation
from ..skillset import SkillSet, SkillArg, SkillSetLevel
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
    def fetch_processed_result(self) -> tuple[Image.Image, list]:
        return self.yolo_client.latest_result

class PodWrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo, system_skill_func: list[callable]):
        self.podtp = Podtp(robot_info.extra)
        super().__init__(robot_info, PodObservation(self.podtp.sensor_data, robot_info), system_skill_func)

        # extra movement skills
        self.ll_skillset.add_low_level_skill("lift", self.lift, "Move up/down by a distance", args=[SkillArg("dist", float)])
        self.ll_skillset.add_low_level_skill("land", self.land, "Land the drone")
        
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
        if abs(dist) < MOVEMENT_MIN:
            return MOVEMENT_MIN if dist > 0 else -MOVEMENT_MIN
        elif abs(dist) > MOVEMENT_MAX:
            return MOVEMENT_MAX if dist > 0 else -MOVEMENT_MAX
        return dist

    @overrides
    def start(self) -> bool:
        if self.podtp.connect():
            if not self.podtp.ctrl_lock(False):
                print("Failed to unlock control")
                return False
            else:
                self.podtp.start_stream()
                self._take_off_from_dog()
                print("Drone started")
        else:
            print("Failed to connect to the drone")
            return False
        self.observation.start()
        return True
    
    def _take_off_from_dog(self):
        # dog is around 40cm high
        self.podtp.reset_estimator(0)
        count = 0
        while count < 15:
            self.podtp.command_hover(0, 0, 0, 0.6)
            time.sleep(0.2)
            count += 1
        time.sleep(1)
        self.podtp.command_position(0, 0, 0, 0)

    @overrides
    def stop(self):
        self.observation.stop()
        self.podtp.command_land()
        self.podtp.disconnect()

    @overrides
    def move(self, dx: float, dy: float) -> tuple[bool, bool]:
        print(f"-> Move by ({dx}, {dy}) cm")
        if dx != 0:
            self.podtp.command_position(self._cap_dist(dx) / 100.0, 0, 0, 0)
        time.sleep(EXECUTION_DELAY)

        if dy != 0:
            self.podtp.command_position(0, self._cap_dist(dx) / 100.0, 0, 0)
        time.sleep(EXECUTION_DELAY)
        return True, False

    @overrides
    def rotate(self, deg: float) -> tuple[bool, bool]:
        print(f"-> Rotate by {deg} degrees")
        self.podtp.command_position(0, 0, 0, deg)
        time.sleep(EXECUTION_DELAY)
        return True, False
    
    def lift(self, dist: float) -> tuple[bool, bool]:
        print(f"-> Lift for {dist} cm")
        self.podtp.command_position(0, 0, self._cap_dist(dist) / 100.0, 0)
        time.sleep(EXECUTION_DELAY)
        return True, False
    
    def land(self) -> tuple[bool, bool]:
        print("-> Land")
        self.podtp.command_land()
        time.sleep(EXECUTION_DELAY)
        return True, False