import time, cv2
import numpy as np
from typing import Optional, Tuple
from djitellopy import Tello

from ..abs.robot_wrapper import RobotWrapper, RobotObservation
from ..skillset import SkillSet, LowLevelSkillItem, SkillArg, SkillSetLevel, HighLevelSkillItem
from ..vision_skill_wrapper import VisionSkillWrapper
from ..yolo_client import YoloClient

import logging
Tello.LOGGER.setLevel(logging.WARNING)

MOVEMENT_MIN = 20
MOVEMENT_MAX = 300

SCENE_CHANGE_DISTANCE = 120
SCENE_CHANGE_ANGLE = 90

def adjust_exposure(img, alpha=1.0, beta=0):
    """
    Adjust the exposure of an image.
    
    :param img: Input image
    :param alpha: Contrast control (1.0-3.0). Higher values increase exposure.
    :param beta: Brightness control (0-100). Higher values add brightness.
    :return: Exposure adjusted image
    """
    # Apply exposure adjustment using the formula: new_img = img * alpha + beta
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_img

def sharpen_image(img):
    """
    Apply a sharpening filter to an image.
    
    :param img: Input image
    :return: Sharpened image
    """
    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    # Apply the sharpening filter
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

class TelloObservation(RobotObservation):
    def __init__(self, frame_reader, asyncio_loop, rate: int = 10):
        self.interval: float = 1.0 / rate
        self.asyncio_loop = asyncio_loop
        self.yolo_client = YoloClient()
        self.frame_reader = frame_reader
    
    def update_observation(self):
        # Read frame from drone
        while self.running:
            start_time = time.time()
            frame = self.frame_reader.frame
            frame = adjust_exposure(frame, alpha=1.3, beta=-30)
            self._image = sharpen_image(frame)
            self.asyncio_loop.call_soon_threadsafe(self.yolo_client.detect, self._image)
            elapsed_time = time.time() - start_time
            time.sleep(max(0, self.interval - elapsed_time))
        
def cap_distance(distance):
    if distance < MOVEMENT_MIN:
        return MOVEMENT_MIN
    elif distance > MOVEMENT_MAX:
        return MOVEMENT_MAX
    return distance

class TelloWrapper(RobotWrapper):
    def __init__(self):
        self.drone = Tello()
        self.observation: RobotObservation = None
        self.vision: VisionSkillWrapper = None
        self.alive_count = 0
        self.stream_on = False

        self.low_level_skillset = SkillSet()
        # movement skills
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_forward", self.move_forward, "Move forward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_backward", self.move_backward, "Move backward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_left", self.move_left, "Move left by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_right", self.move_right, "Move right by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_up", self.move_up, "Move up by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_down", self.move_down, "Move down by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_cw", self.turn_cw, "Rotate clockwise/right by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_ccw", self.turn_ccw, "Rotate counterclockwise/left by certain degrees", args=[SkillArg("degrees", int)]))
        # vision skills
        self.low_level_skillset.add_skill(LowLevelSkillItem("is_visible", self.vision.is_visible, "Check the visibility of target object", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_x", self.vision.object_x, "Get object's X-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_y", self.vision.object_y, "Get object's Y-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_width", self.vision.object_width, "Get object's width in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_height", self.vision.object_height, "Get object's height in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_dis", self.vision.object_distance, "Get object's distance in cm", args=[SkillArg("object_name", str)]))

        high_level_skills = [
            {
                "name": "scan",
                "definition": """
                8{?}
""",
                "description": "Rotate to find object $1 when it's *not* in current scene",
            }
        ]


        self.high_level_skillset = SkillSet(SkillSetLevel.HIGH, self.low_level_skillset)
        self.high_level_skillset.add_skill(HighLevelSkillItem("scan", "", "Rotate to find object $1 when it's *not* in current scene"))

    def start(self) -> bool:
        self.drone.connect()
        if not self._is_battery_good():
            return False
        else:
            self.drone.takeoff()
        self.move_up(25)
        self.stream_on = True
        self.drone.streamon()
        return True

    def stop(self):
        self.drone.land()
        self.stream_on = False
        self.drone.streamoff()

    def keep_alive(self):
        if self.alive_count % 20 == 0:
            self.drone.send_control_command("command")
        self.alive_count += 1

    def get_observation(self) -> Optional[RobotObservation]:
        if not self.stream_on:
            return None
        return TelloObservation(self.drone.get_frame_read())

    def move_forward(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_forward(cap_distance(distance))
        time.sleep(0.5)
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_backward(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_back(cap_distance(distance))
        time.sleep(0.5)
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_left(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_left(cap_distance(distance))
        time.sleep(0.5)
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_right(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_right(cap_distance(distance))
        time.sleep(0.5)
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_up(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_up(cap_distance(distance))
        time.sleep(0.5)
        return True, False

    def move_down(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_down(cap_distance(distance))
        time.sleep(0.5)
        return True, False

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        self.drone.rotate_counter_clockwise(degree)
        time.sleep(1)
        # return True, degree > SCENE_CHANGE_ANGLE
        return True, False

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        self.drone.rotate_clockwise(degree)
        time.sleep(1)
        # return True, degree > SCENE_CHANGE_ANGLE
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