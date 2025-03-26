import time, cv2
import numpy as np
from djitellopy import Tello
from PIL import Image
import asyncio
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
    def __init__(self, drone, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info)
        self.drone = drone
        self.interval: float = 1.0 / rate
        self.yolo_client = YoloClient(robot_info)
        self.alive_count = 0

    def keep_alive(self):
        self.alive_count += 1
        if self.alive_count > 15:
            self.drone.send_control_command("command")
            self.alive_count = 0
    
    @overrides
    def _start(self):
        self.drone.streamon()
    
    @overrides
    def _stop(self):
        self.drone.streamoff()

    @overrides
    def update_observation(self):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def schedule_tasks():
            tasks = set()
            
            while self.running:
                self.keep_alive()

                start_time = time.time()
                frame = self.drone.get_frame_read().frame
                self._image = Image.fromarray(frame)
                # Add a new task to the set
                task = asyncio.create_task(self.yolo_client.detect(self._image))
                tasks.add(task)
                
                # Clean up completed tasks
                tasks = {t for t in tasks if not t.done()}
                with self._image_process_lock:
                    self._image_process_result = self.yolo_client.latest_result
                # Sleep for the interval
                elapsed_time = time.time() - start_time
                await asyncio.sleep(max(0, self.interval - elapsed_time))
        # Run the async function in the event loop
        loop.run_until_complete(schedule_tasks())

class TelloWrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo, system_skill_func: list[callable]):
        self.drone = Tello()
        super().__init__(robot_info, TelloObservation(self.drone, robot_info), system_skill_func)

        # extra movement skills
        self.ll_skillset.add_low_level_skill("move_up", self.move_up, "Move up by a distance", args=[SkillArg("dist", int)])
        self.ll_skillset.add_low_level_skill("move_down", self.move_down, "Move down by a distance", args=[SkillArg("dist", int)])
        
        high_level_skills = [
            {
                "name": "scan",
                "definition": "{8{?is_visible($1){->True}turn_cw(45)}->False}",
                "description": "Rotate to find object $1 when it's *not* in current view",
            },
            {
                "name": "scan_description",
                "definition": "{8{_1=probe($1);?_1!=False{->_1}turn_cw(45)}->False}",
                "description": "Rotate to find object $1 when it's *not* in current view",
            },
            {
                "name": "orienting",
                "definition": "4{_1=ox($1);?_1>0.6{tc(15)}:?_1<0.4{tu(15)}:{->True}}->False",
                "description": "Rotate to align with object $1",
            },
            {
                "name": "goto",
                "definition": "?orienting($1){move_forward(80)}",
                "description": "Move to object $1 in the view"
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
        self.move_up(25)
        self.observation.start()
        return True

    @overrides
    def stop(self):
        self.drone.land()
        self.observation.stop()

    @overrides
    def move_forward(self, dist: int) -> tuple[bool, bool]:
        self.drone.move_forward(self._cap_dist(dist))
        time.sleep(0.5)
        return True, dist > SCENE_CHANGE_DIST

    @overrides
    def move_backward(self, dist: int) -> tuple[bool, bool]:
        self.drone.move_back(self._cap_dist(dist))
        time.sleep(0.5)
        return True, dist > SCENE_CHANGE_DIST

    @overrides
    def move_left(self, dist: int) -> tuple[bool, bool]:
        self.drone.move_left(self._cap_dist(dist))
        time.sleep(0.5)
        return True, dist > SCENE_CHANGE_DIST

    @overrides
    def move_right(self, dist: int) -> tuple[bool, bool]:
        self.drone.move_right(self._cap_dist(dist))
        time.sleep(0.5)
        return True, dist > SCENE_CHANGE_DIST

    @overrides
    def turn_ccw(self, deg: int) -> tuple[bool, bool]:
        self.drone.rotate_counter_clockwise(deg)
        time.sleep(1)
        # return True, deg > SCENE_CHANGE_ANGLE
        return True, False

    @overrides
    def turn_cw(self, deg: int) -> tuple[bool, bool]:
        self.drone.rotate_clockwise(deg)
        time.sleep(1)
        # return True, deg > SCENE_CHANGE_ANGLE
        return True, False
    
    def move_up(self, dist: int) -> tuple[bool, bool]:
        self.drone.move_up(self._cap_dist(dist))
        time.sleep(0.5)
        return True, False

    def move_down(self, dist: int) -> tuple[bool, bool]:
        self.drone.move_down(self._cap_dist(dist))
        time.sleep(0.5)
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