import cv2, time
import asyncio
from PIL import Image
from typing import Optional
from overrides import overrides

from controller.skillset import LowLevelSkillItem, SkillSet, SkillArg, SkillSetLevel, HighLevelSkillItem
from ..robot_wrapper import RobotWrapper, RobotObservation
from ..yolo_client import YoloClient
from ..robot_info import RobotInfo

class VirtualObservation(RobotObservation):
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info)
        self.interval: float = 1.0 / rate
        self.yolo_client = YoloClient(robot_info)
    
    @overrides
    def _start(self):
        self.cap = cv2.VideoCapture(self.robot_info.extra["capture"])
        if not self.cap.isOpened():
            raise ValueError("Could not open video device")

    @overrides  
    def _stop(self):
        self.cap.release()

    @overrides
    def update_observation(self):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def schedule_tasks():
            tasks = set()
            
            while self.running:
                start_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    raise ValueError("Could not read frame")
                self._image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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

class VirtualRobotWrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo, system_skill_func: list[callable]):
        super().__init__(robot_info, VirtualObservation(robot_info), system_skill_func)

        # extra movement skills
        self.ll_skillset.add_low_level_skill("move_up", self.move_up, "Move up by a distance", args=[SkillArg("dist", int)])
        self.ll_skillset.add_low_level_skill("move_down", self.move_down, "Move down by a distance", args=[SkillArg("dist", int)])
        # print(f"{self.ll_skillset}")
        
        ### TODO: simplify the logic "?is_visible($1){->True}turn_cw(45)}->False"
        high_level_skills = [
            {
                "name": "scan",
                "definition": "8{?is_visible($1)==True{->True}turn_cw(45)}->False",
                "description": "Rotate to find object $1 when it's *not* in current view"
            },
            {
                "name": "scan_description",
                "definition": "8{_1=probe($1);?_1!=False{->_1}turn_cw(45)}->False",
                "description": "Rotate to find object $1 when it's *not* in current view"
            },
            {
                "name": "orienting",
                "definition": "4{_1=ox($1);?_1>0.6{tc(15)};?_1<0.4{tu(15)};_2=ox($1);?_2<0.6&_2>0.4{->True}}->False",
                "description": "Rotate to align with object $1",
            },
            {
                "name": "goto",
                "definition": "?orienting($1)==True{move_forward(80)}",
                "description": "Move to object $1 in the view"
            }
        ]

        self.hl_skillset = SkillSet(SkillSetLevel.HIGH, self.ll_skillset)
        for skill in high_level_skills:
            self.hl_skillset.add_high_level_skill(skill['name'], skill['definition'], skill['description'])
        # print(f"{self.hl_skillset}")

    @overrides
    def start(self) -> bool:
        self.observation.start()
        return True

    @overrides
    def stop(self):
        self.observation.stop()

    @overrides
    def move_forward(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving forward {dist} cm")
        time.sleep(1)
        return True, False

    @overrides
    def move_backward(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving backward {dist} cm")
        time.sleep(1)
        return True, False

    @overrides
    def move_left(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving left {dist} cm")
        time.sleep(1)
        return True, False

    @overrides
    def move_right(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving right {dist} cm")
        time.sleep(1)
        return True, False

    @overrides
    def turn_ccw(self, deg: int) -> tuple[bool, bool]:
        print(f"-> Turning CCW {deg} degrees")
        if deg >= 90:
            print("-> Turning CCW over 90 degrees")
            return True, False
        time.sleep(1)
        return True, False

    @overrides
    def turn_cw(self, deg: int) -> tuple[bool, bool]:
        print(f"-> Turning CW {deg} degrees")
        if deg >= 90:
            print("-> Turning CW over 90 degrees")
            return True, False
        time.sleep(1)
        return True, False
    
    def move_up(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving up {dist} cm")
        time.sleep(1)
        return True, False

    def move_down(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving down {dist} cm")
        time.sleep(1)
        return True, False