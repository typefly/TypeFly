import cv2, time
import threading
from PIL import Image
from overrides import overrides

from ..skillset import SkillSet, SkillArg, SkillSetLevel
from ..robot_wrapper import RobotWrapper, RobotObservation
from ..yolo_client import YoloClient
from ..robot_info import RobotInfo

SKILL_EXECUTION_TIME = 0.2

class VirtualObservation(RobotObservation):
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        self.yolo_client = YoloClient(robot_info)

        if "capture" not in robot_info.extra:
            raise ValueError("Robot info must contain 'capture' key in extra, which is the camera index")

        self.cap: cv2.VideoCapture = None
        def _capture_spin():
            # must create the capture and read in the same thread
            self.cap = cv2.VideoCapture(int(self.robot_info.extra["capture"]))
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open GStreamer pipeline")
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                # Convert the frame to RGB and store it in self._image
                self._image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
        self.capture_thread = threading.Thread(target=_capture_spin)
    
    @overrides
    def _start(self):
        self.capture_thread.start()

    @overrides  
    def _stop(self):
        self.capture_thread.join()
        if self.cap is not None:
            self.cap.release()

    @overrides
    async def process_image(self, image: Image.Image):
        await self.yolo_client.detect(image)
    
    @overrides
    def fetch_processed_result(self) -> tuple[Image.Image, list]:
        return self.yolo_client.latest_result

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
                "definition": "8{?is_visible($1){->True}turn_cw(45)}->False",
                "description": "Rotate to find object $1 when it's *not* in current view"
            },
            {
                "name": "scan_description",
                "definition": "8{_1=probe($1);?_1!=False{->_1}turn_cw(45)}->False",
                "description": "Rotate to find object $1 when it's *not* in current view"
            },
            {
                "name": "orienting",
                "definition": "4{_1=ox($1);?_1>0.6{tc(15)};?_1<0.4{tu(15)};_2=ox($1);?_2<0.6&&_2>0.4{->True}}->False",
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
        time.sleep(SKILL_EXECUTION_TIME)
        return True, False

    @overrides
    def move_backward(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving backward {dist} cm")
        time.sleep(SKILL_EXECUTION_TIME)
        return True, False

    @overrides
    def move_left(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving left {dist} cm")
        time.sleep(SKILL_EXECUTION_TIME)
        return True, False

    @overrides
    def move_right(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving right {dist} cm")
        time.sleep(SKILL_EXECUTION_TIME)
        return True, False

    @overrides
    def turn_ccw(self, deg: int) -> tuple[bool, bool]:
        print(f"-> Turning CCW {deg} degrees")
        if deg >= 90:
            print("-> Turning CCW over 90 degrees")
            return True, False
        time.sleep(SKILL_EXECUTION_TIME)
        return True, False

    @overrides
    def turn_cw(self, deg: int) -> tuple[bool, bool]:
        print(f"-> Turning CW {deg} degrees")
        if deg >= 90:
            print("-> Turning CW over 90 degrees")
            return True, False
        time.sleep(SKILL_EXECUTION_TIME)
        return True, False
    
    def move_up(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving up {dist} cm")
        time.sleep(SKILL_EXECUTION_TIME)
        return True, False

    def move_down(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving down {dist} cm")
        time.sleep(SKILL_EXECUTION_TIME)
        return True, False