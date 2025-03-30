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
        self.ll_skillset.add_low_level_skill("lift", self.lift, "Move up/down by a distance", args=[SkillArg("dist", int)])
        
        high_level_skills = [
            {
                "name": "scan",
                "definition": "8{?is_visible($1){->True}rotate(45)}->False",
                "description": "Rotate to find object $1 when it's *not* in current view"
            },
            {
                "name": "scan_description",
                "definition": "8{_1=probe($1);?_1!=False{->_1}rotate(45)}->False",
                "description": "Rotate to find object $1 when it's *not* in current view"
            },
            {
                "name": "orienting",
                "definition": "4{_1=ox($1);?_1>0.6{rotate(-15)};?_1<0.4{rotate(15)};_2=ox($1);?_2<0.6&&_2>0.4{->True}}->False",
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
        # print(f"{self.hl_skillset}")

    @overrides
    def start(self) -> bool:
        self.observation.start()
        return True

    @overrides
    def stop(self):
        self.observation.stop()

    @overrides
    def move(self, dx: float, dy: float) -> tuple[bool, bool]:
        print(f"-> Move by ({dx}, {dy}) cm")
        time.sleep(SKILL_EXECUTION_TIME)
        return True, False

    @overrides
    def rotate(self, deg: float) -> tuple[bool, bool]:
        print(f"-> Rotate by {deg} degrees")
        time.sleep(SKILL_EXECUTION_TIME)
        return True, False
    
    def lift(self, dist: float) -> tuple[bool, bool]:
        print(f"-> Lift for {dist} cm")
        time.sleep(SKILL_EXECUTION_TIME)
        return True, False