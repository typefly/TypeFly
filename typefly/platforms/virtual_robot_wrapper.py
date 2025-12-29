import cv2, time
import threading
from PIL import Image
from overrides import overrides

from ..skillset import SkillItem
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
    def __init__(self, robot_info: RobotInfo):
        super().__init__(robot_info, VirtualObservation(robot_info))

        # Example of adding a skill
        self.skillset.add_skill(self.lift, "Lift the robot by a certain distance")

    @overrides
    def start(self) -> bool:
        self.observation.start()
        return True

    @overrides
    def stop(self) -> bool:
        self.observation.stop()
        return True

    @overrides
    def _move(self, dx: float, dy: float):
        print(f"-> Move by ({dx}, {dy}) cm")
        time.sleep(SKILL_EXECUTION_TIME)

    @overrides
    def _rotate(self, deg: float):
        print(f"-> Rotate by {deg} degrees")
        time.sleep(SKILL_EXECUTION_TIME)

    def lift(self, dist: float):
        print(f"-> Lift for {dist} cm")
        time.sleep(SKILL_EXECUTION_TIME)