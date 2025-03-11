import cv2, time
import asyncio
from PIL import Image
from typing import Optional, Tuple
from ..abs.robot_wrapper import RobotWrapper, RobotObservation
from ..yolo_client import YoloClient
from ..robot_info import RobotInfo

class VirtualObservation(RobotObservation):
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info)
        self.interval: float = 1.0 / rate
        self.yolo_client = YoloClient(robot_info)
        
    def _start(self):
        self.cap = cv2.VideoCapture(self.robot_info.extra["capture"])
        if not self.cap.isOpened():
            raise ValueError("Could not open video device")
        
    def _stop(self):
        self.cap.release()

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
                
                self._image_process_result = self.yolo_client.latest_result
                # Sleep for the interval
                elapsed_time = time.time() - start_time
                await asyncio.sleep(max(0, self.interval - elapsed_time))
        # Run the async function in the event loop
        loop.run_until_complete(schedule_tasks())

class VirtualRobotWrapper(RobotWrapper):
    def __init__(self, info):
        self.robot_info = info
        self.observation = VirtualObservation(info)

    def keep_alive(self):
        return

    def start(self) -> bool:
        self.observation.start()
        return True

    def stop(self):
        self.observation.stop()

    def get_observation(self) -> Optional[RobotObservation]:
        return self.observation

    def move_forward(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving forward {distance} cm")
        time.sleep(1)
        return True, False

    def move_backward(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving backward {distance} cm")
        time.sleep(1)
        return True, False

    def move_left(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving left {distance} cm")
        time.sleep(1)
        return True, False

    def move_right(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving right {distance} cm")
        time.sleep(1)
        return True, False

    def move_up(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving up {distance} cm")
        time.sleep(1)
        return True, False

    def move_down(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving down {distance} cm")
        time.sleep(1)
        return True, False

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CCW {degree} degrees")
        if degree >= 90:
            print("-> Turning CCW over 90 degrees")
            return True, False
        time.sleep(1)
        return True, False

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CW {degree} degrees")
        if degree >= 90:
            print("-> Turning CW over 90 degrees")
            return True, False
        time.sleep(1)
        return True, False