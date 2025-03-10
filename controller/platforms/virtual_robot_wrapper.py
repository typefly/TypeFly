import cv2, time
from typing import Optional, Tuple
from ..abs.robot_wrapper import RobotWrapper, RobotObservation

class VirtualObservation(RobotObservation):
    def __init__(self, cap):
        self.cap = cap
        if not self.cap.isOpened():
            raise ValueError("Could not open video device")

    @property
    def image(self):
        # Read a frame from the video capture
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

class VirtualRobotWrapper(RobotWrapper):
    def __init__(self):
        self.stream_on = False
        pass

    def keep_alive(self):
        return

    def start(self) -> bool:
        self.stream_on = True
        self.cap = cv2.VideoCapture(0)
        return True

    def stop(self):
        self.stream_on = False
        self.cap.release()

    def get_observation(self) -> Optional[RobotObservation]:
        if not self.stream_on:
            return None
        return VirtualObservation(self.cap)

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