import cv2
from .drone_wrapper import DroneWrapper

class FrameReader:
    def __init__(self, cap):
        # Initialize the video capture
        self.cap = cap
        if not self.cap.isOpened():
            raise ValueError("Could not open video device")

    @property
    def frame(self):
        # Read a frame from the video capture
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

class VirtualDroneWrapper(DroneWrapper):
    def __init__(self):
        self.stream_on = False
        pass

    def keep_active(self):
        pass

    def connect(self):
        pass

    def takeoff(self) -> bool:
        return True

    def land(self):
        pass

    def start_stream(self):
        self.cap = cv2.VideoCapture(0)
        self.stream_on = True

    def stop_stream(self):
        self.cap.release()
        self.stream_on = False

    def get_frame_reader(self):
        if not self.stream_on:
            return None
        return FrameReader(self.cap)

    def move_forward(self, distance: int) -> bool:
        return True

    def move_backward(self, distance: int) -> bool:
        return True

    def move_left(self, distance: int) -> bool:
        return True

    def move_right(self, distance: int) -> bool:
        return True

    def move_up(self, distance: int) -> bool:
        return True

    def move_down(self, distance: int) -> bool:
        return True

    def turn_ccw(self, degree: int) -> bool:
        return True

    def turn_cw(self, degree: int) -> bool:
        return True
