from drone_wrapper import DroneWrapper
import cv2

class VirtualDroneWrapper(DroneWrapper):
    def __init__(self):
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

    def stop_stream(self):
        self.cap.release()

    def get_image(self):
        ret, frame = self.cap.read()
        # convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
