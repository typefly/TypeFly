from djitellopy import Tello
from drone_wrapper import DroneWrapper
import time

class TelloWrapper(DroneWrapper):
    def __init__(self):
        self.drone = Tello()
        self.active_count = 0
        self.streamOn = False

    def keep_active(self):
        if self.active_count % 20 == 0:
            self.drone.send_control_command("command")
        self.active_count += 1

    def connect(self):
        self.drone.connect()

    def takeoff(self) -> bool:
        if not self.is_battery_good():
            return False
        else:
            self.drone.takeoff()
            return True

    def land(self):
        self.drone.land()

    def start_stream(self):
        self.streamOn = True
        self.drone.streamon()
        self.frame_reader = self.drone.get_frame_read()

    def stop_stream(self):
        self.streamOn = False
        self.drone.streamoff()

    def get_image(self):
        if not self.streamOn:
            return None
        return self.frame_reader.frame

    def move_forward(self, distance: int) -> bool:
        self.drone.move_forward(distance)
        time.sleep(0.5)
        return True

    def move_backward(self, distance: int) -> bool:
        self.drone.move_back(distance)
        time.sleep(0.5)
        return True

    def move_left(self, distance: int) -> bool:
        self.drone.move_left(distance)
        time.sleep(0.5)
        return True

    def move_right(self, distance: int) -> bool:
        self.drone.move_right(distance)
        time.sleep(0.5)
        return True

    def move_up(self, distance: int) -> bool:
        self.drone.move_up(distance)
        time.sleep(0.5)
        return True

    def move_down(self, distance: int) -> bool:
        self.drone.move_down(distance)
        time.sleep(0.5)
        return True

    def turn_ccw(self, degree: int) -> bool:
        self.drone.rotate_counter_clockwise(degree)
        time.sleep(1.5)
        return True

    def turn_cw(self, degree: int) -> bool:
        self.drone.rotate_clockwise(degree)
        time.sleep(1.5)
        return True
    
    def is_battery_good(self):
        self.battery = self.drone.query_battery()
        print(f"> Battery level: {self.battery}% ", end='')
        if self.battery < 20:
            print('is too low [WARNING]')
        else:
            print('[OK]')
            return True
        return False
