import time
from djitellopy import Tello

from .abs.drone_wrapper import DroneWrapper

def cap_distance(distance):
    if distance < 20:
        return 20
    elif distance > 100:
        return 100
    return distance

class TelloWrapper(DroneWrapper):
    def __init__(self):
        self.drone = Tello()
        self.active_count = 0
        self.stream_on = False

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
        self.stream_on = True
        self.drone.streamon()

    def stop_stream(self):
        self.stream_on = False
        self.drone.streamoff()

    def get_frame_reader(self):
        if not self.stream_on:
            return None
        return self.drone.get_frame_read()

    def move_forward(self, distance: int) -> bool:
        self.drone.move_forward(cap_distance(distance))
        time.sleep(0.5)
        return True

    def move_backward(self, distance: int) -> bool:
        self.drone.move_back(cap_distance(distance))
        time.sleep(0.5)
        return True

    def move_left(self, distance: int) -> bool:
        self.drone.move_left(cap_distance(distance))
        time.sleep(0.5)
        return True

    def move_right(self, distance: int) -> bool:
        self.drone.move_right(cap_distance(distance))
        time.sleep(0.5)
        return True

    def move_up(self, distance: int) -> bool:
        self.drone.move_up(cap_distance(distance))
        time.sleep(0.5)
        return True

    def move_down(self, distance: int) -> bool:
        self.drone.move_down(cap_distance(distance))
        time.sleep(0.5)
        return True

    def turn_ccw(self, degree: int) -> bool:
        self.drone.rotate_counter_clockwise(degree)
        time.sleep(0.5)
        return True

    def turn_cw(self, degree: int) -> bool:
        self.drone.rotate_clockwise(degree)
        time.sleep(0.5)
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
