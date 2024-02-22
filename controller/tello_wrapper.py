import time
from typing import Tuple
from djitellopy import Tello

from .abs.drone_wrapper import DroneWrapper

MOVEMENT_MIN = 20
MOVEMENT_MAX = 300

SCENE_CHANGE_DISTANCE = 120
SCENE_CHANGE_ANGLE = 90

def cap_distance(distance):
    if distance < MOVEMENT_MIN:
        return MOVEMENT_MIN
    elif distance > MOVEMENT_MAX:
        return MOVEMENT_MAX
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

    def move_forward(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_forward(cap_distance(distance))
        time.sleep(0.5)
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_backward(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_back(cap_distance(distance))
        time.sleep(0.5)
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_left(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_left(cap_distance(distance))
        time.sleep(0.5)
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_right(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_right(cap_distance(distance))
        time.sleep(0.5)
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_up(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_up(cap_distance(distance))
        time.sleep(0.5)
        return True, False

    def move_down(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_down(cap_distance(distance))
        time.sleep(0.5)
        return True, False

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        self.drone.rotate_counter_clockwise(degree)
        time.sleep(2.5)
        return True, degree > SCENE_CHANGE_ANGLE

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        self.drone.rotate_clockwise(degree)
        time.sleep(2.5)
        return True, degree > SCENE_CHANGE_ANGLE
    
    def is_battery_good(self) -> bool:
        self.battery = self.drone.query_battery()
        print(f"> Battery level: {self.battery}% ", end='')
        if self.battery < 20:
            print('is too low [WARNING]')
        else:
            print('[OK]')
            return True
        return False
