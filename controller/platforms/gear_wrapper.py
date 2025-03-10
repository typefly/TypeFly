import time, os
from typing import Tuple
from .abs.robot_wrapper import RobotWrapper
from podtp import Podtp
import torch
import torch.nn as nn
import numpy as np

DEFAULT_NO_VALID_READING = 0
SAFE_DISTANCE_THRESHOLD = 250
SIDE_DISTANCE_THRESHOLD = 65
JUMP_DISTANCE_THRESHOLD = 60

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_sensor_data(raw_data):
    cleaned_data = raw_data[:]  # Create a copy of the raw data for cleaning

    for i in range(len(cleaned_data)):
        if cleaned_data[i] < 0:
            valid_previous = None
            valid_next = None

            # Find the previous valid value
            for j in range(i-1, -1, -1):
                if cleaned_data[j] >= 0:
                    valid_previous = cleaned_data[j]
                    break

            # Find the next valid value
            for k in range(i+1, len(cleaned_data)):
                if cleaned_data[k] >= 0:
                    valid_next = cleaned_data[k]
                    break

            # Decide what value to assign to the bad reading
            if valid_previous is not None and valid_next is not None:
                # Average if both previous and next valid values are found
                cleaned_data[i] = (valid_previous + valid_next) / 2
            elif valid_previous is not None:
                # Use the previous if only it is available
                cleaned_data[i] = valid_previous
            elif valid_next is not None:
                # Use the next if only it is available
                cleaned_data[i] = valid_next
            else:
                # If no valid readings are available, handle it with a default value or recheck
                cleaned_data[i] = DEFAULT_NO_VALID_READING

    return cleaned_data

# Define the model
class DirectionPredictor(nn.Module):
    def __init__(self):
        super(DirectionPredictor, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.mean = 381.9494323730469
        self.std = 306.9201965332031

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class GearWrapper(RobotWrapper):
    def __init__(self):
        self.stream_on = False
        config = {
            'ip': '192.168.8.169',
            'ip1': '192.168.8.195',
            'port': 80,
            'stream_port': 81
        }
        self.robot = Podtp(config)
        self.move_speed_x = 2.5
        self.move_speed_y = 2.8
        self.unlock_count = 0
        self.model = DirectionPredictor()
        self.model.load_state_dict(torch.load(os.path.join(CURRENT_DIR, 'assets/gear/model.pth')))
        self.model.eval()

    def keep_active(self):
        self.unlock_count += 1
        if self.unlock_count > 100:
            self.robot.send_ctrl_lock(False)
            self.unlock_count = 0

    def connect(self):
        if not self.robot.connect():
            raise ValueError("Could not connect to the robot")
        if not self.robot.send_ctrl_lock(False):
            raise ValueError("Could not unlock the robot control")

    def takeoff(self) -> bool:
        return True

    def land(self):
        pass

    def start_stream(self):
        self.robot.start_stream()
        self.stream_on = True

    def stop_stream(self):
        self.robot.stop_stream()
        self.stream_on = False

    def get_frame_reader(self):
        if not self.stream_on:
            return None
        return self.robot.sensor_data
    
    def move_forward(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving forward {distance} cm")
        self.robot.send_command_hover(0, 0, 0, 0)
        small_move = distance <= 15
        while distance > 0:
            if small_move:
                self.robot.send_command_hover(0, self.move_speed_x, 0, 0)
            else:
                array = self.robot.sensor_data.depth.data
                left_distance = clean_sensor_data(array[0, :])
                front_distance = clean_sensor_data(array[2, :])
                right_distance = clean_sensor_data(array[7, :])
                if max(front_distance) < 50:
                    self.move_backward(10)

                x = np.concatenate((left_distance, front_distance, right_distance))
                x = torch.tensor(x, dtype=torch.float32)
                x = (x - self.model.mean) / self.model.std
                y = self.model(x.unsqueeze(0)).squeeze(0)
                command = torch.argmax(y).item() - 1
                
                left_margin = min(left_distance)
                right_margin = min(right_distance)
                if left_margin > SIDE_DISTANCE_THRESHOLD and right_margin > SIDE_DISTANCE_THRESHOLD:
                    vy = 0
                elif left_margin > SIDE_DISTANCE_THRESHOLD:
                    vy = -1.5
                elif right_margin > SIDE_DISTANCE_THRESHOLD:
                    vy = 1.5
                else:
                    if abs(left_margin - right_margin) > 80:
                        if left_margin < right_margin:
                            vy = 1.5
                        else:
                            vy = -1.5

                if command == 0:
                    self.robot.send_command_hover(0, self.move_speed_x, vy, 0)
                elif command == 1:
                    self.turn_ccw(30)
                elif command == -1:
                    self.turn_cw(30)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_backward(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving backward {distance} cm")
        self.robot.send_command_hover(0, 0, 0, 0)
        while distance > 0:
            self.robot.send_command_hover(0, -self.move_speed_x, 0, 0)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_left(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving left {distance} cm")
        self.robot.send_command_hover(0, 0, 0, 0)
        while distance > 0:
            self.robot.send_command_hover(0, 0, -self.move_speed_y, 0)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_right(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving right {distance} cm")
        self.robot.send_command_hover(0, 0, 0, 0)
        while distance > 0:
            self.robot.send_command_hover(0, 0, self.move_speed_y, 0)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_up(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving up {distance} cm")
        return True, False

    def move_down(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving down {distance} cm")
        return True, False

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CCW {degree} degrees")
        self.robot.send_command_hover(0, 0, 0, 0)
        self.robot.send_command_position(0, 0, 0, degree)
        time.sleep(1 + degree / 50.0)
        self.robot.send_command_hover(0, 0, 0, 0)
        # if degree >= 90:
        #     print("-> Turning CCW over 90 degrees")
        #     return True, True
        return True, False

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CW {degree} degrees")
        self.robot.send_command_hover(0, 0, 0, 0)
        self.robot.send_command_position(0, 0, 0, -degree)
        time.sleep(1 + degree / 50.0)
        self.robot.send_command_hover(0, 0, 0, 0)
        # if degree >= 90:
        #     print("-> Turning CW over 90 degrees")
        #     return True, True
        return True, False
    
    def move_in_circle(self, cw) -> Tuple[bool, bool]:
        if cw:
            vy = -8
            vr = -12
        else:
            vy = 8
            vr = 12
        for i in range(50):
            self.robot.send_command_hover(0, 0, vy, vr)
            time.sleep(0.1)
        self.robot.send_command_hover(0, 0, 0, 0)
        return True, False
