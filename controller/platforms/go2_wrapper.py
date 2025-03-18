import time, os
from ..robot_wrapper import RobotWrapper, RobotObservation
from ..robot_info import RobotInfo
from ..yolo_client import YoloClient
import torch
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class TelloObservation(RobotObservation):
    def __init__(self, drone, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info)
        self.drone = drone
        self.interval: float = 1.0 / rate
        self.yolo_client = YoloClient(robot_info)
        self.alive_count = 0

    def keep_alive(self):
        self.alive_count += 1
        if self.alive_count > 15:
            self.drone.send_control_command("command")
            self.alive_count = 0
    
    @overrides
    def _start(self):
        self.drone.streamon()
    
    @overrides
    def _stop(self):
        self.drone.streamoff()

    @overrides
    def update_observation(self):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def schedule_tasks():
            tasks = set()
            
            while self.running:
                self.keep_alive()

                start_time = time.time()
                frame = self.drone.get_frame_read().frame
                self._image = Image.fromarray(frame)
                # Add a new task to the set
                task = asyncio.create_task(self.yolo_client.detect(self._image))
                tasks.add(task)
                
                # Clean up completed tasks
                tasks = {t for t in tasks if not t.done()}
                with self._image_process_lock:
                    self._image_process_result = self.yolo_client.latest_result
                # Sleep for the interval
                elapsed_time = time.time() - start_time
                await asyncio.sleep(max(0, self.interval - elapsed_time))
        # Run the async function in the event loop
        loop.run_until_complete(schedule_tasks())

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
    
    def move_forward(self, distance: int) -> tuple[bool, bool]:
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

    def move_backward(self, distance: int) -> tuple[bool, bool]:
        print(f"-> Moving backward {distance} cm")
        self.robot.send_command_hover(0, 0, 0, 0)
        while distance > 0:
            self.robot.send_command_hover(0, -self.move_speed_x, 0, 0)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_left(self, distance: int) -> tuple[bool, bool]:
        print(f"-> Moving left {distance} cm")
        self.robot.send_command_hover(0, 0, 0, 0)
        while distance > 0:
            self.robot.send_command_hover(0, 0, -self.move_speed_y, 0)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_right(self, distance: int) -> tuple[bool, bool]:
        print(f"-> Moving right {distance} cm")
        self.robot.send_command_hover(0, 0, 0, 0)
        while distance > 0:
            self.robot.send_command_hover(0, 0, self.move_speed_y, 0)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_up(self, distance: int) -> tuple[bool, bool]:
        print(f"-> Moving up {distance} cm")
        return True, False

    def move_down(self, distance: int) -> tuple[bool, bool]:
        print(f"-> Moving down {distance} cm")
        return True, False

    def turn_ccw(self, degree: int) -> tuple[bool, bool]:
        print(f"-> Turning CCW {degree} degrees")
        self.robot.send_command_hover(0, 0, 0, 0)
        self.robot.send_command_position(0, 0, 0, degree)
        time.sleep(1 + degree / 50.0)
        self.robot.send_command_hover(0, 0, 0, 0)
        # if degree >= 90:
        #     print("-> Turning CCW over 90 degrees")
        #     return True, True
        return True, False

    def turn_cw(self, degree: int) -> tuple[bool, bool]:
        print(f"-> Turning CW {degree} degrees")
        self.robot.send_command_hover(0, 0, 0, 0)
        self.robot.send_command_position(0, 0, 0, -degree)
        time.sleep(1 + degree / 50.0)
        self.robot.send_command_hover(0, 0, 0, 0)
        # if degree >= 90:
        #     print("-> Turning CW over 90 degrees")
        #     return True, True
        return True, False
    
    def move_in_circle(self, cw) -> tuple[bool, bool]:
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
