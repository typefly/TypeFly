import queue
from tello_wrapper import TelloWrapper

class Toolset():
    def __init__(self, yolo_results_queue: queue.Queue, drone_wrapper: TelloWrapper = None, image_size: tuple = (960, 720)):
        self.drone_wrapper = drone_wrapper
        self.yolo_results_queue = yolo_results_queue
        self.image_size = image_size

    def move_forward(self, distance):
        print(f"#L Moving forward {distance}cm")
        if self.drone_wrapper is not None:
            self.drone_wrapper.drone.move_forward(int(distance))
            return True
        return False

    def move_backward(self, distance):
        print(f"#L Moving backward {distance}cm")
        if self.drone_wrapper is not None:
            self.drone_wrapper.drone.move_backward(int(distance))
            return True
        return False

    def move_left(self, distance):
        print(f"#L Moving left {distance}cm")
        if self.drone_wrapper is not None:
            self.drone_wrapper.drone.move_left(int(distance))
            return True
        return False

    def move_right(self, distance):
        print(f"#L Moving right {distance}cm")
        if self.drone_wrapper is not None:
            self.drone_wrapper.drone.move_right(int(distance))
            return True
        return False

    def move_up(self, distance):
        print(f"#L Moving up {distance}cm")
        if self.drone_wrapper is not None:
            self.drone_wrapper.drone.move_up(int(distance))
            return True
        return False

    def move_down(self, distance):
        print(f"#L Moving down {distance}cm")
        if self.drone_wrapper is not None:
            self.drone_wrapper.drone.move_down(int(distance))
            return True
        return False

    def turn_left(self, angle):
        print(f"#L Turning left {angle} degrees")
        if self.drone_wrapper is not None:
            self.drone_wrapper.drone.rotate_counter_clockwise(int(angle))
            return True
        return False

    def turn_right(self, angle):
        print(f"#L Turning right {angle} degrees")
        if self.drone_wrapper is not None:
            self.drone_wrapper.drone.rotate_clockwise(int(angle))
            return True
        return False

    def get_obj_info(self, obj):
        print(f"#L get info for {obj}")
        if not self.yolo_results_queue.empty():
            yolo_results = self.yolo_results_queue.queue[0]
            for item in yolo_results:
                if item['label'].replace(' ', '_') == obj:
                    return item
        return None

    def is_in_sight(self, obj) -> bool:
        print(f"#L is {obj} in sight?")
        return self.get_obj_info(obj) is not None
        
    def is_not_in_sight(self, obj) -> bool:
        print(f"#L is {obj} not in sight?")
        return self.get_obj_info(obj) is None
    
    def check_location_x(self, obj, compare, val) -> bool:
        print(f"#L check {obj} location x {compare} {val}")
        info = self.get_obj_info(obj)
        if info is None:
            return None
        float_val = float(val)
        if compare == '<':
            return (info['x1'] + info['x2']) / 2 / self.image_size[0] < float_val
        elif compare == '>':
            return (info['x1'] + info['x2']) / 2 / self.image_size[0] > float_val
        else:
            return None
    
    def check_location_y(self, obj, compare, val) -> bool:
        print(f"#L check {obj} location y {compare} {val}")
        info = self.get_obj_info(obj)
        if info is None:
            return None
        float_val = float(val)
        if compare == '<':
            return (info['y1'] + info['y2']) / 2 / self.image_size[1] < float_val
        elif compare == '>':
            return (info['y1'] + info['y2']) / 2 / self.image_size[1] > float_val
        else:
            return None
