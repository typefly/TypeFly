from queue import Queue

class VisionToolWrapper():
    def __init__(self, yolo_results_queue: Queue, image_size: tuple = (960, 720)):
        self.yolo_results_queue = yolo_results_queue
        self.image_size = image_size

    def get_obj_info(self, object_name: str):
        if not self.yolo_results_queue.empty():
            yolo_results = self.yolo_results_queue.queue[0]
            for item in yolo_results:
                if item['label'].replace(' ', '_') == object_name:
                    return item
        return None

    def is_in_sight(self, object_name: str) -> bool:
        print(f"#L is {object_name} in sight?")
        return self.get_obj_info(object_name) is not None
        
    def is_not_in_sight(self, object_name: str) -> bool:
        print(f"#L is {object_name} not in sight?")
        return self.get_obj_info(object_name) is None
    
    def check_location_x(self, object_name: str, compare, val) -> bool:
        print(f"#L check {object_name} location x {compare} {val}")
        info = self.get_obj_info(object_name)
        if info is None:
            return None
        float_val = float(val)
        if compare == '<':
            return (info['x1'] + info['x2']) / 2 / self.image_size[0] < float_val
        elif compare == '>':
            return (info['x1'] + info['x2']) / 2 / self.image_size[0] > float_val
        else:
            return None
    
    def check_location_y(self, object_name: str, compare, val) -> bool:
        print(f"#L check {object_name} location y {compare} {val}")
        info = self.get_obj_info(object_name)
        if info is None:
            return None
        float_val = float(val)
        if compare == '<':
            return (info['y1'] + info['y2']) / 2 / self.image_size[1] < float_val
        elif compare == '>':
            return (info['y1'] + info['y2']) / 2 / self.image_size[1] > float_val
        else:
            return None