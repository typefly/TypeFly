from queue import Queue
from typing import Optional

class VisionWrapper():
    def __init__(self, yolo_results_queue: Queue, image_size: tuple = (960, 720)):
        self.yolo_results_queue = yolo_results_queue
        self.image_size = image_size

    def get_obj_info(self, object_name: str) -> Optional[dict]:
        if not self.yolo_results_queue.empty():
            yolo_results = self.yolo_results_queue.queue[0]
            for item in yolo_results:
                if item['label'].replace(' ', '_') == object_name:
                    return item
        return None

    def is_in_sight(self, object_name: str) -> bool:
        return self.get_obj_info(object_name) is not None
        
    def is_not_in_sight(self, object_name: str) -> bool:
        return self.get_obj_info(object_name) is None
    
    def check_location_x(self, object_name: str, compare: str, val: float) -> Optional[bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return None
        if compare == '<':
            return (info['x1'] + info['x2']) / 2 / self.image_size[0] < val
        elif compare == '>':
            return (info['x1'] + info['x2']) / 2 / self.image_size[0] > val
        else:
            return None
    
    def check_location_y(self, object_name: str, compare: str, val: float) -> Optional[bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return None
        if compare == '<':
            return (info['y1'] + info['y2']) / 2 / self.image_size[1] < val
        elif compare == '>':
            return (info['y1'] + info['y2']) / 2 / self.image_size[1] > val
        else:
            return None