from queue import Queue
from typing import Optional
from .yolo_client import SharedYoloResults

class VisionSkillWrapper():
    def __init__(self, shared_yolo_results: SharedYoloResults):
        self.shared_yolo_results = shared_yolo_results

    def format_results(results):
        (image, json_data) = results
        formatted_results = []
        for item in json_data['result']:
            box = item['box']
            formatted_results.append(str({
                'name': item['name'],
                'loc_x': round((box['x1'] + box['x2']) / 2, 2),
                'loc_y': round((box['y1'] + box['y2']) / 2, 2),
                'size_x': round((box['x2'] - box['x1']), 2),
                'size_y': round((box['y2'] - box['y1']), 2)
            }).replace("'", ''))
        return formatted_results

    def get_obj_list(self) -> [str]:
        return VisionSkillWrapper.format_results(self.shared_yolo_results.get())

    def get_obj_info(self, object_name: str) -> Optional[dict]:
        (_, yolo_results) = self.shared_yolo_results.get()
        for item in yolo_results['result']:
            # change this to start_with
            if item['name'].startswith(object_name):
                return item
        return None

    def is_in_sight(self, object_name: str) -> bool:
        return self.get_obj_info(object_name) is not None
        
    def obj_loc_x(self, object_name: str) -> Optional[float]:
        info = self.get_obj_info(object_name)
        if info is None:
            return None
        box = info['box']
        return (box['x1'] + box['x2']) / 2
    
    def obj_loc_y(self, object_name: str) -> Optional[float]:
        info = self.get_obj_info(object_name)
        if info is None:
            return None
        box = info['box']
        return (box['y1'] + box['y2']) / 2