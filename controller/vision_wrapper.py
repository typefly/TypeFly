from queue import Queue
from typing import Optional
from llm_wrapper import LLMWrapper

class VisionWrapper():
    def __init__(self, yolo_results_queue: Queue, llm: LLMWrapper):
        self.yolo_results_queue = yolo_results_queue
        self.llm = llm
        with open("./assets/query_skill_prompt.txt", "r") as f:
            self.prompt = f.read()

    def format_results(results):
        formatted_results = []
        for result in results:
            box = result['box']
            formatted_results.append({
                'name': result['name'],
                'loc_x': round((box['x1'] + box['x2']) / 2, 2),
                'loc_y': round((box['y1'] + box['y2']) / 2, 2),
                'size_x': round((box['x2'] - box['x1']), 2),
                'size_y': round((box['y2'] - box['y1']), 2),
            })
        return formatted_results

    def get_obj_list(self) -> [str]:
        if not self.yolo_results_queue.empty():
            yolo_results = self.yolo_results_queue.queue[0]
            return VisionWrapper.format_results(yolo_results)
        return []

    def get_obj_info(self, object_name: str) -> Optional[dict]:
        if not self.yolo_results_queue.empty():
            yolo_results = self.yolo_results_queue.queue[0]
            for item in yolo_results:
                if item['name'] == object_name:
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

    def query(self, question: str) -> bool:
        def parse_value(s):
            # Check for boolean values
            if s.lower() == "true":
                return True
            elif s.lower() == "false":
                return False
            return s
        objects = self.get_obj_list()
        prompt = self.prompt.format(objects=objects, question=question)
        return parse_value(self.llm.query(prompt))