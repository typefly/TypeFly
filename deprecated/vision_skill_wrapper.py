from typing import Union, Optional
import numpy as np
import time, math
import cv2
from filterpy.kalman import KalmanFilter
from abc import ABC, abstractmethod

from ..typefly.skill_item import SkillArg
from ..typefly.skillset import LowLevelSkillItem, SkillSet
from ..typefly.robot_wrapper import RobotObservation
from ..typefly.yolo_client import ObjectInfo

def iou(boxA, boxB):
    # Calculate the intersection over union (IoU) of two bounding boxes
    xA = max(boxA['x1'], boxB['x1'])
    yA = max(boxA['y1'], boxB['y1'])
    xB = min(boxA['x2'], boxB['x2'])
    yB = min(boxA['y2'], boxB['y2'])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA['x2'] - boxA['x1']) * (boxA['y2'] - boxA['y1'])
    boxBArea = (boxB['x2'] - boxB['x1']) * (boxB['y2'] - boxB['y1'])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def euclidean_distance(boxA, boxB):
    centerA = ((boxA['x1'] + boxA['x2']) / 2, (boxA['y1'] + boxA['y2']) / 2)
    centerB = ((boxB['x1'] + boxB['x2']) / 2, (boxB['y1'] + boxB['y2']) / 2)
    return math.sqrt((centerA[0] - centerB[0])**2 + (centerA[1] - centerB[1])**2)

class ObjectTracker:
    def __init__(self, name, x, y, w, h) -> None:
        self.name = name
        self.kf_pos = self.init_filter()
        self.kf_siz = self.init_filter()
        self.timestamp = 0
        self.size = None
        self.update(x, y, w, h)

    def update(self, x, y, w, h):
        self.kf_pos.update((x, y))
        self.kf_siz.update((w, h))
        self.timestamp = time.time()

    def predict(self) -> Optional[ObjectInfo]:
        # if no update in 2 seconds, return None
        if time.time() - self.timestamp > 0.5:
            return None
        self.kf_pos.predict()
        self.kf_siz.predict()
        return ObjectInfo(self.name, self.kf_pos.x[0][0], self.kf_pos.x[1][0], self.kf_siz.x[0][0], self.kf_siz.x[1][0])

    def init_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state dimensions (x, y, vx, vy), 2 measurement dimensions (x, y)
        kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                        [0, 1, 0, 0]])
        kf.R *= 1  # Measurement uncertainty
        kf.P *= 1000  # Initial uncertainty
        kf.Q *= 0.01  # Process uncertainty
        return kf

class VisionSkillWrapper():
    def __init__(self, observation: RobotObservation):
        self.observation = observation
        self.object_trackers: dict[str, ObjectTracker] = {}
        self.aruco_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
            cv2.aruco.DetectorParameters()
        )

    # def _update(self):
    #     objs = self.observation.image_process_result[1] if self.observation.image_process_result else []
    #     updated_trackers = {}

    #     for obj in objs:
    #         name = obj['name']
    #         box = obj['box']
    #         x = (box['x1'] + box['x2']) / 2
    #         y = (box['y1'] + box['y2']) / 2
    #         w = box['x2'] - box['x1']
    #         h = box['y2'] - box['y1']

    #         best_match_key = None
    #         best_match_distance = float('inf')
            
    #         # Find the best matching tracker
    #         for key, tracker in self.object_trackers.items():
    #             if tracker.name == name:
    #                 existing_box = {
    #                     'x1': tracker.kf_pos.x[0][0] - tracker.kf_siz.x[0][0] / 2,
    #                     'y1': tracker.kf_pos.x[1][0] - tracker.kf_siz.x[1][0] / 2,
    #                     'x2': tracker.kf_pos.x[0][0] + tracker.kf_siz.x[0][0] / 2,
    #                     'y2': tracker.kf_pos.x[1][0] + tracker.kf_siz.x[1][0] / 2,
    #                 }
    #                 distance = euclidean_distance(existing_box, box)
    #                 if distance < best_match_distance:
    #                     best_match_distance = distance
    #                     best_match_key = key

    #         # Update the best matching tracker or create a new one
    #         if best_match_key is not None and best_match_distance < 50:  # Threshold can be adjusted
    #             self.object_trackers[best_match_key].update(x, y, w, h)
    #             updated_trackers[best_match_key] = self.object_trackers[best_match_key]
    #         else:
    #             new_key = f"{name}_{len(self.object_trackers)}"  # Create a unique key
    #             updated_trackers[new_key] = ObjectTracker(name, x, y, w, h)

    #     # Replace the old trackers with the updated ones
    #     self.object_trackers = updated_trackers

    #     # Create the list of current objects
    #     self.object_list = []
    #     to_delete = []
    #     for key, tracker in self.object_trackers.items():
    #         obj = tracker.predict()
    #         if obj is not None:
    #             self.object_list.append(obj)
    #         else:
    #             to_delete.append(key)
        
    #     # Remove trackers that should be deleted
    #     for key in to_delete:
    #         del self.object_trackers[key]


    # def update(self):
    #     if self.shared_frame.timestamp == self.last_update:
    #         return
    #     self.last_update = self.shared_frame.timestamp
    #     objs = self.shared_frame.get_yolo_result()['result'] + self.shared_frame.get_yolo_result()['result_custom']
    #     for obj in objs:
    #         name = obj['name']
    #         box = obj['box']
    #         x = (box['x1'] + box['x2']) / 2
    #         y = (box['y1'] + box['y2']) / 2
    #         w = box['x2'] - box['x1']
    #         h = box['y2'] - box['y1']
    #         if name not in self.object_trackers:
    #             self.object_trackers[name] = ObjectTracker(name, x, y, w, h)
    #         else:
    #             self.object_trackers[name].update(x, y, w, h)
        
    #     self.object_list = []
    #     to_delete = []
    #     for name, tracker in self.object_trackers.items():
    #         obj = tracker.predict()
    #         if obj is not None:
    #             self.object_list.append(obj)
    #         else:
    #             to_delete.append(name)
    #     for name in to_delete:
    #         del self.object_trackers[name]

    def get_obj_list(self) -> list[ObjectInfo]:
        """Returns a formatted string of detected objects."""
        return self.observation.image_process_result[1] if self.observation.image_process_result else []
    
    def get_obj_list_str(self) -> str:
        """Returns a formatted string of detected objects."""
        object_list = self.get_obj_list()
        return str([str(obj) for obj in object_list]).replace("'", "")

    def get_obj_info(self, object_name: str) -> ObjectInfo:
        # try to get the object info for 10 times
        for _ in range(10):
            object_list = self.get_obj_list()
            for obj in object_list:
                if obj.name.startswith(object_name):
                    return obj
            time.sleep(0.2)
        return None

    def is_visible(self, object_name: str) -> tuple[bool, bool]:
        return self.get_obj_info(object_name) is not None, False

    def _get_object_attribute(self, object_name: str, attr: str) -> tuple[Union[float, str], bool]:
        """Helper function to retrieve an object's attribute."""
        info = self.get_obj_info(object_name)
        if info is None:
            return f'{attr}: {object_name} is not in sight', True
        return getattr(info, attr), False
    
    def object_x(self, object_name: str) -> tuple[Union[float, str], bool]:
        return self._get_object_attribute(object_name, 'x')
    
    def object_y(self, object_name: str) -> tuple[Union[float, str], bool]:
        return self._get_object_attribute(object_name, 'y')
    
    def object_width(self, object_name: str) -> tuple[Union[float, str], bool]:
        return self._get_object_attribute(object_name, 'w')
    
    def object_height(self, object_name: str) -> tuple[Union[float, str], bool]:
        return self._get_object_attribute(object_name, 'h')
    
    # def _object_distance(self, object_name: str) -> Tuple[Union[int, str], bool]:
    #     info = self.get_obj_info(object_name)
    #     if info is None:
    #         return f'object_distance: {object_name} not in sight', True
    #     mid_point = (info.x, info.y)
    #     FOV_X = 0.42
    #     FOV_Y = 0.55
    #     if mid_point[0] < 0.5 - FOV_X / 2 or mid_point[0] > 0.5 + FOV_X / 2 \
    #     or mid_point[1] < 0.5 - FOV_Y / 2 or mid_point[1] > 0.5 + FOV_Y / 2:
    #         return 30, False
    #     depth = self.shared_frame.get_depth().data
    #     start_x = 0.5 - FOV_X / 2
    #     start_y = 0.5 - FOV_Y / 2
    #     index_x = (mid_point[0] - start_x) / FOV_X * (depth.shape[1] - 1)
    #     index_y = (mid_point[1] - start_y) / FOV_Y * (depth.shape[0] - 1)
    #     return int(depth[int(index_y), int(index_x)] / 10), False