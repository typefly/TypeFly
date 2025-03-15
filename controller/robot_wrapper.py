from abc import ABC, abstractmethod
from typing import Optional
from numpy import ndarray
import time, threading
from PIL import Image

from .skillset import SkillSet
from .robot_info import RobotInfo
from .yolo_client import ObjectInfo

class RobotObservation(ABC):
    def __init__(self, robot_info: RobotInfo):
        self.robot_info = robot_info

        self._image: Optional[Image.Image] = None
        self._depth: Optional[ndarray] = None
        self._orientation: Optional[ndarray] = None
        self._position: Optional[ndarray] = None

        self._image_process_lock = threading.Lock()
        self._image_process_result: tuple[Image.Image, list[ObjectInfo]] = (None, [])

        self.running: bool = False
        self.thread = threading.Thread(target=self.update_observation, daemon=True)

    def start(self):
        self._start()
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self._stop()

    @abstractmethod
    def _start(self):
        pass

    @abstractmethod
    def _stop(self):
        pass

    @property
    def image(self) -> Optional[Image.Image]:
        return self._image

    @property
    def depth(self) -> Optional[ndarray]:
        return self._depth

    @property
    def orientation(self) -> Optional[ndarray]:
        return self._orientation

    @property
    def position(self) -> Optional[ndarray]:
        return self._position
    
    @property
    def image_process_result(self) -> tuple[Image.Image, list[ObjectInfo]]:
        with self._image_process_lock:
            return self._image_process_result
    
    @abstractmethod
    def update_observation(self):
        pass

class RobotWrapper(ABC):
    def __init__(self, robot_info: RobotInfo, observation: RobotObservation, controller_func: list[callable]):
        self.robot_info = robot_info
        self._observation = observation
        self._user_log = controller_func[0]
        common_movement_skill_func = [
            self.move_forward,
            self.move_backward,
            self.move_left,
            self.move_right,
            self.turn_cw,
            self.turn_ccw
        ]

        common_vision_skill_func = [
            self.is_visible,
            self.object_x,
            self.object_y,
            self.object_width,
            self.object_height,
            self.take_picture
        ]

        other_skills = [
            self.log,
            self.delay,
            self.re_plan,
            controller_func[1]
        ]

        self.ll_skillset: SkillSet = SkillSet.get_common_skillset(common_movement_skill_func, common_vision_skill_func, other_skills)
        self.hl_skillset: Optional[SkillSet] = None

    @abstractmethod
    def start(self) -> bool:
        pass

    @abstractmethod
    def stop(self):
        pass

    # some robots need to be kept active
    def keep_alive(self):
        return

    @property
    def observation(self) -> RobotObservation:
        return self._observation

    # movement skills
    @abstractmethod
    def move_forward(self, dist: int) -> tuple[bool, bool]:
        pass
    
    @abstractmethod
    def move_backward(self, dist: int) -> tuple[bool, bool]:
        pass
    
    @abstractmethod
    def move_left(self, dist: int) -> tuple[bool, bool]:
        pass

    @abstractmethod
    def move_right(self, dist: int) -> tuple[bool, bool]:
        pass

    @abstractmethod
    def turn_ccw(self, deg: int) -> tuple[bool, bool]:
        pass

    @abstractmethod
    def turn_cw(self, deg: int) -> tuple[bool, bool]:
        pass

    # vision skills
    def get_obj_list(self) -> list[ObjectInfo]:
        """Returns a formatted string of detected objects."""
        return self._observation.image_process_result[1] if self._observation.image_process_result else []
    
    def get_obj_list_str(self) -> str:
        """Returns a formatted string of detected objects."""
        object_list = self.get_obj_list()
        return "\n".join([str(obj) for obj in object_list]).replace("'", "")

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

    def _get_object_attribute(self, object_name: str, attr: str) -> tuple[float | str, bool]:
        """Helper function to retrieve an object's attribute."""
        info = self.get_obj_info(object_name)
        if info is None:
            return f'{attr}: {object_name} is not in sight', True
        return getattr(info, attr), False
    
    def object_x(self, object_name: str) -> tuple[float | str, bool]:
        return self._get_object_attribute(object_name, 'x')
    
    def object_y(self, object_name: str) -> tuple[float | str, bool]:
        return self._get_object_attribute(object_name, 'y')
    
    def object_width(self, object_name: str) -> tuple[float | str, bool]:
        return self._get_object_attribute(object_name, 'w')
    
    def object_height(self, object_name: str) -> tuple[float | str, bool]:
        return self._get_object_attribute(object_name, 'h')
    
    def take_picture(self) -> tuple[bool, bool]:
        return self._user_log(self.observation.image)
    
    def log(self, message: str) -> tuple[None, bool]:
        return self._user_log(message)

    def delay(self, sec: float) -> tuple[None, bool]:
        time.sleep(sec)
        return None, False
    
    def re_plan(self) -> tuple[None, bool]:
        return None, True