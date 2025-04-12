from abc import ABC, abstractmethod
from typing import Optional
from numpy import ndarray
import time, threading
from PIL import Image
import asyncio
import re

from .skillset import SkillSet
from .robot_info import RobotInfo
from .yolo_client import ObjectInfo
from .skill_item import SKILL_RET_TYPE
from .utils import evaluate_value

class RobotObservation(ABC):
    def __init__(self, robot_info: RobotInfo, rate: int):
        self.interval: float = 1.0 / rate
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
        self.running = True
        self._start()
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
    
    def update_observation(self):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def schedule_tasks():
            tasks: set[asyncio.Task] = set()
            
            while self.running:
                start_time = time.time()

                # Add a new task to the set
                if self._image is not None:
                    task = asyncio.create_task(self.process_image(self._image))
                    tasks.add(task)
                
                # Clean up completed tasks
                tasks = {t for t in tasks if not t.done()}
                with self._image_process_lock:
                    self._image_process_result = self.fetch_processed_result()
                # Sleep for the interval
                elapsed_time = time.time() - start_time
                await asyncio.sleep(max(0, self.interval - elapsed_time))
        # Run the async function in the event loop
        loop.run_until_complete(schedule_tasks())

    @abstractmethod
    async def process_image(self, image: Image.Image):
        pass
    
    @abstractmethod
    def fetch_processed_result(self) -> tuple[Image.Image, list]:
        pass

class RobotWrapper(ABC):
    def __init__(self, robot_info: RobotInfo, observation: RobotObservation, controller_func: list[callable]):
        self.robot_info = robot_info
        self._observation = observation
        self._user_log = controller_func[0]
        self._probe = controller_func[1]
        common_movement_skill_func = [
            self.move,
            self.rotate,
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
            self.probe
        ]

        self.ll_skillset: SkillSet = SkillSet.get_common_skillset(common_movement_skill_func, common_vision_skill_func, other_skills)
        self.hl_skillset: Optional[SkillSet] = None

    @abstractmethod
    def start(self) -> bool:
        pass

    @abstractmethod
    def stop(self):
        pass

    @property
    def observation(self) -> RobotObservation:
        return self._observation

    # movement skills
    @abstractmethod
    def move(self, dx: float, dy: float) -> tuple[bool, bool]:
        pass

    @abstractmethod
    def rotate(self, deg: float) -> tuple[bool, bool]:
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
        object_name = object_name.strip('\'').lower()

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
        # if `[float]` is in the object_name, use it
        match = re.search(r'\[(-?\d+(\.\d+)?)\]', object_name)
        if match:
            # Extract the number and return it as a float
            extracted_number = float(match.group(1))
            return extracted_number, False
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
    
    def probe(self, query: str) -> tuple[SKILL_RET_TYPE, bool]:
        return evaluate_value(self._probe(query, self.robot_info)), False