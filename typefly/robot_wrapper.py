from abc import ABC, abstractmethod
from typing import Any, Optional
from numpy import ndarray
import time, threading
from PIL import Image
import asyncio
import re

from .skillset import SkillSet
from .robot_info import RobotInfo
from .yolo_client import ObjectInfo
from .skill_item import PROBE_RET_TYPE
from .utils import evaluate_value, print_t

class RobotObservation(ABC):
    """
    Get the latest observation from the robot. This information will be used by the LLM to plan and
    execute the program.

    The subclass should implement the following methods:
    - _start: Start the observation thread
    - _stop: Stop the observation thread
    - process_image: Process the image from the robot
    - fetch_processed_result: Fetch the processed result from the robot

    The subclass should also implement the following properties, the image is required:
    - image: The image from the robot (required)
    - depth: The depth from the robot (optional)
    - orientation: The orientation from the robot (optional)
    - position: The position from the robot (optional)
    """
    def __init__(self, robot_info: RobotInfo, rate: int):
        self.interval: float = 1.0 / rate
        self.robot_info = robot_info

        self._image: Optional[Image.Image] = None
        self._depth: Optional[ndarray] = None
        self._orientation: Optional[ndarray] = None
        self._position: Optional[ndarray] = None

        self._image_process_lock = threading.Lock()
        self._image_process_result: tuple[Image.Image, list[ObjectInfo]] = (Image.new("RGB", (640, 480)), [])

        self.running: bool = False
        self.processing_thread = threading.Thread(target=self.update_observation, daemon=True)

    def start(self):
        self.running = True
        self._start()
        self.processing_thread.start()

    def stop(self):
        self.running = False
        self.processing_thread.join()
        self._stop()

    @abstractmethod
    def _start(self):
        """
        This method should be implemented by the subclass to start the observation thread
        """
        pass

    @abstractmethod
    def _stop(self):
        """
        This method should be implemented by the subclass to stop the observation thread
        """
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
        """
        This method should be implemented by the subclass to process the image asynchronously.
        The processing_thread will call this method periodically to process the image.
        """
        pass
    
    @abstractmethod
    def fetch_processed_result(self) -> dict[str, Any]:
        """
        This method should be implemented by the subclass to fetch the processed result
        The return value should be a dictionary, the key is the name of the processed result,
        the value is the processed result. For example, if you have a YOLO client, you can return:
        {
            "yolo": object_list
        }
        """
        pass

class RobotWrapper(ABC):
    controller_func: list[callable] = []
    def __init__(self, robot_info: RobotInfo, observation: RobotObservation):
        self.robot_info = robot_info
        self.observation = observation
        common_movement_skill_func = [
            (self.move_forward, "Move forward by a dist (m)"),
            (self.move_backward, "Move backward by a dist (m)"),
            (self.move_left, "Move left by a dist (m)"),
            (self.move_right, "Move right by a dist (m)"),
            (self.rotate_left, "Rotate left by a deg (deg)"),
            (self.rotate_right, "Rotate right by a deg (deg)"),
        ]

        common_vision_skill_func = [
            (self.is_visible, "Check if object is visible"),
            (self.object_x, "Get object's x position (0-1)"),
            (self.object_y, "Get object's y position (0-1)"),
            (self.object_width, "Get object's width (0-1)"),
            (self.object_height, "Get object's height (0-1)"),
            (self.object_dist, "Get object's dist (m)"),
        ]

        other_skills = [
            (self.take_picture, "Take a picture"),
            (self.log, "Print text to user"),
            (self.delay, "Wait for seconds"),
            (self.re_plan, "Trigger replanning"),
            (self.probe, "Query LLM for reasoning"),
        ]

        high_level_skills = [
            (self.scan, "Scan for a specific object"),
            (self.scan_description, "Scan for an abstract object, return the object name if found"),
            (self.orienting, "Orient to a specific object"),
            (self.goto, "Go to a specific object in the view"),
        ]

        self.skillset: SkillSet = SkillSet.get_common_skillset(common_movement_skill_func + common_vision_skill_func + other_skills + high_level_skills)

    @staticmethod
    def set_controller_func(controller_func: list[callable]):
        RobotWrapper.controller_func = controller_func

    @abstractmethod
    def start(self) -> bool:
        """
        This method should be implemented by the subclass to start the robot
        """
        pass

    @abstractmethod
    def stop(self) -> bool:
        """
        This method should be implemented by the subclass to stop the robot
        """
        pass

    @abstractmethod
    def _move(self, dx: float, dy: float):
        """
        This method should be implemented by the subclass to move the robot
        """
        pass

    @abstractmethod
    def _rotate(self, deg: float):
        """
        This method should be implemented by the subclass to rotate the robot
        """
        pass

    # movement skills
    def move_forward(self, dist: float):
        print_t(f"-> Move forward by {dist} m")
        self._move(dist, 0)
    
    def move_backward(self, dist: float):
        print_t(f"-> Move backward by {dist} m")
        self._move(-dist, 0)
    
    def move_left(self, dist: float):
        print_t(f"-> Move left by {dist} m")
        self._move(0, dist)
    
    def move_right(self, dist: float):
        print_t(f"-> Move right by {dist} m")
        self._move(0, -dist)
    
    def rotate_left(self, deg: float):
        print_t(f"-> Rotate left by {deg} degrees")
        self._rotate(deg)
    
    def rotate_right(self, deg: float):
        print_t(f"-> Rotate right by {deg} degrees")
        self._rotate(-deg)

    # vision skills
    def get_obj_list(self) -> list[ObjectInfo]:
        """
        Returns the list of detected objects.
        You should override this method if your robot has a different way to get the object list.
        """
        return self.observation.image_process_result.get("yolo", [])
    
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

    def is_visible(self, object_name: str) -> bool:
        return self.get_obj_info(object_name) is not None

    def _get_object_attribute(self, object_name: str, attr: str) -> tuple[float | str]:
        """Helper function to retrieve an object's attribute."""
        info = self.get_obj_info(object_name)
        if info is None:
            return f'{attr}: {object_name} is not in sight'
        return getattr(info, attr)
    
    def object_x(self, object_name: str) -> float:
        # if `[float]` is in the object_name, use it
        match = re.search(r'\[(-?\d+(\.\d+)?)\]', object_name)
        if match:
            # Extract the number and return it as a float
            extracted_number = float(match.group(1))
            if extracted_number is not None:
                return extracted_number
            else:
                raise ValueError(f'{object_name} is not a valid number')
        return self._get_object_attribute(object_name, 'x')
    
    def object_y(self, object_name: str) -> float:
        return self._get_object_attribute(object_name, 'y')
    
    def object_width(self, object_name: str) -> float:
        return self._get_object_attribute(object_name, 'w')
    
    def object_height(self, object_name: str) -> float:
        return self._get_object_attribute(object_name, 'h')
    
    def object_dist(self, object_name: str) -> float:
        depth_info = self._get_object_attribute(object_name, 'depth')
        return depth_info * 100
    
    # other skills
    def take_picture(self):
        self.controller_func[0](self.observation.image)
    
    def log(self, message: str):
        self.controller_func[0](message)

    def delay(self, sec: float):
        time.sleep(sec)
    
    def re_plan(self):
        return None

    def probe(self, query: str) -> PROBE_RET_TYPE:
        return evaluate_value(self.controller_func[1](query, self.robot_info))

    # high-level skills
    def scan(self, object_name: str) -> bool:
        print(f"-> Scan for {object_name}")
        for _ in range(8):
            if self.is_visible(object_name):
                return True
            self.rotate_left(45)
        return False

    def scan_description(self, description: str) -> bool:
        print(f"-> Scan for {description}")
        for _ in range(8):
            ret = self.probe(description)
            if ret != False:
                return ret
            self.rotate_left(45)
        return False

    def orienting(self, object_name: str) -> bool:
        print(f"-> Orient to {object_name}")
        if not self.is_visible(object_name):
            return False
        self.rotate_left(int((0.5 - self.object_x(object_name)) * 80))
        return True

    def goto(self, object_name: str) -> bool:
        print(f"-> Go to {object_name}")
        if not self.orienting(object_name):
            return False
        self.move_forward(1.0)
        return True