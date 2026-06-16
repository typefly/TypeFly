from abc import ABC, abstractmethod
from typing import Any, Optional
from numpy import ndarray
import time, threading
from PIL import Image
import asyncio
import re
import numpy as np

from .skillset import SkillSet
from .robot_info import RobotInfo
from .yolo_client import ObjectInfo
from .skill_item import PROBE_RET_TYPE
from .utils import evaluate_value, print_t, sanitize_prompt_text

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
        self._orientation: ndarray = np.zeros(3)
        self._position: ndarray = np.zeros(3)

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
    def __init__(self, robot_info: RobotInfo, obs: RobotObservation):
        self.robot_info = robot_info
        self.obs = obs
        self._policy = None  # PlanPolicy set by the controller during plan execution
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

    # --- plan-policy hooks (set by the controller around plan execution) ---
    def set_policy(self, policy) -> None:
        """Install the active PlanPolicy so skills clamp args / honor cancel."""
        self._policy = policy

    def clear_policy(self) -> None:
        self._policy = None

    def _policy_guard(self, kind: str, value=None):
        """Guard a skill action: clamp args, count work, honor cancellation.

        Returns the (possibly clamped) value. A no-op when no policy is active
        (e.g. high-level skills used outside a plan, or in tests)."""
        if self._policy is None:
            return value
        return self._policy.checkpoint(kind, value)

    def _policy_poll(self) -> None:
        """Cancel/deadline check only, for tight internal wait loops."""
        if self._policy is not None:
            self._policy.poll()

    # --- safety stops (overridden per platform) ---
    def stop_motion(self) -> None:
        """Halt any in-progress movement but keep the robot powered/live.

        Default no-op; platforms override to issue a zero-velocity command.
        Called by the controller when a plan errors or is cancelled."""
        print_t("[Robot] stop_motion (no-op default)")

    def emergency_shutdown(self) -> None:
        """Bring the robot to a fully safe resting state (land/rest).

        Default no-op; platforms override. Reserved for fatal paths."""
        print_t("[Robot] emergency_shutdown (no-op default)")

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
        dist = self._policy_guard("move", dist)
        print_t(f"-> Move forward by {dist} m")
        self._move(dist, 0)

    def move_backward(self, dist: float):
        dist = self._policy_guard("move", dist)
        print_t(f"-> Move backward by {dist} m")
        self._move(-dist, 0)

    def move_left(self, dist: float):
        dist = self._policy_guard("move", dist)
        print_t(f"-> Move left by {dist} m")
        self._move(0, dist)

    def move_right(self, dist: float):
        dist = self._policy_guard("move", dist)
        print_t(f"-> Move right by {dist} m")
        self._move(0, -dist)

    def rotate_left(self, deg: float):
        deg = self._policy_guard("rotate", deg)
        print_t(f"-> Rotate left by {deg} degrees")
        self._rotate(deg)

    def rotate_right(self, deg: float):
        deg = self._policy_guard("rotate", deg)
        print_t(f"-> Rotate right by {deg} degrees")
        self._rotate(-deg)

    # vision skills
    def get_obj_list(self) -> list[ObjectInfo]:
        """
        Returns the list of detected objects.
        You should override this method if your robot has a different way to get the object list.
        """
        return self.obs.image_process_result.get("yolo", [])
    
    def get_obj_list_str(self) -> str:
        """Returns a formatted string of detected objects.

        Each object string is flattened to a single line (sanitize_prompt_text)
        so a crafted/garbled YOLO label cannot inject extra lines into the
        planning prompt. Defense-in-depth only; enforcement is in plan_execution.
        """
        object_list = self.get_obj_list()
        return "\n".join(
            sanitize_prompt_text(str(obj), max_len=200).replace("'", "")
            for obj in object_list
        )

    def get_obj_info(self, object_name: str) -> ObjectInfo:
        self._policy_guard("vision")
        # A generated plan can chain skills so a bool/number reaches here, e.g.
        # goto(scan('x')) or `obj = scan_description(...); goto(obj)` when probe
        # answers a yes/no query with True. Treat any non-string as "not in
        # sight" instead of crashing on .strip() ('bool' has no attribute strip).
        if not isinstance(object_name, str):
            print_t(f"[!] get_obj_info expected an object-name string, got "
                    f"{type(object_name).__name__} ({object_name!r}); treating as not in sight")
            return None
        object_name = object_name.strip('\'').lower()

        # try to get the object info for 10 times
        for _ in range(10):
            self._policy_poll()  # stay cancellable during the ~2s retry window
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
        # if `[float]` is in the object_name, use it (guard re.search against a
        # non-string arg chained in from a bool/number-returning skill)
        if isinstance(object_name, str):
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
        # Not in sight: _get_object_attribute already returned a descriptive
        # string -- pass it through rather than amplifying it by 100x.
        if isinstance(depth_info, str):
            return depth_info
        # In sight but no depth: the YOLO webcam/petoi pipeline emits no depth
        # key, so ObjectInfo.depth is None. Don't compute None * 100 (TypeError).
        if depth_info is None:
            return f'dist: {object_name} has no depth information'
        return depth_info * 100
    
    # other skills
    def take_picture(self):
        self._policy_guard("action")  # count + honor cancellation
        self.controller_func[0](self.obs.image)

    def log(self, message: str):
        self._policy_guard("action")  # count + honor cancellation (prevents log flooding)
        self.controller_func[0](message)

    def delay(self, sec: float):
        sec = self._policy_guard("delay", sec)
        time.sleep(sec)

    def re_plan(self):
        self._policy_guard("action")  # count + honor cancellation
        return None

    def probe(self, query: str) -> PROBE_RET_TYPE:
        self._policy_guard("probe")
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
            # Contract (see skill registration): return the *object name* if
            # found. probe() may answer a yes/no query with True or a count with
            # a number; those are not usable goto() targets, so keep scanning
            # rather than returning a non-string that would crash goto().
            if isinstance(ret, str) and ret:
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