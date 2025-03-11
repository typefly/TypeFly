from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
from numpy import ndarray
import time, threading
from PIL import Image
from ..robot_info import RobotInfo

class RobotObservation(ABC):
    def __init__(self, robot_info: RobotInfo):
        self.robot_info = robot_info

        self._image: Optional[Image.Image] = None
        self._depth: Optional[ndarray] = None
        self._orientation: Optional[ndarray] = None
        self._position: Optional[ndarray] = None

        self._image_process_result: Optional[tuple[Image.Image, list]] = None

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
    def image_process_result(self) -> Optional[tuple[Image.Image, list]]:
        return self._image_process_result
    
    @abstractmethod
    def update_observation(self):
        pass

class RobotWrapper(ABC):
    robot_info = None
    low_level_skillset = None
    high_level_skillset = None

    @abstractmethod
    def start(self) -> bool:
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod # some robots need to be kept active
    def keep_alive(self):
        pass

    @abstractmethod
    def get_observation(self) -> Optional[RobotObservation]:
        pass

    @abstractmethod
    def move_forward(self, distance: int) -> bool:
        pass
    
    @abstractmethod
    def move_backward(self, distance: int) -> bool:
        pass
    
    @abstractmethod
    def move_left(self, distance: int) -> bool:
        pass

    @abstractmethod
    def move_right(self, distance: int) -> bool:
        pass
    
    @abstractmethod
    def move_up(self, distance: int) -> bool:
        pass
    
    @abstractmethod
    def move_down(self, distance: int) -> bool:
        pass

    @abstractmethod
    def turn_ccw(self, degree: int) -> bool:
        pass

    @abstractmethod
    def turn_cw(self, degree: int) -> bool:
        pass