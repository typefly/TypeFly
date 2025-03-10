from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

class RobotType(Enum):
    VIRTUAL = 0
    TELLO = 1
    GO2 = 2

class RobotObservation(ABC):
    @property
    @abstractmethod
    def image(self):
        pass

    @property
    def depth(self):
        return None

    @property
    def orientation(self):
        return None

    @property
    def position(self):
        return None

class RobotWrapper(ABC):
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