from abc import ABC, abstractmethod

class DroneWrapper(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def keep_active(self):
        pass

    @abstractmethod
    def takeoff(self):
        pass

    @abstractmethod
    def land(self):
        pass

    @abstractmethod
    def start_stream(self):
        pass

    @abstractmethod
    def stop_stream(self):
        pass

    @abstractmethod
    def get_image(self):
        pass

    @abstractmethod
    def move_forward(self, distance: int):
        pass
    
    @abstractmethod
    def move_backward(self, distance: int):
        pass
    
    @abstractmethod
    def move_left(self, distance: int):
        pass

    @abstractmethod
    def move_right(self, distance: int):
        pass
    
    @abstractmethod
    def move_up(self, distance: int):
        pass
    
    @abstractmethod
    def move_down(self, distance: int):
        pass

    @abstractmethod
    def turn_counter_clockwise(self, degree: int):
        pass

    @abstractmethod
    def turn_clockwise(self, degree: int):
        pass
