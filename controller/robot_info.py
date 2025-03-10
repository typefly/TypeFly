import json
from typing import Optional, Tuple
import numpy as np
import time

class RobotInfo:
    def __init__(self, robot_id: str, robot_type: str):
        self.robot_id = robot_id
        self.robot_type = robot_type
    
    def __hash__(self):
        return hash((self.robot_id, self.robot_type))

    def __eq__(self, other):
        if not isinstance(other, RobotInfo):
            return False
        return self.robot_id == other.robot_id and self.robot_type == other.robot_type
    
    def to_dict(self):
        return {
            "robot_id": self.robot_id,
            "robot_type": self.robot_type
        }

    def to_json(self):
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(data["robot_id"], data["robot_type"])
    
    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.from_dict(data)

    