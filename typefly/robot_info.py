import json
from typing import Optional

class RobotInfo:
    def __init__(self, robot_id: str, robot_type: str, extra: Optional[dict] = None):
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.extra = extra
    
    def __hash__(self) -> int:
        return hash(self.robot_id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, RobotInfo):
            return False
        return self.robot_id == other.robot_id
    
    def to_dict(self) -> dict:
        info = {
            "robot_id": self.robot_id,
            "robot_type": self.robot_type
        }
        if self.extra:
            info["extra"] = self.extra
        return info

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    def get_robot_type(self, true_type: bool=True) -> str:
        if not true_type and self.robot_type == "virtual":
            return "tello"
        return self.robot_type
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RobotInfo':
        return cls(data["robot_id"], data["robot_type"], data.get("extra"))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RobotInfo':
        data = json.loads(json_str)
        return cls.from_dict(data)