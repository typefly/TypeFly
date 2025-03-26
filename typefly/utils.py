import datetime
import numpy as np
from numpy import ndarray
from .skill_item import SKILL_RET_TYPE

def print_t(*args, **kwargs):
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    print(f"[{current_time}]", *args, **kwargs)

def input_t(literal):
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    return input(f"[{current_time}] {literal}")

def evaluate_value(s: str) -> SKILL_RET_TYPE:
    if s.isdigit():
        return int(s)
    elif s.replace('.', '', 1).isdigit():
        return float(s)
    elif s == 'True':
        return True
    elif s == 'False':
        return False
    elif s == 'None' or len(s) == 0:
        return None
    else:
        if not (s.startswith("'") and s.endswith("'")):
            return f"'{s}'"
        return s
    
def quaternion_to_rpy(qx, qy, qz, qw) -> ndarray:
    """
    Convert quaternion (qx, qy, qz, qw) to roll, pitch, and yaw (RPY) angles in radians.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])