import datetime, cv2
import re
import numpy as np
from numpy import ndarray
import os
from dotenv import load_dotenv
from .skill_item import PROBE_RET_TYPE

# Load .env once, early. utils is imported transitively almost everywhere (and
# before OPENAI_API_KEY / EDGE_SERVICE_* are read), so this is the central load
# point for the client side. Idempotent; never overrides real shell exports.
load_dotenv()

CURRENT_PROJ_DIR = os.path.dirname(os.path.abspath(__file__))


def sanitize_prompt_text(text, max_len: int = 2000) -> str:
    """Flatten untrusted text before it is interpolated into an LLM prompt.

    DEFENSE-IN-DEPTH ONLY. This reduces the surface for prompt-injection
    (newlines/control chars used to fake instruction boundaries) and caps length,
    but it is NOT the security boundary — generated plans are constrained by the
    AST allowlist + PlanPolicy in plan_execution.py. Do not rely on this to make
    untrusted text safe.
    """
    if text is None:
        return ""
    s = str(text)
    # Replace control chars / newlines / tabs with single spaces.
    s = "".join(ch if (ch.isprintable() or ch == " ") else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len] + " …[truncated]"
    return s

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

def evaluate_value(s: str) -> PROBE_RET_TYPE:
    if s.lstrip('-').isdigit():  # Check for negative integers
        return int(s)
    elif s.lstrip('-').replace('.', '', 1).isdigit():  # Check for negative floats
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

def undistort_image(img: cv2.Mat, K: ndarray, D: ndarray, balance: float=0.2) -> cv2.Mat:
    """
    Undistort an image with optional balance parameter to control field of view
    
    :param img: Input image
    :param K: Camera matrix
    :param D: Distortion coefficients
    :param balance: Balance parameter to control FOV (0.0 to 1.0)
    :return: Undistorted image
    """
    dim1 = img.shape[:2][::-1]
    
    # Compute new camera matrix
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, dim1, np.eye(3), balance=balance
    )
    
    # Create map for undistortion
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2
    )
    
    # Remap the image
    undistorted_img = cv2.remap(
        img, map1, map2, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT
    )
    
    return undistorted_img