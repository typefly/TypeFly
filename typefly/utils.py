import datetime, cv2
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