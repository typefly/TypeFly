import time, os, math
from typing import Any
import numpy as np
import threading, requests
from overrides import overrides
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R

import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs import msg
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from ..robot_wrapper import RobotWrapper, RobotObservation
from ..robot_info import RobotInfo
from ..yolo_client import YoloClient
from ..utils import quaternion_to_rpy, print_t, undistort_image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

GO2_CAM_K = np.array([
    [818.18507419, 0.0, 637.94628188],
    [0.0, 815.32431463, 338.3480119],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

GO2_CAM_D = np.array([[-0.07203219],
                      [-0.05228525],
                      [ 0.05415833],
                      [-0.02288355]], dtype=np.float32)

class Go2Observation(RobotObservation):
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        self.yolo_client = YoloClient(robot_info)
        self.init_ros_observation()

    def init_ros_observation(self):
        def _ros_spin():
            rclpy.spin(self.node)        

        def _ros_image_callback(image: msg.Image):
            # Convert RGB to BGR
            buffer = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 3))[:, :, ::-1]
            # Undistort the image
            buffer = undistort_image(buffer, GO2_CAM_K, GO2_CAM_D)
            self._image = Image.fromarray(buffer)

        def _ros_odom_callback(odom: Odometry):
            self._position = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z])
            ori = odom.pose.pose.orientation
            self._orientation = quaternion_to_rpy(ori.x, ori.y, ori.z, ori.w)
        
        def _tf_callback(msg: TFMessage):
            _eye4 = np.eye(4)
            # Extract position and orientation from the TF message
            for tf in msg.transforms:
                t = tf.transform.translation
                r = tf.transform.rotation

                # 🚀 Correct frame assignment
                if tf.child_frame_id == "base_link":      # odom → robot
                    self.odom2robot_translation[:] = [t.x, t.y, 0.0]
                    self.odom2robot_rotation[:] = [r.x, r.y, r.z, r.w]
                elif tf.child_frame_id == "odom":         # map → odom
                    self.map2odom_translation[:] = [t.x, t.y, 0.0]
                    self.map2odom_rotation[:] = [r.x, r.y, r.z, r.w]

            T_map_odom = _eye4.copy()
            T_map_odom[:3, :3] = R.from_quat(self.map2odom_rotation).as_matrix()
            T_map_odom[:3, 3] = self.map2odom_translation

            T_odom_robot = _eye4.copy()
            T_odom_robot[:3, :3] = R.from_quat(self.odom2robot_rotation).as_matrix()
            T_odom_robot[:3, 3] = self.odom2robot_translation

            T_map_robot = T_map_odom @ T_odom_robot

            self._position[:] = T_map_robot[:3, 3].copy()
            self._position[2] = 0.4  # fix z height to 0.4m

            RR = R.from_matrix(T_map_robot[:3, :3])
            self._orientation[:] = RR.as_euler('xyz')
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Match camera publisher
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Initialize ROS if not already done
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node('typefly_go2_observation')
        self.node.create_subscription(
            msg.Image, 
            '/camera/image_raw',  # Change this to your actual topic
            _ros_image_callback, 
            qos_profile
        )
        self.node.create_subscription(
            Odometry, 
            '/tf',  # Change this to your actual topic
            _tf_callback, 
            10
        )
        self.ros_thread = threading.Thread(target=_ros_spin)
        
    @overrides
    def _start(self):
        self.ros_thread.start()
    
    @overrides
    def _stop(self):
        if rclpy.ok():
            rclpy.shutdown()
        self.ros_thread.join()
        self.node.destroy_node()
        
    @overrides
    async def process_image(self, image: Image.Image):
        await self.yolo_client.detect(image)
    
    @overrides
    def fetch_processed_result(self) -> dict[str, Any]:
        _, object_list = self.yolo_client.latest_result
        return {
            "yolo": object_list
        }

class Go2Wrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo, system_skill_func: list[callable]):
        super().__init__(robot_info, Go2Observation(robot_info), system_skill_func)

        self.node = rclpy.create_node('typefly_go2_control')
        self.control_publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.move_speed = 0.8
        self.rotate_speed = 1.0
        self.control_dt = 0.1
        self.action_wait_time = 1.0

    @overrides
    def start(self) -> bool:
        self.observation.start()
        return True

    @overrides
    def stop(self) -> bool:
        self.observation.stop()
        return True

    def _stop_moving(self, wait_time: float = 0.0):
        twist = Twist()
        self.control_publisher.publish(twist)
        time.sleep(wait_time)

    # def _move(self, linear_x: float=0.0, linear_y: float=0.0, angular_z: float=0.0, duration: float=3.0):
    #     """
    #     Helper function to publish Twist messages for a specified duration.
    #     """
    #     twist = Twist()
    #     twist.linear.x = linear_x
    #     twist.linear.y = linear_y
    #     twist.angular.z = angular_z

    #     start_time = time.time()
    #     while time.time() - start_time < duration:
    #         self.control_publisher.publish(twist)
    #         time.sleep(self.ros_control_dt)
        
    #     self._stop_moving(self.action_wait_time)

    @overrides
    def _move(self, dx: float, dy: float):
        """
        Moves the robot by the specified distance in the x (forward/backward) and y (left/right) directions.
        """
        print(f"-> Move by ({dx}, {dy}) cm")
        
        # Convert distances from cm to meters
        dx_m = dx / 100.0
        dy_m = dy / 100.0

        # Calculate duration based on speed
        duration = max(abs(dx_m), abs(dy_m)) / self.ros_move_speed if self.ros else max(abs(dx_m), abs(dy_m))

        # Perform the movement
        # self._move(linear_x=dx_m, linear_y=dy_m, duration=duration)

    @overrides
    def _rotate(self, deg: float):
        """
        Rotates the robot by the specified angle in degrees.
        """
        print(f"-> Rotate by {deg} degrees")
        
        # Convert degrees to radians
        rad = math.radians(deg)

        # Calculate duration based on rotation speed
        if self.ros:
            duration = abs(rad) / self.ros_rotate_speed if self.ros else abs(rad)
            angular_z = self.ros_rotate_speed if deg > 0 else -self.ros_rotate_speed
        else:
            duration = 3.0
            angular_z = rad

        # Perform the rotation
        