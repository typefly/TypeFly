import time, os, math
from typing import Any
import numpy as np
import threading
from overrides import overrides
from PIL import Image
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
from ..pid import PID

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
        
        # Initialize the transformation matrices
        self.odom2robot_translation = [0.0, 0.0, 0.0]
        self.odom2robot_rotation = [0.0, 0.0, 0.0, 1.0]
        self.map2odom_translation = [0.0, 0.0, 0.0]
        self.map2odom_rotation = [0.0, 0.0, 0.0, 1.0]
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

            # print_t(f"-> Position: {self._position}, Orientation: {self._orientation}")
        
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
            '/camera/image_raw',
            _ros_image_callback, 
            qos_profile
        )
        self.node.create_subscription(
            TFMessage,
            '/tf',
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
    def __init__(self, robot_info: RobotInfo):
        super().__init__(robot_info, Go2Observation(robot_info))

        self.node = rclpy.create_node('typefly_go2_control')
        self.control_publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.control_dt = 0.1

        self.speed_xy = 0.4
        self.speed_yaw = 1.0

        """
        The go2_ros2_sdk has a huge latency in TF update. So we use simple velocity control instead of PID control.
        """
        # self.pid_yaw = PID(10.0, 0.0, 0.0, 10.0, 10.0, 0.5, 1.0)
        # self.pid_x = PID(1.5, 1.0, 0.1, 10.0, 10.0, 0.5, 1.0)
        # self.pid_y = PID(1.5, 1.0, 0.1, 10.0, 10.0, 0.5, 1.0)

    @overrides
    def start(self) -> bool:
        self.obs.start()
        return True

    @overrides
    def stop(self) -> bool:
        self.obs.stop()
        return True

    def _stop_moving(self, wait_time: float = 0.0):
        twist = Twist()
        self.control_publisher.publish(twist)
        time.sleep(wait_time)

    def _send_twist(self, linear_x: float=0.0, linear_y: float=0.0, angular_z: float=0.0):
        """
        Helper function to publish Twist messages for a specified duration.
        """
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z
        self.control_publisher.publish(twist)

    @overrides
    def _move(self, dx: float, dy: float):
        """
        Moves the robot by the specified distance in the x (forward/backward) and y (left/right) directions.
        This is a simple velocity control.
        """
        print(f"-> Move by ({dx}, {dy}) m")

        duration = max(abs(dx), abs(dy)) / self.speed_xy / 1.5

        if abs(dx) > abs(dy):
            vx = self.speed_xy if dx > 0 else -self.speed_xy
            vy = dy / duration
        else:
            vx = dx / duration
            vy = self.speed_xy if dy > 0 else -self.speed_xy

        start_time = time.time()
        while time.time() - start_time < duration:
            self._send_twist(vx, vy, 0.0)
            print_t(f"-> Move: vx={vx}, vy={vy}")
            time.sleep(self.control_dt)
        self._send_twist(0.0, 0.0, 0.0)

    @overrides
    def _rotate(self, deg: float):
        """
        Rotates the robot by the specified angle in degrees.
        This is a simple velocity control.
        """
        print(f"-> Rotate by {deg} degrees")
        duration = abs(math.radians(deg)) / self.speed_yaw
        start_time = time.time()
        while time.time() - start_time < duration:
            self._send_twist(0.0, 0.0, self.speed_yaw if deg > 0 else -self.speed_yaw)
            print_t(f"-> Rotate: vyaw={self.speed_yaw if deg > 0 else -self.speed_yaw}")
            time.sleep(self.control_dt)
        self._send_twist(0.0, 0.0, 0.0)

    """
    The go2_ros2_sdk has a huge latency in TF update. So we use simple velocity control instead of PID control.
    """
    # @overrides
    # def _move(self, dx: float, dy: float):
    #     """
    #     Moves the robot by the specified distance in the x (forward/backward) and y (left/right) directions.
    #     """
    #     print(f"-> Move by ({dx}, {dy}) m")
    #     init_x = self.obs._position[0]
    #     init_y = self.obs._position[1]
    #     init_yaw = self.obs._orientation[2]
    #     target_x = init_x + dx * math.cos(init_yaw) - dy * math.sin(init_yaw)
    #     target_y = init_y + dx * math.sin(init_yaw) + dy * math.cos(init_yaw)

    #     timeout = 2.0
    #     action_start_time = time.time() 
    #     while time.time() - action_start_time < timeout:
    #         loop_start_time = time.time()
    #         current_x = self.obs._position[0]
    #         current_y = self.obs._position[1]
    #         current_yaw = self.obs._orientation[2]

    #         # Compute error in world frame
    #         error_world_x = target_x - current_x
    #         error_world_y = target_y - current_y

    #         # Convert error into body frame
    #         error_body_x = error_world_x * math.cos(current_yaw) + error_world_y * math.sin(current_yaw)
    #         error_body_y = -error_world_x * math.sin(current_yaw) + error_world_y * math.cos(current_yaw)

    #         print_t(f"-> Error: error_body_x={error_body_x}, error_body_y={error_body_y}")

    #         if math.hypot(error_body_x, error_body_y) < 0.08:
    #             break

    #         vx = self.pid_x.update(error_body_x)
    #         vy = self.pid_y.update(error_body_y)

    #         print_t(f"-> Move: vx={vx}, vy={vy}, current=(x={current_x}, y={current_y}), yaw={current_yaw}")
    #         self._send_twist(vx, vy, 0.0)
    #         time.sleep(max(0, self.control_dt - (time.time() - loop_start_time)))
    #     self._send_twist(0.0, 0.0, 0.0)

    # @overrides
    # def _rotate(self, deg: float):
    #     """
    #     Rotates the robot by the specified angle in degrees.
    #     """
    #     print(f"-> Rotate by {deg} degrees")
    #     init_yaw = self.obs._orientation[2]
    #     delta_rad = math.radians(deg)
    #     accumulated_angle = 0.0
    #     previous_yaw = init_yaw

    #     timeout = 5.0
    #     action_start_time = time.time()
    #     while time.time() - action_start_time < timeout:
    #         cycle_start_time = time.time()
    #         current_yaw = self.obs._orientation[2]
    #         yaw_diff = current_yaw - previous_yaw

    #         # Normalize yaw difference to the range [-pi, pi]
    #         if yaw_diff > math.pi:
    #             yaw_diff -= 2 * math.pi
    #         elif yaw_diff < -math.pi:
    #             yaw_diff += 2 * math.pi

    #         accumulated_angle += yaw_diff
    #         previous_yaw = current_yaw

    #         remaining_angle = delta_rad - accumulated_angle

    #         # print_t(f"-> Remaining angle: {math.degrees(remaining_angle):.2f} degrees, accumulated: {math.degrees(accumulated_angle):.2f} degrees")
    #         if abs(remaining_angle) < 0.01 or delta_rad * remaining_angle < 0:
    #             # If the remaining angle is small enough or we have overshot the target
    #             break

    #         vyaw = self.pid_yaw.update(remaining_angle)
    #         # print_t(f"-> vyaw: {vyaw:.2f} rad/s")
    #         print_t(f"-> Rotate: vyaw={vyaw:.2f} rad/s")
    #         self._send_twist(0.0, 0.0, vyaw)
    #         time.sleep(max(0, self.control_dt - (time.time() - cycle_start_time)))
    #     self._send_twist(0.0, 0.0, 0.0)