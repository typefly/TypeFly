import time, os, math
import numpy as np
import threading, requests
from overrides import overrides
from PIL import Image
import cv2

from ..robot_wrapper import RobotWrapper, RobotObservation
from ..robot_info import RobotInfo
from ..yolo_client import YoloClient
from ..skillset import SkillSet, SkillArg, SkillSetLevel
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

        self.ros = self.robot_info.extra.get("ros", True)
        if self.ros:
            self.init_ros_observation()
        else:
            self.init_custom_sdk()

    def init_ros_observation(self):
        from sensor_msgs import msg
        from nav_msgs.msg import Odometry
        import rclpy
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
        
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
            '/odom',  # Change this to your actual topic
            _ros_odom_callback, 
            qos_profile
        )
        self.ros_thread = threading.Thread(target=_ros_spin)

    def init_custom_sdk(self):
        # Use gstreamer and OpenCV to read the video stream
        # You need to start the gstreamer pipeline on the robot, see platforms/README.md
        GSTREAMER_PIPELINE_STR = """
            udpsrc address=230.1.1.1 port=1720 multicast-iface=wlan0
            ! application/x-rtp, media=video, encoding-name=H264
            ! rtph264depay
            ! h264parse
            ! avdec_h264
            ! videoconvert
            ! video/x-raw, format=BGR
            ! appsink name=appsink emit-signals=true max-buffers=1 drop=true
        """
        self.gstreamer_cap: cv2.VideoCapture = None
        def _gstreamer_spin():
            # must create the capture and read in the same thread
            self.gstreamer_cap = cv2.VideoCapture(GSTREAMER_PIPELINE_STR, cv2.CAP_GSTREAMER)
            if not self.gstreamer_cap.isOpened():
                raise RuntimeError("Failed to open GStreamer pipeline")
            while self.running:
                ret, frame = self.gstreamer_cap.read()
                if not ret:
                    continue
                # Convert the frame to RGB and store it in self._image
                
                frame = undistort_image(frame, GO2_CAM_K, GO2_CAM_D)
                self._image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
        self.gstreamer_thread = threading.Thread(target=_gstreamer_spin)
        
    @overrides
    def _start(self):
        if self.ros:
            self.ros_thread.start()
        else:
            self.gstreamer_thread.start()
    
    @overrides
    def _stop(self):
        if self.ros:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
            self.ros_thread.join()
            self.node.destroy_node()
        else:
            self.gstreamer_thread.join()
            if self.gstreamer_cap is not None:
                self.gstreamer_cap.release()
        
    @overrides
    async def process_image(self, image: Image.Image):
        await self.yolo_client.detect(image)
    
    @overrides
    def fetch_processed_result(self) -> tuple[Image.Image, list]:
        return self.yolo_client.latest_result

class Go2Wrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo, system_skill_func: list[callable]):
        super().__init__(robot_info, Go2Observation(robot_info), system_skill_func)

        self.ros = self.robot_info.extra.get("ros", True)
        if self.ros:
            from geometry_msgs.msg import Twist
            import rclpy
            self.node = rclpy.create_node('typefly_go2_control')
            self.control_publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)
            self.ros_move_speed = 0.8
            self.ros_rotate_speed = 1.0
            self.ros_control_dt = 0.1
        else:
            extra = self.robot_info.extra
            if "url" not in extra:
                raise ValueError("Control url must be provided in extra")
            self.robot_url = extra["url"]
            response = requests.get(self.robot_url)
            if response.status_code != 200:
                raise RuntimeError(f"[Go2] Failed to connect to robot at {self.robot_url}")
            else:
                print_t(f"[Go2] {response.text}")

            self.ll_skillset.add_low_level_skill("stand_up", lambda: self._action('stand_up'), "Stand up")
            self.ll_skillset.add_low_level_skill("lay_down", lambda: self._action('stand_down'), "Stand down")

        self.action_wait_time = 1.0

        high_level_skills = [
            {
                "name": "scan",
                "definition": "{8{?is_visible($1){->True}rotate(45)}->False}",
                "description": "Rotate to find a specific object $1 when it's *not* in current view",
            },
            {
                "name": "scan_description",
                "definition": "{8{_1=probe($1);?_1!=False{->_1}rotate(45)}->False}",
                "description": "Rotate to find an abstract object $1 when it's *not* in current view",
            },
            {
                "name": "orienting",
                "definition": "{_1=ox($1);rotate((0.5-_1)*80)}",
                "description": "Rotate to align with object $1",
            },
            {
                "name": "goto",
                "definition": "?orienting($1){move(80, 0)}",
                "description": "Move to object $1 in the view"
            }
        ]

        self.hl_skillset = SkillSet(SkillSetLevel.HIGH, self.ll_skillset)
        for skill in high_level_skills:
            self.hl_skillset.add_high_level_skill(skill['name'], skill['definition'], skill['description'])

    @overrides
    def start(self) -> bool:
        self.observation.start()
        return True

    @overrides
    def stop(self):
        self.observation.stop()

    def _action(self, action: str):
        if not self.ros:
            control = {
                "command": action,
                "timeout": 3.0
            }
            self._send_control(control)
        else:
            print_t(f"[Go2] Unsupported action for ros: {action}")

    def _send_control(self, control: dict):
        response = requests.post(
            self.robot_url + "control",
            json=control,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code != 200:
            print_t(f"[Go2] Failed to send command: {response.json()}")

    def _stop_moving(self, wait_time: float = 0.0):
        if self.ros:
            from geometry_msgs.msg import Twist
            twist = Twist()
            self.control_publisher.publish(twist)
        time.sleep(wait_time)

    def _move(self, linear_x: float=0.0, linear_y: float=0.0, angular_z: float=0.0, duration: float=3.0):
        """
        Helper function to publish Twist messages for a specified duration.
        """
        if self.ros:
            from geometry_msgs.msg import Twist
            twist = Twist()
            twist.linear.x = linear_x
            twist.linear.y = linear_y
            twist.angular.z = angular_z

            start_time = time.time()
            while time.time() - start_time < duration:
                self.control_publisher.publish(twist)
                time.sleep(self.ros_control_dt)
        else:
            control = { "timeout": duration }
            if linear_x != 0.0 or linear_y != 0.0:
                control["command"] = "move"
                control["dx"] = linear_x
                control["dy"] = linear_y
                control["body_frame"] = True
            elif angular_z != 0.0:
                control["command"] = "rotate"
                control["delta_rad"] = angular_z
            self._send_control(control)

        self._stop_moving(self.action_wait_time)

    @overrides
    def move(self, dx: float, dy: float) -> tuple[bool, bool]:
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
        self._move(linear_x=dx_m, linear_y=dy_m, duration=duration)

        return True, False

    @overrides
    def rotate(self, deg: float) -> tuple[bool, bool]:
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

        self._move(angular_z=angular_z, duration=duration)

        return True, False
        