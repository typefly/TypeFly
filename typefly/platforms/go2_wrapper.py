import time, os
import numpy as np
import threading
from overrides import overrides
from PIL import Image
import cv2

from ..robot_wrapper import RobotWrapper, RobotObservation
from ..robot_info import RobotInfo
from ..yolo_client import YoloClient
from ..skillset import SkillSet, SkillArg, SkillSetLevel
from ..utils import quaternion_to_rpy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class Go2Observation(RobotObservation):
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        self.yolo_client = YoloClient(robot_info)

        self.ros = self.robot_info.extra.get("ros", True)
        if self.ros:
            self.init_ros_observation()
        else:
            self.init_custom_sdk(self.robot_info.extra)

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

    def init_custom_sdk(self, extra: dict):
        if "ip" not in extra or "port" not in extra:
            raise ValueError("IP and port must be provided in extra")
        self.ip = extra["ip"]
        self.port = extra["port"]

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

        self.dog_move_speed = 0.8
        self.dog_control_dt = 0.1
        self.dog_wait_time = 1.0

        high_level_skills = [
            {
                "name": "scan",
                "definition": "{8{?is_visible($1){->True}turn_cw(45)}->False}",
                "description": "Rotate to find object $1 when it's *not* in current view",
            },
            {
                "name": "scan_description",
                "definition": "{8{_1=probe($1);?_1!=False{->_1}turn_cw(45)}->False}",
                "description": "Rotate to find object $1 when it's *not* in current view",
            },
            {
                "name": "orienting",
                "definition": "4{_1=ox($1);?_1>0.6{tc(15)}:?_1<0.4{tu(15)}:{->True}}->False",
                "description": "Rotate to align with object $1",
            },
            {
                "name": "goto",
                "definition": "?orienting($1){move_forward(80)}",
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

    def _stop_moving(self, wait_time: float = 0.0):
        if self.ros:
            from geometry_msgs.msg import Twist
            twist = Twist()
            self.control_publisher.publish(twist)
        time.sleep(wait_time)

    @overrides
    def move_forward(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving forward {dist} cm")
        if self.ros:
            from geometry_msgs.msg import Twist
            twist = Twist()
            twist.linear.x = self.dog_move_speed

            t = dist / self.dog_move_speed / 100.0
            start_time = time.time()
            while time.time() - start_time < t:
                self.control_publisher.publish(twist)
                time.sleep(self.dog_control_dt)

        self._stop_moving()
        return True, False

    @overrides
    def move_backward(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving backward {dist} cm")
        if self.ros:
            from geometry_msgs.msg import Twist
            twist = Twist()
            twist.linear.x = -self.dog_move_speed

            t = dist / self.dog_move_speed / 100.0
            start_time = time.time()
            while time.time() - start_time < t:
                self.control_publisher.publish(twist)
                time.sleep(self.dog_control_dt)

        self._stop_moving()
        return True, False

    @overrides
    def move_left(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving left {dist} cm")
        if self.ros:
            from geometry_msgs.msg import Twist
            twist = Twist()
            twist.linear.y = self.dog_move_speed

            t = dist / 100.0 / self.dog_move_speed
            start_time = time.time()
            while time.time() - start_time < t:
                self.control_publisher.publish(twist)
                time.sleep(self.dog_control_dt)

        self._stop_moving()
        return True, False

    @overrides
    def move_right(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving right {dist} cm")
        if self.ros:
            from geometry_msgs.msg import Twist
            twist = Twist()
            twist.linear.y = -self.dog_move_speed

            t = dist / 100.0 / self.dog_move_speed
            start_time = time.time()
            while time.time() - start_time < t:
                self.control_publisher.publish(twist)
                time.sleep(self.dog_control_dt)

        self._stop_moving()
        return True, False

    def turn_45(self, clockwise: bool):
        if self.ros:
            from geometry_msgs.msg import Twist
            twist = Twist()
            twist.angular.z = -1.5 if clockwise else 1.5
            start_time = time.time()
            while time.time() - start_time < 0.3:
                self.control_publisher.publish(twist)
                time.sleep(self.dog_control_dt)

    def turn_slow(self, deg: int, clockwise: bool):
        if self.ros:
            from geometry_msgs.msg import Twist
            twist = Twist()
            twist.angular.z = -0.5 if clockwise else 0.5

            t = deg * 0.02 / 0.5
            start_time = time.time()
            while time.time() - start_time < t:
                self.control_publisher.publish(twist)
                time.sleep(self.dog_control_dt)

    @overrides
    def turn_ccw(self, deg: int) -> tuple[bool, bool]:
        print(f"-> Turning CCW {deg} degrees")
        if self.ros:
            for _ in range(deg // 45):
                self.turn_45(False)

            self.turn_slow(deg % 45, False)
        self._stop_moving(self.dog_wait_time)
        return True, False

    @overrides
    def turn_cw(self, deg: int) -> tuple[bool, bool]:
        print(f"-> Turning CW {deg} degrees")
        if self.ros:
            for _ in range(deg // 45):
                self.turn_45(True)
            
            self.turn_slow(deg % 45, True)
        self._stop_moving(self.dog_wait_time)
        return True, False
