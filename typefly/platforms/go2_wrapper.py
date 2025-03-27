import time, os
import numpy as np
import asyncio, threading
from overrides import overrides
from PIL import Image
from sensor_msgs import msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from ..robot_wrapper import RobotWrapper, RobotObservation
from ..robot_info import RobotInfo
from ..yolo_client import YoloClient
from ..skillset import SkillSet, SkillArg, SkillSetLevel
from ..utils import quaternion_to_rpy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class Go2Observation(RobotObservation):
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info)
        self.interval: float = 1.0 / rate
        self.yolo_client = YoloClient(robot_info)

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('typefly_go2_observation')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Match camera publisher
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.gstreamer = self.robot_info.extra.get("gstreamer", False)
        if self.gstreamer:
            self._init_gstreamer()
        else:
            self.node.create_subscription(
                msg.Image, 
                '/camera/image_raw',  # Change this to your actual topic
                self.ros_image_callback, 
                qos_profile
            )

        self.node.create_subscription(
            Odometry, 
            '/odom',  # Change this to your actual topic
            self.odom_callback, 
            qos_profile
        )

        self.ros_thread = threading.Thread(target=self.ros_spin)

    def _init_gstreamer(self):
        import sys
        sys.path.append("/usr/lib/python3/dist-packages")

        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst, GLib

        Gst.init(None)

        def on_new_sample(sink):
            sample = sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR

            buffer = sample.get_buffer()
            caps = sample.get_caps()
            width = caps.get_structure(0).get_value('width')
            height = caps.get_structure(0).get_value('height')

            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR

            frame = np.frombuffer(map_info.data, np.uint8).reshape((height, width, 3))
            buffer.unmap(map_info)

            self._image = Image.fromarray(frame[:, :, ::-1])
            return Gst.FlowReturn.OK

        pipeline_str = """
            udpsrc address=230.1.1.1 port=1720 multicast-iface=wlan0
            ! application/x-rtp, media=video, encoding-name=H264
            ! rtph264depay
            ! h264parse
            ! avdec_h264
            ! videoconvert
            ! video/x-raw, format=BGR
            ! appsink name=appsink emit-signals=true max-buffers=1 drop=true
        """

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("appsink")
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.connect("new-sample", on_new_sample)

        print("GStreamer pipeline created")
        self.pipeline.set_state(Gst.State.PLAYING)

        from gi.repository import GLib  # Re-import here in case it's needed outside callback
        self.glib_loop = GLib.MainLoop()
        self.glib_thread = threading.Thread(target=self.glib_loop.run, daemon=True)
        self.glib_thread.start()

    def ros_image_callback(self, image: msg.Image):
        # Convert RGB to BGR
        buffer = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 3))[:, :, ::-1]
        self._image = Image.fromarray(buffer)

    def odom_callback(self, odom: Odometry):
        self._position = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z])
        ori = odom.pose.pose.orientation
        self._orientation = quaternion_to_rpy(ori.x, ori.y, ori.z, ori.w)

    def ros_spin(self):
        rclpy.spin(self.node)
    
    @overrides
    def _start(self):
        if not self.ros_thread.is_alive():
            self.ros_thread.start()
    
    @overrides
    def _stop(self):
        if self.gstreamer:
            print("Stopping GStreamer pipeline...")
            from gi.repository import GLib
            self.pipeline.set_state(Gst.State.NULL)
            self.glib_loop.quit()
            self.glib_thread.join()

        # Shutdown ROS to unblock rclpy.spin()
        if rclpy.ok():
            rclpy.shutdown()
        if self.ros_thread.is_alive():
            self.ros_thread.join()
        self.node.destroy_node()

    @overrides
    def update_observation(self):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def schedule_tasks():
            tasks = set()
            
            while self.running:
                start_time = time.time()

                # Add a new task to the set
                if self._image is not None:
                    task = asyncio.create_task(self.yolo_client.detect(self._image))
                    tasks.add(task)
                
                # Clean up completed tasks
                tasks = {t for t in tasks if not t.done()}
                with self._image_process_lock:
                    self._image_process_result = self.yolo_client.latest_result
                # Sleep for the interval
                elapsed_time = time.time() - start_time
                await asyncio.sleep(max(0, self.interval - elapsed_time))
        # Run the async function in the event loop
        loop.run_until_complete(schedule_tasks())

class Go2Wrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo, system_skill_func: list[callable]):
        super().__init__(robot_info, Go2Observation(robot_info), system_skill_func)

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
        twist = Twist()
        self.control_publisher.publish(twist)
        time.sleep(wait_time)

    @overrides
    def move_forward(self, dist: int) -> tuple[bool, bool]:
        print(f"-> Moving forward {dist} cm")
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
        twist = Twist()
        twist.angular.z = -1.5 if clockwise else 1.5
        start_time = time.time()
        while time.time() - start_time < 0.3:
            self.control_publisher.publish(twist)
            time.sleep(self.dog_control_dt)

    def turn_slow(self, deg: int, clockwise: bool):
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
        for _ in range(deg // 45):
            self.turn_45(False)

        self.turn_slow(deg % 45, False)
        self._stop_moving(self.dog_wait_time)
        return True, False

    @overrides
    def turn_cw(self, deg: int) -> tuple[bool, bool]:
        print(f"-> Turning CW {deg} degrees")
        for _ in range(deg // 45):
            self.turn_45(True)
        
        self.turn_slow(deg % 45, True)
        self._stop_moving(self.dog_wait_time)
        return True, False
