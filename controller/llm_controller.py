from PIL import Image
import queue, time, os, json
from typing import Optional
import asyncio
import uuid

from .yolo_client import YoloClient, SharedYoloResult
from .yolo_grpc_client import YoloGRPCClient
from .tello_wrapper import TelloWrapper
from .virtual_drone_wrapper import VirtualDroneWrapper
from .abs.drone_wrapper import DroneWrapper
from .vision_skill_wrapper import VisionSkillWrapper
from .llm_planner import LLMPlanner
from .skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg
from .utils import print_t, input_t
from .minispec_interpreter import MiniSpecInterpreter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class LLMController():
    def __init__(self, use_virtual_drone=True, message_queue: Optional[queue.Queue]=None):
        self.yolo_results_image_queue = queue.Queue(maxsize=30)
        self.shared_yolo_result = SharedYoloResult()
        self.yolo_client = YoloGRPCClient(shared_yolo_result=self.shared_yolo_result)
        self.vision = VisionSkillWrapper(self.shared_yolo_result)
        self.latest_frame = None
        self.controller_state = True
        self.controller_wait_takeoff = True
        self.message_queue = message_queue
        if message_queue is None:
            self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        else:
            self.cache_folder = message_queue.get()

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        
        if use_virtual_drone:
            print_t("[C] Start virtual drone...")
            self.drone: DroneWrapper = VirtualDroneWrapper()
        else:
            print_t("[C] Start real drone...")
            self.drone: DroneWrapper = TelloWrapper()
        
        self.planner = LLMPlanner()

        # load low-level skills
        self.low_level_skillset = SkillSet(level="low")
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_forward", self.drone.move_forward, "Move forward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_backward", self.drone.move_backward, "Move backward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_left", self.drone.move_left, "Move left by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_right", self.drone.move_right, "Move right by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_up", self.drone.move_up, "Move up by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_down", self.drone.move_down, "Move down by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_cw", self.drone.turn_cw, "Rotate clockwise/right by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_ccw", self.drone.turn_ccw, "Rotate counterclockwise/left by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("delay", self.delay, "Wait for specified microseconds", args=[SkillArg("milliseconds", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("is_visible", self.vision.is_visible, "Check the visibility of target object", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_x", self.vision.object_x, "Get object's X-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_y", self.vision.object_y, "Get object's Y-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_w", self.vision.object_w, "Get object's width in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_h", self.vision.object_h, "Get object's height in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("log", self.log, "Output text to console", args=[SkillArg("text", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("picture", self.picture, "Take a picture"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("query", self.planner.request_execution, "Query the LLM for reasoning", args=[SkillArg("question", str)]))

        # load high-level skills
        self.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)
        with open(os.path.join(CURRENT_DIR, "assets/high_level_skills.json"), "r") as f:
            json_data = json.load(f)
            for skill in json_data:
                self.high_level_skillset.add_skill(HighLevelSkillItem.load_from_dict(skill))

        MiniSpecInterpreter.low_level_skillset = self.low_level_skillset
        MiniSpecInterpreter.high_level_skillset = self.high_level_skillset
        self.planner.init(high_level_skillset=self.high_level_skillset, low_level_skillset=self.low_level_skillset, vision_skill=self.vision)

    def picture(self):
        img_path = os.path.join(self.cache_folder, f"{uuid.uuid4()}.jpg")
        Image.fromarray(self.latest_frame).save(img_path)
        self.append_message((img_path,))
        return True

    def log(self, text: str):
        self.append_message(text)
        print_t(f"[LOG] {text}")

    def delay(self, ms: int):
        time.sleep(ms / 1000.0)

    def append_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def stop_controller(self):
        self.controller_state = False

    def get_latest_frame(self):
        return self.yolo_results_image_queue.get()
    
    def execute_minispec(self, minispec: str):
        interpreter = MiniSpecInterpreter()
        return interpreter.execute(minispec)

    def execute_task_description(self, task_description: str):
        if self.controller_wait_takeoff:
            self.append_message("[Warning] Controller is waiting for takeoff...")
            return
        self.append_message('[TASK]: ' + task_description)
        for _ in range(1):
            t1 = time.time()
            result = self.planner.request_planning(task_description)
            t2 = time.time()
            print_t(f"[C] Planning time: {t2 - t1}")
            self.append_message('[PLAN]: ' + result + f', received in ({t2 - t1:.2f}s)')
            consent = input_t(f"[C] Get plan: {result}, executing?")
            if consent == 'n':
                print_t("[C] > Plan rejected <")
                return
            self.execute_minispec(result)
        self.append_message('Task complete!')
        self.append_message('end')

    def start_robot(self):
        print_t("[C] Drone is taking off...")
        self.drone.connect()
        self.drone.takeoff()
        self.drone.move_up(30)
        self.drone.start_stream()
        self.controller_wait_takeoff = False

    def stop_robot(self):
        print_t("[C] Drone is landing...")
        self.drone.land()
        self.drone.stop_stream()
        self.controller_wait_takeoff = True

    def capture_loop(self, asyncio_loop):
        print_t("[C] Start capture loop...")
        frame_reader = self.drone.get_frame_reader()
        while self.controller_state:
            self.drone.keep_active()
            self.latest_frame = frame_reader.frame
            image = Image.fromarray(self.latest_frame)

            if self.yolo_client.local_service():
                self.yolo_client.detect_local(image)
            else:
                # asynchronously send image to yolo server
                asyncio_loop.call_soon_threadsafe(asyncio.create_task, self.yolo_client.detect(image))

            latest_yolo_result = self.yolo_client.retrieve()
            if latest_yolo_result is not None:
                YoloClient.plot_results(latest_yolo_result[0], latest_yolo_result[1]['result'])
                self.yolo_results_image_queue.put(latest_yolo_result[0])
            time.sleep(0.030)
        # Cancel all running tasks (if any)
        for task in asyncio.all_tasks(asyncio_loop):
            task.cancel()
        asyncio_loop.stop()
        print_t("[C] Capture loop stopped")