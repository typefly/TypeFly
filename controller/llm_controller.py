from PIL import Image
import queue, time, os, json
from typing import Optional
import asyncio

from .yolo_client import YoloClient, SharedYoloResults
from .yolo_grpc_client import YoloGRPCClient
from .tello_wrapper import TelloWrapper
from .virtual_drone_wrapper import VirtualDroneWrapper
from .abs.drone_wrapper import DroneWrapper
from .vision_skill_wrapper import VisionSkillWrapper
from .llm_planner import LLMPlanner
from .skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg
from .utils import print_t, input_t
from .minispec_interpreter import MiniSpecInterpreter

current_directory = os.path.dirname(os.path.abspath(__file__))

class LLMController():
    def __init__(self, use_virtual_drone=True, message_queue: Optional[queue.Queue]=None):
        self.yolo_results_image_queue = queue.Queue(maxsize=30)
        self.yolo_results = SharedYoloResults()
        self.yolo_client = YoloGRPCClient(shared_yolo_results=self.yolo_results)
        self.controller_state = True
        self.controller_wait_takeoff = True
        self.message_queue = message_queue
        if use_virtual_drone:
            print_t("[C] Start virtual drone...")
            self.drone: DroneWrapper = VirtualDroneWrapper()
        else:
            print_t("[C] Start real drone...")
            self.drone: DroneWrapper = TelloWrapper()
        self.vision = VisionSkillWrapper(self.yolo_results)
        self.planner = LLMPlanner()

        # load low-level skills
        self.low_level_skillset = SkillSet(level="low")
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_forward", self.drone.move_forward, "Move drone forward by a distance", args=[SkillArg("dis", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_backward", self.drone.move_backward, "Move drone backward by a distance", args=[SkillArg("dis", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_left", self.drone.move_left, "Move drone left by a distance", args=[SkillArg("dis", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_right", self.drone.move_right, "Move drone right by a distance", args=[SkillArg("dis", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_up", self.drone.move_up, "Move drone up by a distance", args=[SkillArg("dis", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_down", self.drone.move_down, "Move drone down by a distance", args=[SkillArg("dis", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_cw", self.drone.turn_cw, "Turn drone clockwise by a degree", args=[SkillArg("deg", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_c_cw", self.drone.turn_ccw, "Turn drone counterclockwise by a degree", args=[SkillArg("deg", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("is_in_sight", self.vision.is_in_sight, "Check if an object is in sight", args=[SkillArg("obj_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("obj_loc_x", self.vision.obj_loc_x, "Get x loc of an object", args=[SkillArg("obj_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("obj_loc_y", self.vision.obj_loc_y, "Get y loc of an object", args=[SkillArg("obj_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("obj_size_w", self.vision.obj_size_w, "Get width of an object", args=[SkillArg("obj_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("obj_size_h", self.vision.obj_size_h, "Get height of an object", args=[SkillArg("obj_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("query", self.planner.request_execution, "Query the LLM for reasoning", args=[SkillArg("question", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("log", self.log, "Print the text to console", args=[SkillArg("text", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("delay", self.delay, "Sleep for some microseconds", args=[SkillArg("ms", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("picture", self.picture, "Take a picture"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("verification", self.verification, "Restart plan if task description has not been met", args=[SkillArg("retry", int)]))

        # load high-level skills
        self.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)
        with open(os.path.join(current_directory, "./assets/high_level_skills.json"), "r") as f:
            json_data = json.load(f)
            for skill in json_data:
                self.high_level_skillset.add_skill(HighLevelSkillItem.load_from_dict(skill))

        MiniSpecInterpreter.low_level_skillset = self.low_level_skillset
        MiniSpecInterpreter.high_level_skillset = self.high_level_skillset
        self.planner.init(high_level_skillset=self.high_level_skillset, low_level_skillset=self.low_level_skillset, vision_skill=self.vision)

    def verification(self, retry: int):
        pass

    def picture(self):
        pass

    def log(self, text: str):
        print_t(f"[LOG] {text}")

    def delay(self, ms: int):
        time.sleep(ms / 1000.0)

    def stop_controller(self):
        self.controller_state = False

    def get_latest_frame(self):
        return self.yolo_results_image_queue.get()
    
    def execute_minispec(self, minispec: str):
        interpreter = MiniSpecInterpreter()
        return interpreter.execute(minispec)

    def execute_task_description(self, task_description: str):
        if self.controller_wait_takeoff:
            self.message_queue.put("[Warning] Controller is waiting for takeoff...")
            return
        self.message_queue.put('[TASK]: ' + task_description)
        for _ in range(1):
            t1 = time.time()
            result = self.planner.request_planning(task_description)
            t2 = time.time()
            print_t(f"[C] Planning time: {t2 - t1}")
            self.message_queue.put('[PLAN]: ' + result + f', received in ({t2 - t1:.2f}s)')
            consent = input_t(f"[C] Get plan: {result}, executing?")
            if consent == 'n':
                print_t("[C] > Plan rejected <")
                return
            self.execute_minispec(result)
        self.message_queue.put('Task complete!')
        self.message_queue.put('end')

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
            frame = frame_reader.frame
            image = Image.fromarray(frame)

            if self.yolo_client.local_service():
                self.yolo_client.detect_local(image)
            else:
                # asynchronously send image to yolo server
                asyncio_loop.call_soon_threadsafe(asyncio.create_task, self.yolo_client.detect(image))

            latest_result = self.yolo_client.retrieve()
            if latest_result is not None:
                YoloClient.plot_results(latest_result[0], latest_result[1]['result'])
                self.yolo_results_image_queue.put(latest_result[0])
            time.sleep(0.030)
        for task in asyncio.all_tasks(asyncio_loop):
            task.cancel()
        asyncio_loop.stop()
        print_t("[C] Capture loop stopped")