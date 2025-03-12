from threading import Thread
from PIL import Image
import queue, time, os, json, shutil
from typing import Optional
import asyncio
import uuid
import appdirs

from .shared_frame import SharedFrame, Frame
from .yolo_client import YoloClient
from .platforms.tello_wrapper import TelloWrapper
from .platforms.virtual_robot_wrapper import VirtualRobotWrapper
from .robot_wrapper import RobotWrapper
from .vision_skill_wrapper import VisionSkillWrapper
from .llm_planner import LLMPlanner
from .skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg
from .utils import print_t, input_t
from .minispec_interpreter import MiniSpecInterpreter, Statement
from .robot_info import RobotInfo

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = cache_dir = appdirs.user_cache_dir("typefly")

class LLMController():
    def __init__(self, robot_info_list: list[RobotInfo], message_queue: Optional[queue.Queue]=None):
        self.message_queue = message_queue

        # self.planner = LLMPlanner(robot_info_list)

        # cache folder
        self.cache_folder = CACHE_DIR
        os.makedirs(self.cache_folder, exist_ok=True)

        system_skill_funcs = [
            self.system_skill_log,
            self.system_skill_delay,
            self.system_skill_take_picture,
            self.system_skill_re_plan,
            self.system_skill_probe
        ]

        self.robots: dict[RobotInfo, RobotWrapper] = {}
        for info in robot_info_list:
            if info.robot_type == "virtual":
                self.robots[info] = VirtualRobotWrapper(info, system_skill_funcs)
            elif info.robot_type == "tello":
                self.robots[info] = TelloWrapper(info, system_skill_funcs)
            # elif info.robot_type == "go2":
            #     pass
        
        # self.planner.init(high_level_skillset=self.high_level_skillset, low_level_skillset=self.low_level_skillset, vision_skill=self.vision)

        self.current_plan = None
        self.execution_history = None

    ### system skills
    def system_skill_log(self, text: str) -> tuple[None, bool]:
        self._send_message(f"[LOG] {text}")
        print_t(f"[LOG] {text}")
        return None, False
    
    def system_skill_delay(self, sec: float) -> tuple[None, bool]:
        time.sleep(sec)
        return None, False
    
    def system_skill_take_picture(self) -> tuple[None, bool]:
        img_path = os.path.join(self.cache_folder, f"{uuid.uuid4()}.jpg")
        Image.fromarray(self.latest_frame).save(img_path)
        print_t(f"[C] Picture saved to {img_path}")
        self._send_message((img_path,))
        return None, False
    
    def system_skill_re_plan(self) -> tuple[None, bool]:
        return None, True

    def system_skill_probe(self, query: str) -> tuple[Optional[str], bool]:
        pass

    def _send_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def start_controller(self):
        for (_, wrapper) in self.robots.items():
            wrapper.start()
        
    def stop_controller(self):
        for (_, wrapper) in self.robots.items():
            wrapper.stop()

        if os.path.exists(self.cache_folder):
            shutil.rmtree(self.cache_folder)
            print_t("[C] Cache folder cleared")

    def fetch_robot_observation(self, robot_info: RobotInfo, overlay: bool=False) -> Optional[Image.Image]:
        obs = self.robots[robot_info].observation
        if not obs or not obs.image_process_result:
            return None

        image, yolo_results = obs.image_process_result
        if overlay:
            YoloClient.plot_results_oi(image, yolo_results)

        return image
    
    def execute_minispec(self, minispec: str):
        interpreter = MiniSpecInterpreter(self.message_queue)
        interpreter.execute(minispec)
        self.execution_history = interpreter.execution_history
        ret_val = interpreter.ret_queue.get()
        return ret_val

    def handle_task(self, task_description: str):
        self._send_message('[TASK]: ' + task_description)
        ret_val = None
        while True:
            self.current_plan = self.planner.plan(task_description, execution_history=self.execution_history)
            self._send_message(f'[Plan]: \\\\')
            try:
                ret_val = self.execute_minispec(self.current_plan)
            except Exception as e:
                print_t(f"[C] Error: {e}")
            
            # disable replan for debugging
            break
            if ret_val.replan:
                print_t(f"[C] > Replanning <: {ret_val.value}")
                continue
            else:
                break
        self._send_message(f'\n[Task ended]')
        self._send_message('end')
        self.current_plan = None
        self.execution_history = None