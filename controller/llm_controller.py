from threading import Thread
from PIL import Image
import queue, time, os, json, shutil
import asyncio
import uuid
import appdirs
from openai import Stream
from typing import Optional

from .shared_frame import SharedFrame, Frame
from .yolo_client import YoloClient
from .platforms.tello_wrapper import TelloWrapper
from .platforms.virtual_robot_wrapper import VirtualRobotWrapper
from .robot_wrapper import RobotWrapper
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

        self.planner = LLMPlanner()

        # cache folder
        self.cache_folder = CACHE_DIR
        os.makedirs(self.cache_folder, exist_ok=True)

        self.controller_func = [
            self.user_log,
            self.probe
        ]

        self.robots: dict[RobotInfo, RobotWrapper] = {}
        for info in robot_info_list:
            if info.robot_type == "virtual":
                self.robots[info] = VirtualRobotWrapper(info, self.controller_func)
            elif info.robot_type == "tello":
                self.robots[info] = TelloWrapper(info, self.controller_func)
            # elif info.robot_type == "go2":
            #     pass
        
        self.planner.set_robot_list(self.robots.values())

        self.current_plan = None
        self.execution_history = None

    def user_log(self, text: str | Image.Image) -> tuple[None, bool]:
        if isinstance(text, Image.Image):
            img_path = os.path.join(self.cache_folder, f"{uuid.uuid4()}.jpg")
            text.save(img_path)
            self._send_message((img_path,))
            print_t(f"[C] Picture saved to {img_path}")
        else:
            self._send_message(f"[LOG] {text}")
            print_t(f"[LOG] {text}")
        return True, False

    def probe(self, query: str) -> tuple[Optional[str], bool]:
        self.planner.probe(query), False

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
            YoloClient.plot_results_ps(image, yolo_results)

        return image
    
    def execute_minispec(self, json_output: Stream | str):
        interpreter = MiniSpecInterpreter(self.message_queue, self.robots)
        interpreter.execute(json_output)

    def handle_task(self, user_instruction: str):
        self._send_message('[TASK]: ' + user_instruction)
        ret_val = None
        while True:
            self.current_plan = self.planner.plan(user_instruction)
            self._send_message(f'[Plan]: {self.current_plan}')
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