from PIL import Image
import queue, io, base64
from openai import Stream
from typing import Optional

from .yolo_client import YoloClient
from .platforms.virtual_robot_wrapper import VirtualRobotWrapper
from .robot_wrapper import RobotWrapper
from .llm_planner import LLMPlanner
from .utils import print_t
from .minispec_interpreter import MiniSpecInterpreter
from .robot_info import RobotInfo

class LLMController():
    def __init__(self, robot_info_list: list[RobotInfo], message_queue: Optional[queue.Queue]=None):
        self.message_queue = message_queue

        self.planner = LLMPlanner()

        self.controller_func = [
            self.user_log,
            self.probe
        ]

        self.robots: dict[RobotInfo, RobotWrapper] = {}
        for info in robot_info_list:
            if info.robot_type == "virtual":
                self.robots[info] = VirtualRobotWrapper(info, self.controller_func)
            elif info.robot_type == "tello":
                from .platforms.tello_wrapper import TelloWrapper
                self.robots[info] = TelloWrapper(info, self.controller_func)
            elif info.robot_type == "go2":
                from .platforms.go2_wrapper import Go2Wrapper
                self.robots[info] = Go2Wrapper(info, self.controller_func)
            elif info.robot_type == "pod":
                from .platforms.pod_wrapper import PodWrapper
                self.robots[info] = PodWrapper(info, self.controller_func)
        
        self.planner.set_robot_dict(self.robots)
        self.current_plan = None
        self.execution_history = None

    def user_log(self, content: str | Image.Image) -> tuple[None, bool]:
        if isinstance(content, Image.Image):
            buffer = io.BytesIO()
            content.save(buffer, format="JPEG")
            self._send_message(f'<img src="data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode("utf-8")}" />')
        else:
            text = content.strip('\'')
            self._send_message(f'[LOG] {text}')
            print_t(f'[LOG] {text}')
        return True, False

    def probe(self, query: str, robot_info: RobotInfo) -> str:
        return self.planner.probe(query, robot_info)

    def _send_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def start_controller(self):
        for (_, wrapper) in self.robots.items():
            wrapper.start()
        
    def stop_controller(self):
        for (_, wrapper) in self.robots.items():
            wrapper.stop()

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
        # self._send_message('[TASK]: ' + user_instruction)
        self._send_message('Planning...')
        ret_val = None
        while True:
            self.current_plan = self.planner.plan(user_instruction)
            # self._send_message(f'[Plan]: {self.current_plan}')
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