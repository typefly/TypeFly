from PIL import Image
import queue, io, base64
from typing import Optional
import threading
import json
import builtins

from .yolo_client import YoloClient
from .robot_wrapper import RobotWrapper
from .llm_planner import LLMPlanner
from .utils import print_t
from .robot_info import RobotInfo

_USER_LOG_QUEUE = queue.Queue()

class LLMController():
    def __init__(self, robot_info: RobotInfo):
        self.controller_func = [
            self._user_log,
            self._probe
        ]
        RobotWrapper.set_controller_func(self.controller_func)

        if robot_info.robot_type == "virtual":
            from .platforms.virtual_robot_wrapper import VirtualRobotWrapper
            self.robot = VirtualRobotWrapper(robot_info)
        elif robot_info.robot_type == "tello":
            from .platforms.tello_wrapper import TelloWrapper
            self.robot = TelloWrapper(robot_info)
        elif robot_info.robot_type == "go2":
            from .platforms.go2_wrapper import Go2Wrapper
            self.robot = Go2Wrapper(robot_info)
        elif robot_info.robot_type == "pod":
            from .platforms.pod_wrapper import PodWrapper
            self.robot = PodWrapper(robot_info)
        elif robot_info.robot_type == "tello_sim":
            from .platforms.tello_wrapper_janah import TelloSimWrapper
            self.robot = TelloSimWrapper(robot_info)
        elif robot_info.robot_type == "airsim":
            from .platforms.airsim_platform import AirSimDronePlatform
            self.robot = AirSimDronePlatform(robot_info)
            
        self.planner = LLMPlanner(self.robot)
        self.current_plan_loop_thread = None
        # Missing child information for SAR operations
        self.missing_child_info = {
            'name': '',
            'age': '',
            'clothing_color': '',
            'last_location': '',
            'description': '',
            'photo_path': ''
        }

    def _user_log(self, msg: str | Image.Image) -> bool:
        if isinstance(msg, Image.Image):
            buffer = io.BytesIO()
            msg.save(buffer, format="JPEG")
            encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
            _USER_LOG_QUEUE.put(f'<img src="data:image/jpeg;base64,{encoded_img}" />')
        else:
            text = str(msg).strip('\'')
            _USER_LOG_QUEUE.put(f'[ROBOT] {text}')
            print_t(f'[ROBOT] {text}')
        return True

    def _probe(self, query: str, robot_info: RobotInfo) -> str:
        return self.planner.probe(query, robot_info)

    def start_controller(self):
        self.robot.start()
        
    def stop_controller(self):
        self.robot.stop()

    def fetch_robot_pov(self, overlay: bool=True) -> Optional[Image.Image]:
        image = self.robot.obs.image
        yolo_results = self.robot.obs.image_process_result.get("yolo", [])
        if overlay:
            YoloClient.plot_results_ps(image, yolo_results)
        return image

    def plan_loop(self, user_instruction: str):
        while True:
            plan = self.planner.plan(user_instruction, missing_child_info=self.missing_child_info)
            print_t(f"[P] Plan: {plan}")

            if '```json' in plan:
                plan = plan.split('```json')[1].split('```')[0]
            
            # parse the plan json
            try:
                plan = json.loads(plan)
            except json.JSONDecodeError as e:
                print_t(f"[P] Invalid json: {e}")
                continue

            program_str = plan['plan']

            break

        # Execute the program string, routing robot skill calls through skillset
        # Create a namespace that intercepts skill function calls
        class SkillNamespace(dict):
            def __init__(self, robot, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.robot = robot
                # Pre-populate with available skills
                for skill_name in robot.skillset.skills.keys():
                    self[skill_name] = robot.skillset.get_skill(skill_name)
                # Include builtins
                self['__builtins__'] = builtins
            
            def __getitem__(self, key):
                # If key is a skill, return it from skillset
                if key in self.robot.skillset.skills:
                    return self.robot.skillset.get_skill(key)
                # Otherwise, try to get from dict
                if key in self:
                    return super().__getitem__(key)
                # Check builtins if not found in dict
                if hasattr(builtins, key):
                    return getattr(builtins, key)
                # If not found, raise NameError (standard Python behavior)
                raise NameError(f"name '{key}' is not defined")
        
        # Create execution namespace
        exec_namespace = SkillNamespace(self.robot)
        
        try:
            exec(program_str, exec_namespace)
        except Exception as e:
            print_t(f"[P] Error executing plan: {e}")
            import traceback
            print_t(f"[P] Traceback: {traceback.format_exc()}")

        # End the plan loop
        _USER_LOG_QUEUE.put('#end')

    def put_instruction(self, user_instruction: str):
        self.current_plan_loop_thread = threading.Thread(target=self.plan_loop, args=(user_instruction,), daemon=True)
        self.current_plan_loop_thread.start()

    def set_missing_child_info(self, info: dict):
        """Set missing child information for SAR mission"""
        self.missing_child_info.update(info)
        print_t(f"[SAR] Child: {info.get('name')}, Age: {info.get('age')}, Clothing: {info.get('clothing_color')}")
