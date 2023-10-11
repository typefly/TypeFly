from PIL import Image
from threading import Thread
import numpy as np
import queue, time, json
from typing import Optional

from yolo_client import YoloClient
from tello_wrapper import TelloWrapper
from virtual_drone_wrapper import VirtualDroneWrapper
from drone_wrapper import DroneWrapper
from vision_wrapper import VisionWrapper
from llm_wrapper import LLMWrapper
from llm_planner import LLMPlanner
from skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg

class LLMController():
    def __init__(self):
        self.yolo_results_queue = queue.Queue(maxsize=1)
        self.yolo_client = YoloClient(self.yolo_results_queue)
        self.controller_state = True
        self.controller_wait_takeoff = True
        self.llm = LLMWrapper()
        self.drone: DroneWrapper = VirtualDroneWrapper()
        # self.drone: DroneWrapper = TelloWrapper()
        self.vision = VisionWrapper(self.yolo_results_queue, self.llm)
        self.frame_queue = queue.Queue(maxsize=1)

        self.low_level_skillset = SkillSet(level="low")

        self.low_level_skillset.add_skill(LowLevelSkillItem("move_forward", self.drone.move_forward, "Move drone forward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_backward", self.drone.move_backward, "Move drone backward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_left", self.drone.move_left, "Move drone left by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_right", self.drone.move_right, "Move drone right by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_up", self.drone.move_up, "Move drone up by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_down", self.drone.move_down, "Move drone down by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_ccw", self.drone.turn_ccw, "Turn drone counter clockwise by a degree", args=[SkillArg("degree", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_cw", self.drone.turn_cw, "Turn drone clockwise by a degree", args=[SkillArg("degree", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("is_in_sight", self.vision.is_in_sight, "Check if an object is in sight", args=[SkillArg("obj_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("obj_loc_x", self.vision.obj_loc_x, "Get x location of an object", args=[SkillArg("obj_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("obj_loc_y", self.vision.obj_loc_y, "Get y location of an object", args=[SkillArg("obj_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("query", self.vision.query, "Query the LLM to check the environment state", args=[SkillArg("question", str)]))
        
        self.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)
        self.high_level_skillset.add_skill(HighLevelSkillItem("scan", ["loop#8#4", "if#is_in_sight,$1,=,True#1", "return#true", "exec#turn_ccw,45", "delay#300", "return#false"],
                                                           "rotate to find a certain object"))
        self.high_level_skillset.add_skill(HighLevelSkillItem("orienting",
                                                           ["loop#4#7",
                                                            "if#obj_loc_x,$1,>,0.6#1", "exec#turn_cw,15",
                                                            "if#obj_loc_x,$1,<,0.4#1", "exec#turn_ccw,15",
                                                            "if#obj_loc_x,$1,<,0.6#2", "if#obj_loc_x,$1,>,0.4#1", "return#true", "return#false"],
                                                            "align the object to the center of the frame by rotating the drone"))
        self.high_level_skillset.add_skill(HighLevelSkillItem("centering_y",
                                                           ["loop#4#7",
                                                            "if#obj_loc_y,$1,<,0.4#1", "exec#move_up,20",
                                                            "if#obj_loc_y,$1,>,0.6#1", "exec#move_down,20",
                                                            "if#obj_loc_y,$1,>,0.4#2", "if#obj_loc_y,$1,<,0.6#1", "return#true", "return#false"],
                                                            "center the object's y location in the frame by moving the drone up or down"))
        self.high_level_skillset.add_skill(HighLevelSkillItem("find_edible_obj",
                                                           ["loop#8#3", "exec#turn_ccw,45", "if#query,'is there anything edible?',=,True#1", "return#true", "return#false"],
                                                            "find an edible object"))

        self.planner = LLMPlanner(llm=self.llm, high_level_skillset=self.high_level_skillset, low_level_skillset=self.low_level_skillset)

    def stop_controller(self):
        self.controller_state = False

    def get_latest_frame(self):
        if not self.frame_queue.empty():
            self.frame_queue.get()
        return self.frame_queue.get()

    def execute_skill_command(self, segments) -> Optional[bool]:
        # skill_command: skill_name,kwargs
        print(f">> executing skill command: {segments}")
        skill_name = segments[0]

        skill_instance = self.low_level_skillset.get_skill(skill_name)
        if skill_instance is not None:
            return skill_instance.execute(segments[1:])

        skill_instance = self.high_level_skillset.get_skill(skill_name)
        if skill_instance is not None:
            return self.execute_commands(skill_instance.execute(segments[1:]))
        
        print(f"Skill '{skill_name}' not found.")
        return None

    def execute_commands(self, commands: [str]) -> bool:
        def parse_value(s):
            # Check for boolean values
            if s.lower() == "true":
                return True
            elif s.lower() == "false":
                return False
            # Try to parse as a float
            try:
                return float(s)
            except ValueError:
                raise ValueError(f"String '{s}' cannot be parsed as float or bool")

        loop_index = 0
        loop_count = 0
        loop_range = (-1, 0)
        while loop_index < len(commands):
            print(f">> executing: {loop_index} {commands[loop_index]}")
            # check controller state
            if not self.controller_state:
                break
            # parse command
            segments = commands[loop_index].split("#")
            # get command name
            execution_result = False
            match segments[0]:
                case 'exec':
                    execution_result = self.execute_skill_command(segments[1].split(','))
                case 'if':
                    params = segments[1].split(',')
                    compare = params[-2]
                    val = params[-1]
                    execution_result = self.execute_skill_command(params[0:-2])
                    print(f"Execution result: {params[0:-2]} {execution_result}")
                    condition = False
                    if compare == '=':
                        condition = execution_result == parse_value(val)
                        print(f"Comparing {execution_result} and {val} if =. {condition}")
                    elif compare == '<':
                        condition = execution_result < parse_value(val)
                        print(f"Comparing {execution_result} and {val} if <. {condition}")
                    elif compare == '>':
                        condition = execution_result > parse_value(val)
                        print(f"Comparing {execution_result} and {val} if >. {condition}")
                    else:
                        raise ValueError(f"Unrecognized comparison operator: {compare}")

                    if not condition:
                        print("Condition not met.")
                        loop_index += int(segments[-1])
                case 'loop':
                    loop_count = int(segments[1])
                    loop_range = (loop_index + 1, loop_index + int(segments[-1]))
                case 'break':
                    loop_count = 0
                case 'skip':
                    loop_index += int(segments[-1])
                case 'delay':
                    time.sleep(int(segments[1]) / 1000.0)
                case 'str':
                    if segments[1].startswith("'") and segments[1].endswith("'"):
                        print('Response: ' + segments[1])
                    else:
                        result = self.execute_skill_command(segments[1].split(','))
                        print(f'Response: {result}')
                case 'return':
                    return parse_value(segments[1])

            if execution_result is None:
                # if error occurs, break
                return False

            loop_index += 1
            # if in a loop
            if loop_range[0] != -1:
                # if loop_count is not 0 and one loop has reach the end, decrease loop_count
                if loop_count > 0 and loop_index > loop_range[1]:
                    loop_count -= 1
                
                # if loop_count is 0, reset loop_index and loop_range
                if loop_count == 0:
                    loop_index = loop_range[1] + 1
                    loop_range = (-1, 0)
                # if loop_count is not 0 and one loop has reach the end, reset loop_index
                elif loop_index > loop_range[1]:
                    loop_index = loop_range[0]
        return True

    def execute_user_command(self, user_command: str):
        if self.controller_wait_takeoff:
            print("Controller is waiting for takeoff...")
            return
        
        for _ in range(1):
            result = self.planner.request_task(self.vision.get_obj_list(), user_command)
            # json_result = json.dumps(result)
            print(f">> result: {json.dumps(result)}, executing...")
            # consent = input(f">> result: {json.dumps(result)}, executing?")
            # if consent == 'n':
            #     print(">> command rejected.")
            #     return
            self.execute_commands(result)
            # ending = self.planner.request_ending(self.vision.get_obj_list(), result)
            # print(f">> ending: {ending['feedback']}")
            # if ending['result'] == 'True' or ending['result'] == True:
            #     print(">> command executed successfully.")
            #     return
            # else:
            #     print(">> command failed, try again.")

    def run(self):
        self.drone.connect()
        print("Drone is taking off...")
        self.drone.takeoff()
        self.drone.move_up(30)
        self.drone.start_stream()
        self.controller_wait_takeoff = False
        print("Start controller...")

        while self.controller_state:
            self.drone.keep_active()
            frame = self.drone.get_image()
            image = Image.fromarray(frame)
            results = self.yolo_client.detect(image)
            YoloClient.plot_results(image, results)
            self.frame_queue.put(image)
        self.drone.land()
        self.drone.stop_stream()

def main():
    controller = LLMController()
    # controller.run()
    controller.execute_commands(["exec#orienting,person"])

if __name__ == "__main__":
    main()