from PIL import Image
import queue, time
from typing import Optional
import asyncio

from yolo_client import YoloClient, SharedYoloResults
from yolo_grpc_client import YoloGRPCClient
from tello_wrapper import TelloWrapper
from virtual_drone_wrapper import VirtualDroneWrapper
from drone_wrapper import DroneWrapper
from vision_wrapper import VisionWrapper
from llm_wrapper import LLMWrapper
from llm_planner import LLMPlanner
from skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg

class LLMController():
    def __init__(self):
        self.yolo_results_image_queue = queue.Queue(maxsize=30)
        self.yolo_results = SharedYoloResults()
        self.yolo_client = YoloGRPCClient(is_local_service=False, shared_yolo_results=self.yolo_results)
        self.controller_state = True
        self.controller_wait_takeoff = True
        self.drone: DroneWrapper = VirtualDroneWrapper()
        # self.drone: DroneWrapper = TelloWrapper()
        self.vision = VisionWrapper(self.yolo_results)
        self.planner = LLMPlanner()

        self.low_level_skillset = SkillSet(level="low")

        self.low_level_skillset.add_skill(LowLevelSkillItem("move_forward", self.drone.move_forward, "Move drone forward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_backward", self.drone.move_backward, "Move drone backward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_left", self.drone.move_left, "Move drone left by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_right", self.drone.move_right, "Move drone right by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_up", self.drone.move_up, "Move drone up by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_down", self.drone.move_down, "Move drone down by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_ccw", self.drone.turn_ccw, "Turn drone counterclockwise by a degree", args=[SkillArg("degree", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_cw", self.drone.turn_cw, "Turn drone clockwise by a degree", args=[SkillArg("degree", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("is_in_sight", self.vision.is_in_sight, "Check if an object is in sight", args=[SkillArg("obj_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("obj_loc_x", self.vision.obj_loc_x, "Get x location of an object", args=[SkillArg("obj_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("obj_loc_y", self.vision.obj_loc_y, "Get y location of an object", args=[SkillArg("obj_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("query", self.planner.request_execution, "Query the LLM to check the environment state", args=[SkillArg("question", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("verification", self.verification, "Restart plan if task description has not been met", args=[SkillArg("retry", int)]))
        
        self.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)
        self.high_level_skillset.add_skill(HighLevelSkillItem("scan", ["loop#8#4", "if#is_in_sight,$1,=,true#1", "ret#true", "exec#turn_cw,45", "delay#300", "ret#false"],
                                                           "rotate to find a certain object"))
        self.high_level_skillset.add_skill(HighLevelSkillItem("approach", ["exec#move_forward,60"], comment="approach the object"))
        self.high_level_skillset.add_skill(HighLevelSkillItem("orienting",
                                                           ["loop#4#7",
                                                            "if#obj_loc_x,$1,>,0.6#1", "exec#turn_cw,15",
                                                            "if#obj_loc_x,$1,<,0.4#1", "exec#turn_ccw,15",
                                                            "if#obj_loc_x,$1,<,0.6#2", "if#obj_loc_x,$1,>,0.4#1", "ret#true", "ret#false"],
                                                            "align the object to the center of the frame by rotating the drone"))
        self.high_level_skillset.add_skill(HighLevelSkillItem("centering_y",
                                                           ["loop#4#7",
                                                            "if#obj_loc_y,$1,<,0.4#1", "exec#move_up,20",
                                                            "if#obj_loc_y,$1,>,0.6#1", "exec#move_down,20",
                                                            "if#obj_loc_y,$1,>,0.4#2", "if#obj_loc_y,$1,<,0.6#1", "ret#true", "ret#false"],
                                                            "center the object's y location in the frame by moving the drone up or down"))
        # self.high_level_skillset.add_skill(HighLevelSkillItem("find_drink_obj",
        #                                                    ["loop#8#3", "exec#turn_cw,45", "if#query,'is there anything for drink?',=,true#1", "ret#true", "ret#false"],
        #                                                     "find a drinkable object"))

        self.planner.init(high_level_skillset=self.high_level_skillset, low_level_skillset=self.low_level_skillset, vision_skill=self.vision)

    def verification(self, retry: int):
        pass

    def stop_controller(self):
        self.controller_state = False

    def get_latest_frame(self):
        return self.yolo_results_image_queue.get()

    def execute_skill_command(self, segments) -> Optional[bool]:
        # skill_command: skill_name,kwargs
        print(f">> executing skill command: {segments}")
        skill_name = segments[0]
        kwargs = segments[1:]
        # replace $? with self.last_execution_result
        for index, arg in enumerate(kwargs):
            if arg == '$?':
                kwargs[index] = str(self.last_execution_result)

        skill_instance = self.low_level_skillset.get_skill(skill_name)
        if skill_instance is not None:
            return skill_instance.execute(kwargs)

        skill_instance = self.high_level_skillset.get_skill(skill_name)
        if skill_instance is not None:
            return self.execute_commands(skill_instance.execute(kwargs))
        
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
            # print(f">> executing: {loop_index} {commands[loop_index]}")
            # check controller state
            if not self.controller_state:
                break
            # parse command
            segments = commands[loop_index].split("#")
            # get command name
            match segments[0]:
                case 'exec':
                    self.last_execution_result = self.execute_skill_command(segments[1].split(','))
                case 'if':
                    params = segments[1].split(',')
                    compare = params[-2]
                    val = params[-1]
                    local_execution_result = self.execute_skill_command(params[0:-2])
                    # print(f"Execution result: {params[0:-2]} {execution_result}")
                    condition = False
                    if compare == '=':
                        condition = local_execution_result == parse_value(val)
                        # print(f"Comparing {local_execution_result} and {val} if =. {condition}")
                    elif compare == '<':
                        condition = local_execution_result < parse_value(val)
                        # print(f"Comparing {local_execution_result} and {val} if <. {condition}")
                    elif compare == '>':
                        condition = local_execution_result > parse_value(val)
                        # print(f"Comparing {local_execution_result} and {val} if >. {condition}")
                    else:
                        raise ValueError(f"Unrecognized comparison operator: {compare}")

                    if not condition:
                        # print("Condition not met.")
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
                case 'print':
                    if segments[1].startswith("'") and segments[1].endswith("'"):
                        print('Response: ' + segments[1])
                    elif segments[1] == '$?':
                        print('Response: ' + str(self.last_execution_result))

                case 'ret':
                    return parse_value(segments[1])

            if self.last_execution_result is None:
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

    def execute_user_command(self, task_description: str):
        # if self.controller_wait_takeoff:
        #     print("Controller is waiting for takeoff...")
        #     return
        
        for _ in range(1):
            t1 = time.time()
            result = self.planner.request_planning(task_description)
            t2 = time.time()
            print(f">> planning time: {t2 - t1}")
            # ["loop#8#4", "if#query,'is there anything edible?',=,true#2", "exec#approach", "ret#true", "exec#turn_cw,45",
            #          "loop#8#4", "if#query,'is there anything drinkable?',=,true#2", "exec#approach", "ret#true", "exec#turn_cw,45",
            #          "print#'no edible and drinkable item can be found'", "ret#false"]
            # print(f">> result: {result}, executing...")
            consent = input(f">> result: {result}, executing?")
            if consent == 'n':
                print(">> command rejected.")
                return
            self.execute_commands(result)
            # ending = self.planner.request_verification(task_description, result)
            # print(f">> ending: {ending['feedback']}")
            # if ending['result'] == 'True' or ending['result'] == True:
            #     print(">> command executed successfully.")
            #     return
            # else:
            #     print(">> command failed, try again.")

    def start_robot(self):
        print("Drone is taking off...")
        self.drone.connect()
        self.drone.takeoff()
        self.drone.move_up(30)
        self.drone.start_stream()
        self.controller_wait_takeoff = False
        print("Start controller...")

    def stop_robot(self):
        print("Drone is landing...")
        self.drone.land()
        self.drone.stop_stream()
        self.controller_wait_takeoff = True
        print("Stop controller...")

    def capture_loop(self, asyncio_loop):
        while self.controller_state:
            self.drone.keep_active()
            frame = self.drone.get_image()
            image = Image.fromarray(frame)

            if self.yolo_client.local_service():
                self.yolo_client.detect_local(image)
            else:
                # asynchronously send image to yolo server
                asyncio_loop.call_soon_threadsafe(asyncio.create_task, self.yolo_client.detect(image))

            latest_result = self.yolo_client.retrieve()
            if latest_result is not None:
                YoloClient.plot_results(latest_result[0], latest_result[1])
                self.yolo_results_image_queue.put(latest_result[0])

            time.sleep(0.050)

def main():
    controller = LLMController()
    # controller.run()
    with open("./assets/test.txt", "r") as f:
        text = f.read()
    controller.execute_user_command(f"please generate the following plan: {text}")

if __name__ == "__main__":
    main()