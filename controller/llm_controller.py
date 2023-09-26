from PIL import Image
from threading import Thread
import numpy as np
import queue, time, json

from yolo_client import YoloClient
from tello_wrapper import TelloWrapper
from drone_wrapper import DroneWrapper
from vision_wrapper import VisionWrapper
from llm_planner import LLMPlanner
from toolset import ToolSet, LowLevelToolItem, HighLevelToolItem, ToolArg

class LLMController():
    def __init__(self):
        self.yolo_results_queue = queue.Queue(maxsize=1)
        self.yolo_client = YoloClient(self.yolo_results_queue)
        self.controller_state = True
        self.controller_wait_takeoff = True
        self.drone: DroneWrapper = TelloWrapper()
        self.vision = VisionWrapper(self.yolo_results_queue)
        self.frame_queue = queue.Queue(maxsize=1)

        self.low_level_toolset = ToolSet(level="low")

        self.low_level_toolset.add_tool(LowLevelToolItem("move_forward", self.drone.move_forward, "Move drone forward by a distance", args=[ToolArg("distance", int)]))
        self.low_level_toolset.add_tool(LowLevelToolItem("move_backward", self.drone.move_backward, "Move drone backward by a distance", args=[ToolArg("distance", int)]))
        self.low_level_toolset.add_tool(LowLevelToolItem("move_left", self.drone.move_left, "Move drone left by a distance", args=[ToolArg("distance", int)]))
        self.low_level_toolset.add_tool(LowLevelToolItem("move_right", self.drone.move_right, "Move drone right by a distance", args=[ToolArg("distance", int)]))
        self.low_level_toolset.add_tool(LowLevelToolItem("move_up", self.drone.move_up, "Move drone up by a distance", args=[ToolArg("distance", int)]))
        self.low_level_toolset.add_tool(LowLevelToolItem("move_down", self.drone.move_down, "Move drone down by a distance", args=[ToolArg("distance", int)]))
        self.low_level_toolset.add_tool(LowLevelToolItem("turn_ccw", self.drone.turn_ccw, "Turn drone counter clockwise by a degree", args=[ToolArg("degree", int)]))
        self.low_level_toolset.add_tool(LowLevelToolItem("turn_cw", self.drone.turn_cw, "Turn drone clockwise by a degree", args=[ToolArg("degree", int)]))
        self.low_level_toolset.add_tool(LowLevelToolItem("is_in_sight", self.vision.is_in_sight, "Check if an object is in sight", args=[ToolArg("object_name", str)]))
        self.low_level_toolset.add_tool(LowLevelToolItem("is_not_in_sight", self.vision.is_not_in_sight, "Check if an object is not in sight", args=[ToolArg("object_name", str)]))
        self.low_level_toolset.add_tool(LowLevelToolItem("check_location_x", self.vision.check_location_x, \
                                                         "Check if x location of an object meets the comparison criterion with the value", \
                                                         args=[ToolArg("object_name", str), ToolArg("compare", str), ToolArg("val", float)]))
        self.low_level_toolset.add_tool(LowLevelToolItem("check_location_y", self.vision.check_location_y, \
                                                         "Check if y location of an object meets the comparison criterion with the value", \
                                                         args=[ToolArg("object_name", str), ToolArg("compare", str), ToolArg("val", float)]))
        
        self.high_level_toolset = ToolSet(level="high", lower_level_toolset=self.low_level_toolset)
        self.high_level_toolset.add_tool(HighLevelToolItem("scan", ["loop#5", "if#low,is_not_in_sight,$1#3", "exec#low,turn_ccw,30", "delay#300", "skip#1", "break"],
                                                           "scan the environment to find the object"))
        self.high_level_toolset.add_tool(HighLevelToolItem("align",
                                                           ["loop#7", "if#low,check_location_x,$1,>,0.6#1", "exec#low,turn_cw,15",
                                                            "if#low,check_location_x,$1,<,0.4#1", "exec#low,turn_ccw,15",
                                                            "if#low,check_location_x,$1,<,0.6#2", "if#low,check_location_x,$1,>,0.4#1", "break"],
                                                            "align the object to the center of the frame by rotating the drone"))
        self.high_level_toolset.add_tool(HighLevelToolItem("centering",
                                                           ["loop#13", "if#low,check_location_x,$1,>,0.6#1", "exec#low,move_right,20",
                                                            "if#low,check_location_x,$1,<,0.4#1", "exec#low,move_left,20",
                                                            "if#low,check_location_y,$1,<,0.4#2", "exec#low,move_up,20",
                                                            "if#low,check_location_y,$1,>,0.6#2", "exec#low,move_down,20",
                                                            "if#low,check_location_x,$1,<,0.6#4", "if#low,check_location_x,$1,>,0.4#3",
                                                            "if#low,check_location_y,$1,>,0.4#2", "if#low,check_location_y,$1,<,0.6#1", "break"],
                                                            "center the object in the frame by moving the drone"))

        self.planner = LLMPlanner(high_level_toolset=self.high_level_toolset, low_level_toolset=self.low_level_toolset)

    def stop_controller(self):
        self.controller_state = False

    def get_latest_frame(self):
        if not self.frame_queue.empty():
            self.frame_queue.get()
        return self.frame_queue.get()

    def execute_tool_command(self, tool_command) -> bool:
        # parse tool command
        segments = tool_command.split(",")
        level = segments[0]
        tool_name = segments[1]
        if level == 'low':
            tool = self.low_level_toolset.get_tool(tool_name)
            return tool.execute(segments[2:])
        elif level == 'high':
            tool = self.high_level_toolset.get_tool(tool_name)
            return self.execute_commands(tool.execute(segments[2:]))
        return False

    def execute_commands(self, commands: [str]) -> bool:
        loop_range = (-1, 0)
        loop_index = 0
        while loop_index < len(commands):
            # check controller state
            if not self.controller_state:
                break
            print(f"> loop_index: {loop_index}, part: {commands[loop_index]}")
            # parse command
            segments = commands[loop_index].split("#")
            # get command name
            execution_result = False
            match segments[0]:
                case 'if':
                    execution_result = self.execute_tool_command(segments[1])
                    if not execution_result:
                        loop_index += int(segments[2])
                case 'loop':
                    loop_range = (loop_index + 1, loop_index + int(segments[1]))
                case 'exec':
                    execution_result = self.execute_tool_command(segments[1])
                case 'break':
                    loop_range = (-1, 0)
                case 'skip':
                    loop_index += int(segments[1])
                case 'delay':
                    time.sleep(int(segments[1]) / 1000.0)
                case 'str':
                    print('Answer: ' + segments[1])
                case 'redo':
                    return True

            if execution_result is None:
                # if error occurs, break
                loop_range = (-1, 0)

            loop_index += 1
            # check loop range
            if loop_range[0] != -1 and loop_index > loop_range[1]:
                loop_index = loop_range[0]
        return False

    def execute_user_command(self, user_command: str):
        if self.controller_wait_takeoff:
            print("Controller is waiting for takeoff...")
            return
        while self.controller_state:
            result = self.planner.request(self.vision.get_obj_list(), user_command)
            input_ans = input(f">> result: {json.dumps(result)}, executing?")
            if input_ans == 'n':
                break
            if not self.execute_commands(result):
                break

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
    controller.run()

if __name__ == "__main__":
    main()