from threading import Lock
from PIL import Image
from threading import Thread
import numpy as np
import openai
import os, sys, json
import queue, time
import cv2

from yolo_client import YoloClient
from controller.tello_wrapper import TelloWrapper
from controller.drone_wrapper import DroneToolset
openai.organization = "org-sAnQwPNnbSrHg1XyR4QYALf7"
openai.api_key = os.environ.get('OPENAI_API_KEY')
MODEL_NAME = "gpt-3.5-turbo-16k"

class LLMController():
    def __init__(self):
        self.yolo_results_queue = queue.Queue(maxsize=1)
        self.yolo_client = YoloClient(self.yolo_results_queue)
        self.controller_state = True
        self.controller_wait_takeoff = True
        self.drone = TelloWrapper()
        self.toolset = DroneToolset(self.yolo_results_queue, self.drone)
        self.low_level_tools = {
            "move_forward": self.toolset.move_forward,
            "move_backward": self.toolset.move_backward,
            "move_left": self.toolset.move_left,
            "move_right": self.toolset.move_right,
            "move_up": self.toolset.move_up,
            "move_down": self.toolset.move_down,
            "turn_left": self.toolset.turn_left,
            "turn_right": self.toolset.turn_right,
            "is_in_sight": self.toolset.is_in_sight,
            "is_not_in_sight": self.toolset.is_not_in_sight,
            "check_location_x": self.toolset.check_location_x,
            "check_location_y": self.toolset.check_location_y,
        }

        self.high_level_tools = {
            "find": "loop#4 if#low,is_not_in_sight,$1#2 exec#low,turn_left,10 skip#1 break",
            "centering": "loop#7 if#low,check_location_x,$1,>,0.6#1 exec#low,turn_right,10 \
                if#low,check_location_x,$1,<,0.4#1 exec#low,turn_left,10 \
                if#low,check_location_x,$1,<,0.6#2 if#low,check_location_x,$1,>,0.4#1 break",
        }

        self.commands_list = [
            "exec#high,find,person exec#high,centering,person",
        ]

        self.prompt = "You are a drone pilot. You are provided with a toolset that allows \
            you to move the drone or get visual information. The toolset contains both \
            high-level and low-level tools, please use high-level tools as much as possible. \
            Here is an example: input: \"Find an apple\", output: \"exec#high,find,apple exec#high,centering,apple\""

        self.conversation = []

    def execute_tool_command(self, tool_command) -> bool:
        # parse tool command
        segments = tool_command.split(",")
        level = segments[0]
        tool_name = segments[1]
        if level == 'low':
            tool = self.low_level_tools.get(tool_name)
            if tool is not None:
                print(f'> > exec low-level tool: {tool}, {segments[2:]}')
                if callable(tool):
                    return tool(*segments[2:])
                return True
        elif level == 'high':
            tool = self.high_level_tools.get(tool_name)
            if tool is not None:
                # replace all $1, $2, ... with segments
                for i in range(2, len(segments)):
                    tool = tool.replace(f"${i - 1}", segments[i])
                print(f"> > expanded high-level tool: {tool}")
                return self.execute_commands(tool)
        return False
    
    def execute_commands(self, commands) -> bool:
        parts = commands.split()
        loop_range = (-1, 0)
        print (f"> parts: {parts}")
        loop_index = 0
        while loop_index < len(parts):
            # check controller state
            if not self.controller_state:
                break
            print(f"> loop_index: {loop_index}, part: {parts[loop_index]}")
            # parse command
            segments = parts[loop_index].split("#")
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

            if execution_result is None:
                # if error occurs, break
                loop_range = (-1, 0)

            loop_index += 1
            # check loop range
            if loop_range[0] != -1 and loop_index > loop_range[1]:
                loop_index = loop_range[0]

    def control_thread(self):
        while self.controller_wait_takeoff:
            time.sleep(1)
        print("=> Start control thread")
        while True:
            # check controller state
            if not self.controller_state:
                break
            # get commands
            if len(self.commands_list) > 0:
                commands = self.commands_list.pop(0)
                print(f"Execute commands: {commands}")
                self.execute_commands(commands)
                print(f"Done executing commands: {commands}")
            time.sleep(0.5)

    def run(self):
        self.drone.connect()
        self.drone.start()
        self.controller_wait_takeoff = False
        control_thread = Thread(target=self.control_thread)
        control_thread.start()

        while self.controller_state:
            self.drone.keep_alive()
            frame = self.drone.get_image()
            image = Image.fromarray(frame)
            results = self.yolo_client.detect(image)
            YoloClient.plot_results(image, results)
            cv2.imshow("Tello", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(10) & 0xff
            # Press esc to exit
            if key == 27:
                break
        self.controller_state = False
        self.drone.stop()
        control_thread.join()

def main():
    controller = LLMController()
    controller.run()

if __name__ == "__main__":
    main()