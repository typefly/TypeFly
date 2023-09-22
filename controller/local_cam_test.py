import os, time, json
from threading import Thread, Lock
from PIL import Image
import numpy as np
import openai
import cv2

from yolo_client import YoloClient
openai.organization = "org-sAnQwPNnbSrHg1XyR4QYALf7"
openai.api_key = os.environ.get('OPENAI_API_KEY')

global controller_state
global yolo_results
global yolo_client
global yolo_results_lock
global commands_list
global image_width
global image_height

def is_in_sight(obj) -> bool:
    with yolo_results_lock:
        if yolo_results is None:
            return False
        for item in yolo_results:
            if item['label'].replace(' ', '_') == obj:
                return True
        return False

def is_not_in_sight(obj) -> bool:
    return not is_in_sight(obj)

def get_location_x(obj) -> float:
    global image_width
    with yolo_results_lock:
        for item in yolo_results:
            if item['label'].replace(' ', '_') == obj:
                return (item['x1'] + item['x2']) / 2 / image_width
    return -1

def get_location_y(obj) -> float:
    global image_height
    with yolo_results_lock:
        for item in yolo_results:
            if item['label'].replace(' ', '_') == obj:
                return (item['y1'] + item['y2']) / 2 / image_height
    return -1

def check_location_x(obj, compare, val) -> bool:
    float_val = float(val)
    if compare == '<':
        return get_location_x(obj) < float_val
    elif compare == '>':
        return get_location_x(obj) > float_val
    return False

def check_location_y(obj, compare, val) -> bool:
    float_val = float(val)
    if compare == '<':
        return get_location_y(obj) < float_val
    elif compare == '>':
        return get_location_y(obj) > float_val
    return False

low_level_tools = {
    "move_forward": "tello.forward(20)",
    "move_backward": "tello.backward(20)",
    "move_left": "tello.left(20)",
    "move_right": "tello.right(20)",
    "move_up": "tello.up(20)",
    "move_down": "tello.down(20)",
    "turn_left": "tello.rotate_ccw(20)",
    "turn_right": "tello.rotate_cw(20)",
    "is_in_sight": is_in_sight,
    "is_not_in_sight": is_not_in_sight,
    "check_location_x": check_location_x,
    "check_location_y": check_location_y,
}

high_level_tools = {
    "find": "loop#4 if#low,is_not_in_sight,$1#2 exec#low,turn_left,10 skip#1 break",
    "centering": "loop#7 if#low,check_location_x,$1,>,0.6#1 exec#low,turn_right,10 if#low,check_location_x,$1,<,0.4#1 exec#low,turn_left,10 \
        if#low,check_location_x,$1,<,0.6#2 if#low,check_location_x,$1,>,0.4#1 break",
}

commands_list = [
    "exec#high,find,cell_phone exec#high,centering,cell_phone",
]

def camera_capture():
    global yolo_results
    global controller_state
    global image_width
    global image_height
    cap = cv2.VideoCapture(1)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")
    else:
        # get image size
        image_width = int(cap.get(3))
        image_height = int(cap.get(4))
    # camera capture
    while True:
        ret, frame = cap.read()
        if ret == True:
            # convert cv2 mat to PIL image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with yolo_results_lock:
                yolo_results = yolo_client.detect(image)
            YoloClient.plot_results(image, yolo_results)
            print(yolo_results)

            cv2.imshow('Frame', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                controller_state = False
                break

def execute_tool_command(tool_command) -> bool:
    # parse tool command
    segments = tool_command.split(",")
    level = segments[0]
    tool_name = segments[1]
    if level == 'low':
        tool = low_level_tools.get(tool_name)
        if tool is not None:
            print(f'> > exec low-level tool: {tool}, {segments[2:]}')
            if callable(tool):
                return tool(*segments[2:])
            return True
    elif level == 'high':
        tool = high_level_tools.get(tool_name)
        if tool is not None:
            # replace all $1, $2, ... with segments
            for i in range(2, len(segments)):
                tool = tool.replace(f"${i - 1}", segments[i])
            print(f"> > expanded high-level tool: {tool}")
            return execute_commands(tool)
    return False

def execute_commands(commands) -> bool:
    global controller_state
    parts = commands.split()
    loop_range = (-1, 0)
    print (f"> parts: {parts}")
    loop_index = 0
    while loop_index < len(parts):
        # check controller state
        if not controller_state:
            break
        print(f"> loop_index: {loop_index}, part: {parts[loop_index]}")
        # parse command
        segments = parts[loop_index].split("#")
        # get command name
        match segments[0]:
            case 'if':
                if not execute_tool_command(segments[1]):
                    loop_index += int(segments[2])
            case 'loop':
                loop_range = (loop_index + 1, loop_index + int(segments[1]))
            case 'exec':
                execute_tool_command(segments[1])
            case 'break':
                loop_range = (-1, 0)
            case 'skip':
                loop_index += int(segments[1])

        loop_index += 1
        # check loop range
        if loop_range[0] != -1 and loop_index > loop_range[1]:
            loop_index = loop_range[0]
        time.sleep(1)

def control():
    global commands
    global controller_state
    while True:
        # check controller state
        if not controller_state:
            break
        # get commands
        if len(commands_list) > 0:
            commands = commands_list.pop(0)
            print(f"Execute commands: {commands}")
            execute_commands(commands)
            print(f"Done executing commands: {commands}")
        time.sleep(0.5)

def init():
    global controller_state
    global yolo_results
    global yolo_client
    global yolo_results_lock
    controller_state = True
    yolo_results = None
    yolo_client = YoloClient()
    yolo_results_lock = Lock()

if __name__ == "__main__":
    init()
    # control_thread = Thread(target=control)
    # control_thread.start()
    camera_capture()
    # control_thread.join()
