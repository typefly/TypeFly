import queue
import sys, os
import asyncio
import io, time, json
from typing import List
import gradio as gr
from flask import Flask, Response
from threading import Thread

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJ_DIR)
from controller.llm_controller import LLMController
from controller.utils import print_t
from controller.robot_info import RobotInfo

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_drone_povs(robot_info_list: List[RobotInfo]):
    """Generate HTML string for multiple drone POVs."""
    num = len(robot_info_list)
    html_content = "<h2>Robot POVs</h2><div style='display: flex; gap: 10px;'>"
    for robot in robot_info_list:
        html_content += f"""
        <div>
            <h3>{robot.robot_type}</h3>
            <img src="http://localhost:50000/robot-pov/{robot.robot_id}/" alt="{robot.robot_id}-pov" 
                 style="border-radius: 10px; object-fit: contain; width: {1/num};">
        </div>
        """
    html_content += "</div>"
    return html_content

class TypeFly:
    def __init__(self, robot_info_list: List[RobotInfo]):
        self.robot_info_list = robot_info_list
        self.message_queue = queue.Queue()
        self.llm_controller = LLMController(robot_info_list, self.message_queue)
        self.system_stop = False

        self.ui = gr.Blocks(title="TypeFly")
        self.setup_ui(robot_info_list)

    def setup_ui(self, robot_info_list: List[RobotInfo]):
        """Sets up the Gradio UI components."""
        default_sentences = [
            "Find something I can eat.",
            "What you can see?",
            "Follow that ball for 20 seconds",
            "Find a chair for me.",
            "Go to the chair without book.",
        ]
        with self.ui:
            gr.HTML(open(os.path.join(CURRENT_DIR, 'header.html'), 'r').read())
            gr.HTML(generate_drone_povs(robot_info_list))
            gr.ChatInterface(self.ui_process_message, retry_btn=None, fill_height=False, examples=default_sentences).queue()

    def ui_process_message(self, message, history):
        print_t(f"[S] Receiving task description: {message}")
        if message == "exit":
            self.llm_controller.stop_controller()
            self.system_stop = True
            yield "Shutting down..."
        elif len(message) == 0:
            return "[WARNING] Empty command!]"
        else:
            task_thread = Thread(target=self.llm_controller.handle_task, args=(message,))
            task_thread.start()
            complete_response = ''
            while True:
                msg = self.message_queue.get()
                if isinstance(msg, tuple): # (image,)
                    history.append((None, msg))
                elif isinstance(msg, str): # "text"
                    if msg == 'end':
                        # Indicate end of the task to Gradio chat
                        return "Command Complete!"
                    
                    if msg.startswith('[LOG]'):
                        complete_response += '\n'
                    if msg.endswith('\\\\'):
                        complete_response += msg.rstrip('\\\\')
                    else:
                        complete_response += msg + '\n'
                yield complete_response

    # def generate_mjpeg_stream(self):
    #     while True:
    #         if self.system_stop:
    #             break
    #         frame = self.llm_controller.get_latest_frame(self.robot_info_list[0], True)
    #         if frame is None:
    #             continue
    #         buf = io.BytesIO()
    #         frame.save(buf, format='JPEG')
    #         buf.seek(0)
    #         yield (b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
    #         time.sleep(1.0 / 30.0)
    def generate_mjpeg_stream(self, robot_id):
        """Generate MJPEG stream for a specific robot by robot_id."""
        while True:
            if self.system_stop:
                break
                
            # Find the robot with matching robot_id
            robot_info = None
            for robot in self.robot_info_list:
                if robot.robot_id == robot_id:
                    robot_info = robot
                    break
                    
            if robot_info is None:
                time.sleep(1.0 / 30.0)
                continue
                
            frame = self.llm_controller.get_latest_frame(robot_info, True)
            if frame is None:
                time.sleep(1.0 / 30.0)
                continue
                
            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0 / 30.0)

    def run(self):
        self.llm_controller.start_controller()

        app = Flask(__name__)
        # @app.route('/robot-pov/')
        # def video_feed():
        #     return Response(self.generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
        @app.route('/robot-pov/<robot_id>/')
        def video_feed(robot_id):
            """Route to get video feed for a specific robot."""
            return Response(
                self.generate_mjpeg_stream(robot_id), 
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        flask_thread = Thread(target=app.run, kwargs={'host': 'localhost', 'port': 50000, 'debug': True, 'use_reloader': False})
        flask_thread.start()

        # Start the Gradio UI
        self.ui.launch(show_api=False, server_port=50001, prevent_thread_lock=True)

        while not self.system_stop:
            time.sleep(1)

        self.llm_controller.stop_controller()


"""
Available robot types: ['tello', 'virtual', 'go2']
"""
if __name__ == "__main__":
    robot_info_list = []
    with open(os.path.join(CURRENT_DIR, 'robot_list.json'), 'r') as f:
        robot_list = json.load(f)
        for robot in robot_list:
            robot_info_list.append(RobotInfo.from_dict(robot))

    typefly = TypeFly(robot_info_list)
    typefly.run()