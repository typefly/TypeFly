import queue
import sys, os
import asyncio
import io, time
import gradio as gr
from flask import Flask, Response
from threading import Thread
import argparse

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJ_DIR)
from controller.llm_controller import LLMController
from controller.utils import print_t
from controller.abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class TypeFly:
    def __init__(self, robot_type):
         # create a cache folder
        self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        os.makedirs(self.cache_folder, exist_ok=True)

        self.message_queue = queue.Queue()
        self.llm_controller = LLMController(robot_type, self.message_queue)
        self.system_stop = False

        self.asyncio_loop = asyncio.new_event_loop()
        self.asyncio_thread = Thread(target=self.run_async_loop, daemon=True)
        self.asyncio_thread.start()

        self.ui = gr.Blocks(title="TypeFly")
        self.setup_ui()

    def setup_ui(self):
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
            gr.HTML(open(os.path.join(CURRENT_DIR, 'drone-pov.html'), 'r').read())
            gr.ChatInterface(self.process_message, retry_btn=None, fill_height=False, examples=default_sentences).queue()

    def run_async_loop(self):
        """Runs an asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self.asyncio_loop)
        self.asyncio_loop.run_forever()

    def process_message(self, message, history):
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

    def generate_mjpeg_stream(self):
        while True:
            if self.system_stop:
                break
            frame = self.llm_controller.get_latest_frame(True)
            if frame is None:
                continue
            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0 / 30.0)

    def run(self):
        # asyncio_thread = Thread(target=self.asyncio_loop.run_forever)
        # asyncio_thread.start()

        self.llm_controller.start_robot()
        llmc_thread = Thread(target=self.llm_controller.capture_loop, args=(self.asyncio_loop,), daemon=True)
        llmc_thread.start()

        app = Flask(__name__)
        @app.route('/drone-pov/')
        def video_feed():
            return Response(self.generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
        flask_thread = Thread(target=app.run, kwargs={'host': 'localhost', 'port': 50000, 'debug': True, 'use_reloader': False})
        flask_thread.start()

        # Start the Gradio UI
        self.ui.launch(show_api=False, server_port=50001, prevent_thread_lock=True)

        while not self.system_stop:
            time.sleep(1)

        llmc_thread.join()
        
        self.shutdown(llmc_thread)

    def shutdown(self, llmc_thread):
        """Shuts down the system gracefully."""
        self.llm_controller.stop_robot()
        llmc_thread.join()

        # Stop asyncio loop
        self.asyncio_loop.call_soon_threadsafe(self.asyncio_loop.stop)
        self.asyncio_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--virtual', action='store_true')
    parser.add_argument('--go2', action='store_true')

    args = parser.parse_args()
    robot_type = RobotType.TELLO
    if args.virtual:
        robot_type = RobotType.VIRTUAL
    elif args.go2:
        robot_type = RobotType.GO2
    typefly = TypeFly(robot_type)
    typefly.run()