import queue
import sys
import asyncio
import io, time
import gradio as gr
from flask import Flask, Response
from threading import Thread

sys.path.append("..")
from controller.llm_controller import LLMController
from controller.utils import print_t

class TypeFly:
    def __init__(self, use_virtual_cam):
        self.message_queue = queue.Queue()
        self.llm_controller = LLMController(use_virtual_cam, self.message_queue)
        self.system_stop = False
        self.ui = gr.Blocks(title="TypeFly")
        self.asyncio_loop = asyncio.get_event_loop()
        with self.ui:
            gr.HTML(open('./header.html', 'r').read())
            gr.HTML(open('./drone-pov.html', 'r').read())
            gr.ChatInterface(self.process_message).queue()

    def process_message(self, message, history):
        print_t(f"[S] Receiving task description: {message}")
        if message == "exit":
            self.llm_controller.stop_controller()
            self.system_stop = True
            yield "Shutting down..."
        elif len(message) == 0:
            yield "[WARNING] Empty command!]"
        else:
            task_thread = Thread(target=self.llm_controller.execute_task_description, args=(message,))
            task_thread.start()
            complete_message = ''
            while True:
                msg = self.message_queue.get()
                if msg == 'end':
                    # Indicate end of the task to Gradio chat
                    return "Command Complete!"
                else:
                    complete_message += msg + '\n'
                    yield complete_message
                
    def generate_mjpeg_stream(self):
        while True:
            if self.system_stop:
                break
            frame = self.llm_controller.get_latest_frame()
            if frame is None:
                continue
            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0 / 30.0)

    def run(self):
        asyncio_thread = Thread(target=self.asyncio_loop.run_forever)
        asyncio_thread.start()

        self.llm_controller.start_robot()
        llmc_thread = Thread(target=self.llm_controller.capture_loop, args=(self.asyncio_loop,))
        llmc_thread.start()

        app = Flask(__name__)
        @app.route('/drone-pov/')
        def video_feed():
            return Response(self.generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
        flask_thread = Thread(target=app.run, kwargs={'host': 'localhost', 'port': 50000, 'debug': True, 'use_reloader': False})
        flask_thread.start()
        self.ui.launch(show_api=False, server_port=50001, prevent_thread_lock=True)
        while True:
            time.sleep(1)
            if self.system_stop:
                break

        llmc_thread.join()
        asyncio_thread.join()

        self.llm_controller.stop_robot()

if __name__ == "__main__":
    typefly = TypeFly(use_virtual_cam=True)
    typefly.run()
        