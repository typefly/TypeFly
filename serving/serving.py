import gradio as gr
import logging, os, signal, time
import argparse, asyncio, sys
from threading import Thread
from flask import Flask, Response, request
import io
import queue

sys.path.append("..")
from controller.llm_controller import LLMController
from controller.utils import print_t

logging.disable(logging.CRITICAL + 1)

'''
    Shutdown the service
'''
global system_stop
system_stop = False
def shutdown():
    print_t("[S] Shutting down gracefully in 0.5 seconds...")
    time.sleep(0.5)
    os.kill(os.getpid(), signal.SIGINT)

global message_queue
message_queue = queue.Queue()

global llm_controller
def init_llm_controller(use_virtual_cam):
    global llm_controller
    llm_controller = LLMController(use_virtual_cam, message_queue)

def process_command(command, history):
    global system_stop
    print_t(f"[S] Receiving task description: {command}")
    if command == "exit":
        llm_controller.stop_controller()
        system_stop = True
        yield "Shutting down..."
    elif len(command) == 0:
        return
    else:
        task_thread = Thread(target=llm_controller.execute_task_description, args=(command,))
        task_thread.start()
        complete_message = ''
        while True:
            msg = message_queue.get()
            if msg == 'end':
                # Indicate end of the task to Gradio chat
                return "Command Complete!"
            else:
                time.sleep(0.1)
                complete_message += msg + '\n'
                yield complete_message

with gr.Blocks(title="TypeFly") as demo:
    gr.HTML(open('./header.html', 'r').read())
    gr.HTML(open('./drone-pov.html', 'r').read())
    gr.ChatInterface(process_command).queue()

'''
    Flask video stream service
'''
app = Flask(__name__)
def run_flask():
    app.run(host='localhost', port=50000, debug=True, use_reloader=False)

def generate_mjpeg_stream():
    while True:
        image_data = llm_controller.get_latest_frame()
        buf = io.BytesIO()
        image_data.save(buf, format='JPEG')
        buf.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
        time.sleep(1.0 / 30)

@app.route('/drone-pov/')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

'''
    Async even loop
'''
global asyncio_loop
asyncio_loop = asyncio.get_event_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-v", "--virtual-camera", action='store_true', help="Use laptop camera")
    args = parser.parse_args()
    init_llm_controller(args.virtual_camera)
    # Start the asyncio loop in a separate thread
    async_thread = Thread(target=asyncio_loop.run_forever)
    async_thread.start()

    # Start the LLM controller
    llm_controller.start_robot()
    
    # Start the LLM capture loop in a separate thread
    llm_thread = Thread(target=llm_controller.capture_loop, args=(asyncio_loop,))
    llm_thread.start()

    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    # Start the gradio server
    demo.launch(show_api=False, server_port=50001, prevent_thread_lock=True)

    while True:
        time.sleep(1)
        if system_stop:
            break

    # Stop the LLM controller
    print("Main thread waiting for llm thread to finish")
    llm_thread.join()
    print("Main thread waiting for flask thread to finish")
    # flask_thread.join()
    print("Main thread waiting for async thread to finish")
    async_thread.join()
    print("Main thread exiting")
    llm_controller.stop_robot()
    