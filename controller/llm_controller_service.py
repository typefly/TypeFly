import io, os, time
from threading import Thread
from flask import Flask, request, jsonify, Response
import logging
import asyncio

from llm_controller import LLMController

logging.disable(logging.CRITICAL + 1)
app = Flask(__name__)
FRAME_RATE = 30

llm_controller = LLMController()

@app.route('/send_command', methods=['POST'])
def process_command():
    data = request.json
    received_string = data.get('command')
    print(f"Received string: {received_string}")
    if received_string == "exit":
        llm_controller.stop_controller()
        def shutdown():
            print("Shutting down gracefully...")
            time.sleep(1)
            os._exit(0)
        shutdown_thread = Thread(target=shutdown).start()
    else:
        llm_controller.execute_user_command(received_string)
    return jsonify({'result': 'success'})

def generate_mjpeg_stream():
    while True:
        image_data = llm_controller.get_latest_frame()
        buf = io.BytesIO()
        image_data.save(buf, format='JPEG')
        buf.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
        time.sleep(1.0 / FRAME_RATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''<html>
                <head>
                    <title>Drone POV</title>
                </head>
                <body>
                    <h1>Drone POV</h1>
                    <img src="/video_feed" width="960" height="720" />
                </body>
              </html>'''

# asyncio functions
global asyncio_loop
asyncio_loop = asyncio.get_event_loop()
def start_async_loop():
    global asyncio_loop
    asyncio_loop.run_forever()

async def stop_async_loop():
    global asyncio_loop
    asyncio_loop.stop()

def stop_loop_from_thread():
    global asyncio_loop
    # Schedule the stopping function to run on the loop
    asyncio_loop.call_soon_threadsafe(asyncio_loop.create_task, stop_async_loop())

if __name__ == "__main__":
    # Start the asyncio loop in a separate thread
    async_thread = Thread(target=start_async_loop)
    async_thread.start()

    # Start the LLM controller
    llm_controller.start_robot()
    
    # Start the LLM capture loop in a separate thread
    llm_thread = Thread(target=llm_controller.capture_loop, args=(asyncio_loop,))
    llm_thread.start()

    # Start the Flask server
    app.run(host='localhost', port=50001, debug=True)

    # Stop the LLM controller
    stop_loop_from_thread()
    llm_thread.join()
    async_thread.join()
    llm_controller.stop_robot()