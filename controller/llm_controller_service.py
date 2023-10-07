import io, os, time
from threading import Thread
from flask import Flask, request, jsonify, Response
import logging
logging.disable(logging.CRITICAL + 1)

from llm_controller import LLMController

app = Flask(__name__)
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

def llm_run():
    llm_controller.run()

if __name__ == "__main__":
    llm_thread = Thread(target=llm_run)
    llm_thread.start()
    app.run(host='localhost', port=50001)
    llm_thread.join()