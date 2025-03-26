import sys, os, json
from quart import Quart, request, jsonify
import multiprocessing, signal
import logging

from service_manager import ServiceManager
import yolo_service

PROJ_DIR = os.environ.get("PROJ_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, PROJ_DIR)
from typefly.robot_info import RobotInfo

sys.path.append(os.path.join(PROJ_DIR, "typefly/proto"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.log'),
    level=logging.DEBUG,  # Capture all levels
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

EDGE_SERVICE_PORT = os.environ.get("EDGE_SERVICE_PORT", "50049")
YOLO_SERVICE_INFO = { "host": "localhost", "port" : [50050, 50051]}

app = Quart(__name__)
grpcServiceManager = ServiceManager()

@app.before_serving
async def before_serving():
    global grpcServiceManager
    grpcServiceManager.add_service("yolo", YOLO_SERVICE_INFO["host"], YOLO_SERVICE_INFO["port"])
    await grpcServiceManager._initialize_channels()

@app.route('/process', methods=['POST'])
async def process():
    global grpcServiceManager
    form = await request.form
    json_str = form.get('json_data')

    # print(f"Received request with json_data: {json_str}")
    if not json_str:
        return {"error": "Missing json data"}, 400
    
    try:
        json_data: dict = json.loads(json_str)
        robot_info = RobotInfo.from_json(json_data["robot_info"])
        service_type = json_data["service_type"]

        if service_type == "yolo":
            files = await request.files
            image_data = files['image']
            image_bytes = image_data.read()
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}, 400
    except KeyError as e:
        return {"error": f"Missing key {e} in JSON data"}, 400
    except Exception as e:
        return {"error": f"Error: {e}"}, 400

    channel = await grpcServiceManager.get_service_channel(service_type=service_type, robot_info=robot_info)

    if type(channel) == str:
        return {"error": f"Error connecting to service: {channel}"}, 400

    if service_type == "yolo":
        stub = hyrch_serving_pb2_grpc.YoloServiceStub(channel)
        response = await stub.Detect(hyrch_serving_pb2.DetectRequest(json_data=json_str, image_data=image_bytes))

    return response.json_data

def start_yolo_service():
    process_count = len(YOLO_SERVICE_INFO["port"])
    processes = []

    for i in range(process_count):
        process = multiprocessing.Process(target=yolo_service.serve, args=(YOLO_SERVICE_INFO["port"][i],), daemon=True)
        process.start()
        processes.append(process)
    
    return processes

if __name__ == "__main__":
    processes = start_yolo_service()
    def cleanup(_signalnum, _frame):
        print("Shutting down YOLO services...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        print("Shutdown complete.")
        exit(0)

    # Catch termination signals
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    app.run(debug=True, host='0.0.0.0', port=EDGE_SERVICE_PORT)