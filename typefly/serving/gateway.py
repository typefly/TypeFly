from quart import Quart, request
import json, sys, os

from typefly.serving.service_manager import ServiceManager
from typefly.serving.config import SERVICE_INFO, PROJ_DIR
from typefly.proto._bootstrap import ensure_stubs

ensure_stubs()  # generate gRPC stubs on first run before importing them
sys.path.append(os.path.join(PROJ_DIR, "./proto"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

app = Quart(__name__)
grpcServiceManager = ServiceManager()

@app.route('/health', methods=['GET'])
async def health():
    """Readiness probe used by the launcher and the web UI."""
    return {"status": "ok"}

@app.before_serving
async def before_serving():
    for service_name, service_info in SERVICE_INFO.items():
        grpcServiceManager.add_service(service_name, service_info["host"], service_info["port"])
    await grpcServiceManager._initialize_channels()

@app.route('/process', methods=['POST'])
async def process():
    form = await request.form
    json_str = form.get('json_data')

    if not json_str:
        return {"error": "Missing json data"}, 400

    try:
        json_data = json.loads(json_str)
        robot_info = json_data["robot_info"]
        service_type = json_data["service_type"]

        if SERVICE_INFO[service_type]["require_image"]:
            image_data = await request.files
            if "image" not in image_data:
                return {"error": "Missing image data"}, 400
            image_data = image_data["image"]
            image_bytes = image_data.read()
        else:
            image_bytes = None

    except Exception as e:
        print(f"Error: {e}")
        return {"error": f"{type(e).__name__}: {e}"}, 400

    channel = await grpcServiceManager.get_service_channel(service_type, robot_info)

    if isinstance(channel, str):
        return {"error": f"Channel error: {channel}"}, 400

    if service_type == "yolo":
        stub = hyrch_serving_pb2_grpc.YoloServiceStub(channel)
        response = await stub.Detect(hyrch_serving_pb2.DetectRequest(
            json_data=json_str,
            image_data=image_bytes
        ))
    # elif other services, you can add more here
    else:
        return {"error": "Service not found"}, 400

    return response.json_data
