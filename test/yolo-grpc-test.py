from io import BytesIO
from PIL import Image
import json, sys, os
import grpc

def image_to_bytes(image: Image.Image) -> bytes:
    # compress and convert the image to bytes
    imgByteArr = BytesIO()
    image.save(imgByteArr, format='WEBP')
    return imgByteArr.getvalue()

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(PARENT_DIR, "proto/generated"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

VISION_SERVICE_IP = "localhost"
YOLO_SERVICE_PORT = 50050

with grpc.insecure_channel(f'{VISION_SERVICE_IP}:{YOLO_SERVICE_PORT}') as channel:
    stub = hyrch_serving_pb2_grpc.YoloServiceStub(channel)

    json_data = {
        'robot_info': '{"robot_id": "robot", "robot_type": "drone"}',
        'service_type': 'yolo',
        'tracking_mode': False,
        'conf': 0.3
    }
    detect_request = hyrch_serving_pb2.DetectRequest(image_data=image_to_bytes(Image.open("./images/kitchen.webp")), json_data=json.dumps(json_data))
    response = stub.Detect(detect_request)

    json_results = json.loads(response.json_data)
    print(json_results)