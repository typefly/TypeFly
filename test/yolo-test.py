import json, requests, os
from PIL import Image

import sys
sys.path.append('..')
from typefly.yolo_client import YoloClient
from typefly.robot_info import RobotInfo

EDGE_SERVICE_IP = os.environ.get("EDGE_SERVICE_IP", "localhost")

def detect_local(image: Image, conf=0.2):
    image_bytes = YoloClient.image_to_bytes(image.resize((640, 352)))

    json_data = {
        'robot_info': RobotInfo('robot3', 'drone', '127.0.0.1').to_json(),
        'service_type': 'yolo',
        'tracking_mode': False,
        'image_id': 1,
        'conf': conf
    }
    http_load = {
        'image': ('image', image_bytes),
        'json_data': (None, json.dumps(json_data))
    }

    response = requests.post(f"http://{EDGE_SERVICE_IP}:{50049}/process", files=http_load)
    print(f"[Y] Response: {response.json()}")

image = Image.open("./images/kitchen.webp")
print(image.size)
detect_local(image)