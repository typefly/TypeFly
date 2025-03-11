import sys, json, requests
from PIL import Image
sys.path.append("..")
from controller.yolo_client import YoloClient
from serving.edge.service_manager import RobotInfo

def detect_local(image: Image, conf=0.2):
    image_bytes = YoloClient.image_to_bytes(image.resize((640, 352)))

    json_data = {
        'robot_info': RobotInfo('robot3', 'drone').to_json(),
        'service_type': 'yolo',
        'tracking_mode': False,
        'conf': conf
    }
    http_load = {
        'image': ('image', image_bytes),
        'json_data': (None, json.dumps(json_data))
    }

    response = requests.post(f"http://{'127.0.0.1'}:{50049}/process", files=http_load)
    print(f"[Y] Response: {response.json()}")

image = Image.open("./images/kitchen.webp")
print(image.size)
detect_local(image)