import os, json
import requests
from io import BytesIO
from PIL import Image
from .yolo_grpc_client import YoloGRPCClient
from .yolo_client import YoloClient

current_directory = os.path.dirname(os.path.abspath(__file__))

class VLMWrapper:
    def __init__(self):
        self.url = 'http://172.29.249.77:50049/llava'
        with open(os.path.join(current_directory, "./assets/vlm_prompt.txt"), "r") as f:
             self.vlm_prompt = f.read()

        self.yolo = YoloGRPCClient(True)

    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()

    def request(self, question, image):
        self.yolo.detect_local(image)
        detect_result = self.yolo.retrieve()[1]['result']

        image_bytes = VLMWrapper.image_to_bytes(image)
        prompt = self.vlm_prompt.format(task_descrption=question, detection_results=detect_result)
        print(f"> VLM request: {prompt}...")
        files = {
            'image': ('image', image_bytes),
            'json_data': (None, json.dumps({'query': prompt}))
        }
        response = requests.post(self.url, files=files)
        print(response)
        return response.text
