import os, json
import requests
from io import BytesIO
from PIL import Image
from yolo_grpc_client import YoloGRPCClient

class VLMWrapper:
    def __init__(self):
        self.url = 'http://172.29.249.77:50049/llava'
        with open("./assets/vlm_prompt.txt", "r") as f:
            self.vlm_prompt = f.read()

        self.yolo = YoloGRPCClient(True)

    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()

    def request(self, question, image):
        self.yolo.detect_local(image)
        detect_result = self.yolo.retrieve()[1]

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
    
if __name__ == "__main__":
    vlm_wrapper = VLMWrapper()
    result = vlm_wrapper.request("which person wears the blue shirt", Image.open("../test/images/people.webp"))
    print(result)