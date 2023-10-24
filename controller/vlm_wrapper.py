import os, json
import requests
from io import BytesIO
from PIL import Image

class VLMWrapper:
    def __init__(self):
        self.url = 'http://172.29.249.77:50049/llava'

    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()

    def request(self, prompt, image):
        image_bytes = VLMWrapper.image_to_bytes(image)
        files = {
            'image': ('image', image_bytes),
            'json_data': (None, json.dumps({'query': prompt}))
        }
        response = requests.post(self.url, files=files)
        return response.text
    
if __name__ == "__main__":
    vlm_wrapper = VLMWrapper()
    result = vlm_wrapper.request("From left to right, which one is in dark green?", Image.open("../test/images/people.webp"))
    print(result)