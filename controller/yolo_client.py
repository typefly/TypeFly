from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple

import json
import queue
import asyncio, aiohttp
import threading

# YOLO_SERVICE_IP = 'localhost'
YOLO_SERVICE_IP = '172.29.249.77'
YOLO_SERVICE_PORT = '50051'
ROUTER_SERVICE_PORT = '50049'

class SharedYoloResults():
    def __init__(self) -> None:
        self.result_with_image = None
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.result_with_image
        
    def set(self, val):
        with self.lock:
            self.result_with_image = val

class YoloClient():
    def __init__(self, shared_yolo_results: SharedYoloResults=None):
        self.service_url = 'http://{}:{}/yolo'.format(YOLO_SERVICE_IP, ROUTER_SERVICE_PORT)
        self.image_size = (640, 352)
        self.image_queue = queue.Queue()
        self.shared_yolo_results = shared_yolo_results
        self.latest_result_with_image = None
        self.latest_result_with_image_lock = asyncio.Lock()

    def local_service(self):
        return YOLO_SERVICE_IP == 'localhost'

    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()
    
    def plot_results(frame, results):
        def str_float_to_int(value, multiplier):
            return int(float(value) * multiplier)
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=50)
        w, h = frame.size
        for result in results:
            box = result["box"]
            draw.rectangle((str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h), str_float_to_int(box["x2"], w), str_float_to_int(box["y2"], h)),
                        fill=None, outline='blue', width=4)
            draw.text((str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h) - 50), result["name"], fill='red', font=font)

    def retrieve(self) -> Optional[Tuple[Image.Image, str]]:
        return self.latest_result_with_image

    async def detect(self, image):
        image_bytes = YoloClient.image_to_bytes(image.resize(self.image_size))
        self.image_queue.put(image)

        files = {
            'image': image_bytes,
            'json_data': json.dumps({'user_name': 'yolo', 'stream_mode': True})
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.service_url, data=files) as response:
                results = await response.text()
                async with self.latest_result_with_image_lock:
                    json_results = json.loads(results)
                    self.latest_result_with_image = (self.image_queue.get(), json_results)
                    if self.shared_yolo_results is not None:
                        self.shared_yolo_results.set(self.latest_result_with_image)