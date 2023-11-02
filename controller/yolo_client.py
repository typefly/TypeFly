from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple
from contextlib import asynccontextmanager

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
        self.result_with_image = (None, {})
        self.lock = threading.Lock()

    def get(self) -> Tuple[Image.Image, dict]:
        with self.lock:
            return self.result_with_image
        
    def set(self, val: Tuple[Image.Image, dict]):
        with self.lock:
            self.result_with_image = val

class YoloClient():
    def __init__(self, shared_yolo_results: SharedYoloResults=None):
        self.service_url = 'http://{}:{}/yolo'.format(YOLO_SERVICE_IP, ROUTER_SERVICE_PORT)
        self.image_size = (640, 352)
        self.image_queue = queue.Queue() # queue element: (image_id, image)
        self.shared_yolo_results = shared_yolo_results
        self.latest_result_with_image = None # (image, json_data: {'image_id': int, 'result': [result]})
        self.latest_result_with_image_lock = asyncio.Lock()
        self.image_id = 0
        self.image_id_lock = asyncio.Lock()

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

    def retrieve(self) -> Optional[Tuple[Image.Image, dict]]:
        return self.latest_result_with_image
    
    @asynccontextmanager
    async def get_aiohttp_session_response(service_url, data):
        async with aiohttp.ClientSession() as session:
            async with session.post(service_url, data=data) as response:
                yield response

    async def detect(self, image):
        image_bytes = YoloClient.image_to_bytes(image.resize(self.image_size))

        async with self.image_id_lock:
            self.image_queue.put((self.image_id, image))
            files = {
                'image': image_bytes,
                'json_data': json.dumps({'user_name': 'yolo', 'stream_mode': True, 'image_id': self.image_id})
            }
            self.image_id += 1

        async with YoloClient.get_aiohttp_session_response(self.service_url, files) as response:
            results = await response.text()

        try:
            json_results = json.loads(results)
        except:
            print(f"Invalid json results: {results}")
            return
        async with self.latest_result_with_image_lock:
            # discard old images
            if self.image_queue.empty():
                return
            while self.image_queue.queue[0][0] < json_results['image_id']:
                self.image_queue.get()
            # discard old results
            if self.image_queue.queue[0][0] > json_results['image_id']:
                return

            self.latest_result_with_image = (self.image_queue.get()[1], json_results)
            if self.shared_yolo_results is not None:
                self.shared_yolo_results.set(self.latest_result_with_image)