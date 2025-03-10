from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple
from numpy.typing import NDArray
import numpy as np
from contextlib import asynccontextmanager

import json, os
import requests
import queue
import asyncio, aiohttp

from .utils import print_t
from .shared_frame import SharedFrame, Frame
from .robot_info import RobotInfo

DIR = os.path.dirname(os.path.abspath(__file__))

EDGE_SERVICE_IP = os.environ.get("EDGE_SERVICE_IP", "localhost")
EDGE_SERVICE_PORT = os.environ.get("EDGE_SERVICE_PORT", "50049")

'''
Access the YOLO service through http.
'''
class YoloClient():
    def __init__(self, shared_frame: SharedFrame=None):
        self.service_url = 'http://{}:{}/process'.format(EDGE_SERVICE_IP, EDGE_SERVICE_PORT)
        self.image_size = (640, 352)
        self.frame_queue = queue.Queue() # queue element: (frame_id, frame)
        self.shared_frame = shared_frame
        self.frame_id = 0
        self.frame_id_lock = asyncio.Lock()

    def is_local_service(self) -> bool:
        return EDGE_SERVICE_IP == 'localhost'

    @staticmethod
    def image_to_bytes(image: Image.Image) -> bytes:
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()
    
    @staticmethod
    def plot_results(frame, results):
        if results is None:
            return
        def str_float_to_int(value, multiplier):
            return int(float(value) * multiplier)
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype(os.path.join(DIR, "assets/Roboto-Medium.ttf"), size=50)
        w, h = frame.size
        for result in results:
            box = result["box"]
            draw.rectangle((str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h), str_float_to_int(box["x2"], w), str_float_to_int(box["y2"], h)),
                        fill=None, outline='blue', width=4)
            draw.text((str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h) - 50), result["name"], fill='red', font=font)

    @staticmethod
    def plot_results_oi(frame, object_list):
        if object_list is None or len(object_list) == 0:
            return
        def str_float_to_int(value, multiplier):
            return int(float(value) * multiplier)
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype(os.path.join(DIR, "assets/Roboto-Medium.ttf"), size=50)
        w, h = frame.size
        for obj in object_list:
            draw.rectangle((str_float_to_int(obj.x - obj.w / 2, w), str_float_to_int(obj.y - obj.h / 2, h), str_float_to_int(obj.x + obj.w / 2, w), str_float_to_int(obj.y + obj.h / 2, h)),
                        fill=None, outline='blue', width=4)
            draw.text((str_float_to_int(obj.x - obj.w / 2, w), str_float_to_int(obj.y - obj.h / 2, h) - 50), obj.name, fill='red', font=font)

    def retrieve(self) -> Optional[SharedFrame]:
        return self.shared_frame
    
    @asynccontextmanager
    async def get_aiohttp_session_response(service_url, data, timeout_seconds=3):
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        try:
            # The ClientSession now uses the defined timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(service_url, data=data) as response:
                    response.raise_for_status()  # Optional: raises exception for 4XX/5XX responses
                    yield response
        except aiohttp.ServerTimeoutError:
            print_t(f"[Y] Timeout error when connecting to {service_url}")

    def detect_local(self, frame: Frame, conf=0.2):
        image = frame.image
        image_bytes = YoloClient.image_to_bytes(image.resize(self.image_size))
        self.frame_queue.put(frame)

        config = {
            'robot_info': RobotInfo('robot', 'drone').to_json(),
            'service_type': 'yolo',
            'tracking_mode': False,
            'image_id': self.frame_id,
            'conf': conf
        }
        files = {
            'image': ('image', image_bytes),
            'json_data': (None, json.dumps(config))
        }

        print_t(f"[Y] Sending request to {self.service_url}")

        response = requests.post(self.service_url, files=files)
        print_t(f"[Y] Response: {response.text}")
        json_results = json.loads(response.text)
        if self.shared_frame is not None:
            self.shared_frame.set(self.frame_queue.get(), json_results)

    async def detect(self, frame: Frame, conf=0.3):
        if self.is_local_service():
            self.detect_local(frame, conf)
            return
        
        image = frame.image
        image_bytes = YoloClient.image_to_bytes(image.resize(self.image_size))

        async with self.frame_id_lock:
            self.frame_queue.put((self.frame_id, frame))
            config = {
                'robot_info': RobotInfo('robot', 'drone').to_json(),
                'service_type': 'yolo',
                'tracking_mode': False,
                'image_id': self.frame_id,
                'conf': conf
            }
            http_load = {
                'image': image_bytes,
                'json_data': json.dumps(config)
            }
            self.frame_id += 1

        async with YoloClient.get_aiohttp_session_response(self.service_url, http_load) as response:
            results = await response.text()

        try:
            json_results = json.loads(results)
        except:
            print_t(f"[Y] Invalid json results: {results}")
            return
        
        if 'image_id' not in json_results:
            print_t(f"[Y] Missing image_id in results: {json_results}")
            return
        
        # Safe queue processing
        result_image_id = json_results['image_id']
        try:
            # discard old images
            if self.frame_queue.empty():
                print_t("[Y] Frame queue empty, cannot process results")
                return
                
            # Discard frames older than our result
            while not self.frame_queue.empty() and self.frame_queue.queue[0][0] < result_image_id:
                discarded = self.frame_queue.get()
                print_t(f"[Y] Discarded old frame: {discarded[0]}")
                
            # Check if we have the matching frame
            if not self.frame_queue.empty() and self.frame_queue.queue[0][0] == result_image_id:
                matched_frame = self.frame_queue.get()
            else:
                print_t(f"[Y] No matching frame for result id: {result_image_id}")
                return
                
        except Exception as e:
            print_t(f"[Y] Error processing frame queue: {e}")
            return

        # Update shared frame with results
        if self.shared_frame is not None and matched_frame is not None:
            self.shared_frame.set(matched_frame[1], json_results)