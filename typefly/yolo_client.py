from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from contextlib import asynccontextmanager

import json, os
import asyncio, aiohttp, threading

from .utils import print_t
from .robot_info import RobotInfo

DIR = os.path.dirname(os.path.abspath(__file__))

# EDGE_SERVICE_IP = os.environ.get("EDGE_SERVICE_IP", "localhost")
EDGE_SERVICE_IP = "localhost"
EDGE_SERVICE_PORT = os.environ.get("EDGE_SERVICE_PORT", "50049")

class ObjectInfo:
    def __init__(self, name: str, x, y, w, h, depth=None):
        self.name: str = name
        self.x: float = float(x)
        self.y: float = float(y)
        self.w: float = float(w)
        self.h: float = float(h)
        self.depth: float = float(depth) if depth is not None else None

    def from_json(json_data: dict):
        return ObjectInfo(json_data['name'], json_data['x'], json_data['y'], json_data['w'], json_data['h'], json_data['depth'])

    def __str__(self) -> str:
        return f"- {self.name}: (x:{self.x:.2f}, y:{self.y:.2f}), size: ({self.w:.2f}x{self.h:.2f})"

from filterpy.kalman import KalmanFilter
from typing import Optional
import time
import numpy as np
class ObjectTracker:
    def __init__(self, name, x, y, w, h, d) -> None:
        self.name = name
        self.kf_pos = self.init_filter()
        self.kf_siz = self.init_filter()
        self.timestamp = 0
        self.size = None
        self.update(x, y, w, h, d)

    def update(self, x, y, w, h, d):
        self.kf_pos.update((x, y))
        self.kf_siz.update((w, h))
        self.depth = d
        self.timestamp = time.time()

    def predict(self) -> Optional[ObjectInfo]:
        # if no update in 2 seconds, return None
        if time.time() - self.timestamp > 1.0:
            return None
        self.kf_pos.predict()
        self.kf_siz.predict()
        if self.kf_siz.x[0][0] <= 0 or self.kf_siz.x[1][0] <= 0:
            return None
        return ObjectInfo(self.name, self.kf_pos.x[0][0], self.kf_pos.x[1][0], self.kf_siz.x[0][0], self.kf_siz.x[1][0], self.depth)

    def init_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state dimensions (x, y, vx, vy), 2 measurement dimensions (x, y)
        kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                        [0, 1, 0, 0]])
        kf.R *= 1  # Measurement uncertainty
        kf.P *= 1000  # Initial uncertainty
        kf.Q *= 0.01  # Process uncertainty
        return kf

"""
Access the YOLO service through http.
"""
class YoloClient():
    def __init__(self, robot_info: RobotInfo):
        self.robot_info = robot_info
        self.service_url = 'http://{}:{}/process'.format(EDGE_SERVICE_IP, EDGE_SERVICE_PORT)
        self.image_size = (640, 352)
        self._latest_result_lock = threading.Lock()
        self._latest_result = (None, [])
        self.frame_id = 0
        self.frame_queue = asyncio.Queue() # queue element: (frame_id, frame)
        self.frame_queue_lock = asyncio.Lock()
        print_t(f"[Y] YoloClient initialized with service url: {self.service_url}")

        self.object_trackers: dict[str, ObjectTracker] = {}

    @property
    def latest_result(self) -> tuple[Image.Image, list]:
        with self._latest_result_lock:
            return self._latest_result

    @staticmethod
    def image_to_bytes(image: Image.Image) -> bytes:
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()

    @staticmethod
    def plot_results_ps(image: Image.Image, object_list: list[ObjectInfo]):
        if not image or len(object_list) == 0:
            return

        def str_float_to_int(value, multiplier):
            return int(float(value) * multiplier)

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(os.path.join(DIR, "assets/Roboto-Medium.ttf"), size=36)
        w, h = image.size

        for obj in object_list:
            x1 = str_float_to_int(obj.x - obj.w / 2, w)
            y1 = str_float_to_int(obj.y - obj.h / 2, h)
            x2 = str_float_to_int(obj.x + obj.w / 2, w)
            y2 = str_float_to_int(obj.y + obj.h / 2, h)

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline='#00FFFF', width=6)

            # Draw label and depth
            label = f"{obj.name}"
            if obj.depth is not None:
                label += f" ({obj.depth:.2f}m)"
            draw_y = y1 - 40 if y1 - 40 > 0 else y2 + 10
            draw.text((x1, draw_y), label, fill='red', font=font)
    
    def cc_to_ps(self, result: list[dict]) -> list[ObjectInfo]:
        rslt = []
        for obj in result:
            obj_info = ObjectInfo.from_json({
                'name': obj['name'],
                'x': (obj['box']['x1'] + obj['box']['x2']) / 2,
                'y': (obj['box']['y1'] + obj['box']['y2']) / 2,
                'w': obj['box']['x2'] - obj['box']['x1'],
                'h': obj['box']['y2'] - obj['box']['y1'],
                'depth': obj['depth'] / 2 if 'depth' in obj else None
            })

            if obj_info.w <= 0 or obj_info.h <= 0:
                continue

            if obj_info.name not in self.object_trackers:
                self.object_trackers[obj_info.name] = ObjectTracker(obj_info.name, obj_info.x, obj_info.y, obj_info.w, obj_info.h, obj_info.depth)
            else:
                self.object_trackers[obj_info.name].update(obj_info.x, obj_info.y, obj_info.w, obj_info.h, obj_info.depth)
            
        to_delete = []
        for name, tracker in self.object_trackers.items():
            obj = tracker.predict()
            if obj is not None:
                rslt.append(obj)
            else:
                to_delete.append(name)   
        for name in to_delete:
            del self.object_trackers[name] 
        
        return rslt

    @asynccontextmanager
    async def get_aiohttp_session_response(service_url, form_data, timeout_seconds=3):
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        try:
            # The ClientSession now uses the defined timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(service_url, data=form_data) as response:
                    if response.status != 200:
                        print_t(f"[Y] Invalid response status: {response.status}")
                        response.raise_for_status()  # Optional: raises exception for 4XX/5XX responses
                    yield response
        except aiohttp.ServerTimeoutError:
            print_t(f"[Y] Timeout error when connecting to {service_url}")

    async def detect(self, image: Image.Image, conf=0.3):
        async with self.frame_queue_lock:
            self.frame_id += 1
            # print_t(f"[Y] Sending request with image id: {self.frame_id} {self.frame_queue.qsize()}")
            await self.frame_queue.put((self.frame_id, image))
            
            config = {
                'robot_info': self.robot_info.to_json(),
                'service_type': 'yolo',
                'tracking_mode': False,
                'image_id': self.frame_id,
                'conf': conf
            }
            form_data = aiohttp.FormData()
            image_bytes = YoloClient.image_to_bytes(image.resize(self.image_size))
            form_data.add_field('image', image_bytes, filename='frame.webp', content_type='image/webp')
            form_data.add_field('json_data', json.dumps(config), content_type='application/json')

        async with YoloClient.get_aiohttp_session_response(self.service_url, form_data) as response:
            data = await response.text()

        try:
            json_results = json.loads(data)
        except:
            print_t(f"[Y] Invalid json results: {data}")
            return
        
        if 'image_id' not in json_results:
            print_t(f"[Y] Missing image_id in results: {json_results}")
            return
        
        # Safe queue processing
        result_image_id = json_results['image_id']
        # print_t(f"[Y] Received results for image id: {result_image_id} {self.frame_queue.qsize()}")
        async with self.frame_queue_lock:
            # Discard frames older than our result
            while not self.frame_queue.empty():
                head_frame = await self.frame_queue.get()
                if head_frame[0] == result_image_id:
                    matched_frame = head_frame
                    break
                elif head_frame[0] > result_image_id:
                    print_t(f"[Y] Discarded old result: {head_frame[0]}")
                    return
                else:
                    print_t(f"[Y] Discarded old frame: {head_frame[0]}")

        # Update latest result
        with self._latest_result_lock:
            self._latest_result = (image, self.cc_to_ps(json_results["result"]))