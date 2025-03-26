from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from contextlib import asynccontextmanager

import json, os
import asyncio, aiohttp, threading

from .utils import print_t
from .robot_info import RobotInfo

DIR = os.path.dirname(os.path.abspath(__file__))

EDGE_SERVICE_IP = os.environ.get("EDGE_SERVICE_IP", "localhost")
# EDGE_SERVICE_IP = "localhost"
EDGE_SERVICE_PORT = os.environ.get("EDGE_SERVICE_PORT", "50049")

class ObjectInfo:
    def __init__(self, name: str, x, y, w, h):
        self.name: str = name
        self.x: float = float(x)
        self.y: float = float(y)
        self.w: float = float(w)
        self.h: float = float(h)

    def from_json(json_data: dict):
        return ObjectInfo(json_data['name'], json_data['x'], json_data['y'], json_data['w'], json_data['h'])

    def __str__(self) -> str:
        return f"- {self.name}: (x:{self.x:.2f}, y:{self.y:.2f}), size: ({self.w:.2f}x{self.h:.2f})"

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
        font = ImageFont.truetype(os.path.join(DIR, "assets/Roboto-Medium.ttf"), size=50)
        w, h = image.size
        for obj in object_list:
            draw.rectangle((str_float_to_int(obj.x - obj.w / 2, w), str_float_to_int(obj.y - obj.h / 2, h), str_float_to_int(obj.x + obj.w / 2, w), str_float_to_int(obj.y + obj.h / 2, h)),
                        fill=None, outline='blue', width=4)
            draw.text((str_float_to_int(obj.x - obj.w / 2, w), str_float_to_int(obj.y - obj.h / 2, h) - 50), obj.name, fill='red', font=font)
    
    @staticmethod
    def cc_to_ps(result: list) -> list[ObjectInfo]:
        return [
            ObjectInfo.from_json({
                'name': obj['name'],
                'x': (obj['box']['x1'] + obj['box']['x2']) / 2,
                'y': (obj['box']['y1'] + obj['box']['y2']) / 2,
                'w': obj['box']['x2'] - obj['box']['x1'],
                'h': obj['box']['y2'] - obj['box']['y1'],
            })
            for obj in result
        ]

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