# -*- coding: utf-8 -*-
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from contextlib import asynccontextmanager
import json, os
import asyncio, aiohttp

from .utils import print_t
from .robot_info import RobotInfo
<<<<<<< HEAD
from .janah_cv import janah_cv
=======
from .janah_cv_v2 import janah_cv_v2 as janah_cv  # ✅ FaceNet فقط — pipeline موحّد
>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2

DIR = os.path.dirname(os.path.abspath(__file__))
EDGE_SERVICE_IP = os.environ.get("EDGE_SERVICE_IP", "localhost")
EDGE_SERVICE_PORT = os.environ.get("EDGE_SERVICE_PORT", "50049")

<<<<<<< HEAD
=======
# ✅ FIX #3: شغّل FaceNet كل N فريم فقط
FACE_MATCH_EVERY_N_FRAMES = 5


>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2
class ObjectInfo:
    def __init__(self, name: str, x, y, w, h, depth=None):
        self.name: str = name
        self.x: float = float(x)
        self.y: float = float(y)
        self.w: float = float(w)
        self.h: float = float(h)
        self.depth: float = float(depth) if depth is not None else None
        self.clothing_color: str = 'unknown'
        self.face_match: int = 0
        self.age_estimate: str = 'unknown'

    @staticmethod
    def from_json(json_data: dict):
        return ObjectInfo(
            json_data['name'], json_data['x'], json_data['y'],
            json_data['w'], json_data['h'], json_data['depth']
        )

    def __str__(self) -> str:
        base = f"- {self.name}: (x:{self.x:.2f}, y:{self.y:.2f}), size: ({self.w:.2f}x{self.h:.2f})"
        if self.name == 'person':
            base += f", clothing: {self.clothing_color}, face_match: {self.face_match}%"
        return base

<<<<<<< HEAD
=======

>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2
from filterpy.kalman import KalmanFilter
from typing import Optional
import time
import numpy as np

<<<<<<< HEAD
=======

>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2
class ObjectTracker:
    def __init__(self, name, x, y, w, h, d) -> None:
        self.name = name
        self.kf_pos = self._init_filter()
        self.kf_siz = self._init_filter()
        self.depth = d
        self.timestamp = 0
        self.update(x, y, w, h, d)

    def update(self, x, y, w, h, d):
        self.kf_pos.update((x, y))
        self.kf_siz.update((w, h))
        self.depth = d
        self.timestamp = time.time()

    def predict(self) -> Optional[ObjectInfo]:
        if time.time() - self.timestamp > 1.0:
            return None
        self.kf_pos.predict()
        self.kf_siz.predict()
        if self.kf_siz.x[0][0] <= 0 or self.kf_siz.x[1][0] <= 0:
            return None
        return ObjectInfo(
            self.name,
            self.kf_pos.x[0][0], self.kf_pos.x[1][0],
            self.kf_siz.x[0][0], self.kf_siz.x[1][0],
            self.depth
        )

<<<<<<< HEAD
    def init_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        kf.R *= 1
        kf.P *= 1000
        kf.Q *= 0.01
=======
    def _init_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        kf.H = np.array([[1,0,0,0],[0,1,0,0]])
        kf.R *= 1; kf.P *= 1000; kf.Q *= 0.01
>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2
        return kf


class YoloClient():
    def __init__(self, robot_info: RobotInfo, enable_tracking: bool = False):
        self.robot_info = robot_info
        self.service_url = 'http://{}:{}/process'.format(EDGE_SERVICE_IP, EDGE_SERVICE_PORT)
        self.target_image_width = 640

        # ✅ FIX #8: tracking من config
        tracking_from_config = robot_info.extra.get('tracking', False) if robot_info.extra else False
        self.enable_tracking = enable_tracking or tracking_from_config

        self._latest_result_lock = asyncio.Lock()
        self._latest_result = (None, [])
        self.frame_id = 0
        self.frame_id_lock = asyncio.Lock()
        self.object_trackers: dict[str, ObjectTracker] = {}
<<<<<<< HEAD
        self._session = None
        print_t(f"[Y] YoloClient initialized with service url: {self.service_url}, tracking: {enable_tracking}")
=======

        # ✅ FIX #3: عدّاد لتقليل تكرار FaceNet
        self._face_match_counter = 0
        self._face_cache: dict[tuple, int] = {}  # cache بالـ bbox position

        print_t(f"[Y] YoloClient → {self.service_url} | tracking: {self.enable_tracking} | face_every: {FACE_MATCH_EVERY_N_FRAMES}f")
>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2

    @property
    def latest_result(self) -> tuple:
        result = self._latest_result
        if result is None:
            return None
        image, objects = result
        return (image, list(objects))

    @staticmethod
    def scale_image(image: Image.Image, target_width: int) -> Image.Image:
        w, h = image.size
        if w <= target_width:
            return image
        scale = target_width / w
<<<<<<< HEAD
        new_size = (target_width, int(h * scale))
        return image.resize(new_size, Image.LANCZOS)

    @staticmethod
    def image_to_bytes(image: Image.Image) -> bytes:
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()
=======
        return image.resize((target_width, int(h * scale)), Image.LANCZOS)

    @staticmethod
    def image_to_bytes(image: Image.Image) -> bytes:
        # ✅ FIX #4: JPEG فقط — content-type متطابق
        buf = BytesIO()
        image.save(buf, format='JPEG', quality=85)
        return buf.getvalue()
>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2

    @staticmethod
    def plot_results_ps(image: Image.Image, object_list: list):
        if not image or len(object_list) == 0:
            return
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(os.path.join(DIR, "assets/Roboto-Medium.ttf"), size=36)
        w, h = image.size
        for obj in object_list:
<<<<<<< HEAD
            x1 = str_float_to_int(obj.x - obj.w / 2, w)
            y1 = str_float_to_int(obj.y - obj.h / 2, h)
            x2 = str_float_to_int(obj.x + obj.w / 2, w)
            y2 = str_float_to_int(obj.y + obj.h / 2, h)
            draw.rectangle([x1, y1, x2, y2], outline='#00FFFF', width=6)
            label = f"{obj.name}"
            if obj.depth is not None:
                label += f" ({obj.depth:.2f}m)"
            draw_y = y1 - 40 if y1 - 40 > 0 else y2 + 10
            draw.text((x1, draw_y), label, fill='red', font=font)

    def cc_to_ps(self, result: list[dict]) -> list[ObjectInfo]:
=======
            x1 = int((obj.x - obj.w/2) * w); y1 = int((obj.y - obj.h/2) * h)
            x2 = int((obj.x + obj.w/2) * w); y2 = int((obj.y + obj.h/2) * h)
            draw.rectangle([x1, y1, x2, y2], outline='#00FFFF', width=6)
            label = f"{obj.name}" + (f" ({obj.depth:.2f}m)" if obj.depth else "")
            draw_y = y1 - 40 if y1 - 40 > 0 else y2 + 10
            draw.text((x1, draw_y), label, fill='red', font=font)

    def cc_to_ps(self, result: list) -> list:
>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2
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
            if self.enable_tracking:
<<<<<<< HEAD
                if obj_info.name not in self.object_trackers:
                    self.object_trackers[obj_info.name] = ObjectTracker(obj_info.name, obj_info.x, obj_info.y, obj_info.w, obj_info.h, obj_info.depth)
=======
                # ✅ FIX #2 جزئي: استخدام اسم + موقع تقريبي كـ key
                # track_id من YOLO يُستخدم لو موجود
                track_id = obj.get('track_id', None)
                key = f"{obj_info.name}_{track_id}" if track_id else obj_info.name
                if key not in self.object_trackers:
                    self.object_trackers[key] = ObjectTracker(
                        obj_info.name, obj_info.x, obj_info.y,
                        obj_info.w, obj_info.h, obj_info.depth
                    )
>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2
                else:
                    self.object_trackers[key].update(
                        obj_info.x, obj_info.y, obj_info.w, obj_info.h, obj_info.depth
                    )
            else:
                rslt.append(obj_info)

        if self.enable_tracking:
            to_delete = []
            for key, tracker in self.object_trackers.items():
                obj = tracker.predict()
                if obj is not None:
                    rslt.append(obj)
                else:
<<<<<<< HEAD
                    to_delete.append(name)
            for name in to_delete:
                del self.object_trackers[name]

        return rslt

    async def detect(self, image: Image.Image, conf=0.2):
        import requests
=======
                    to_delete.append(key)
            for key in to_delete:
                del self.object_trackers[key]

        return rslt

    @asynccontextmanager
    async def get_aiohttp_session_response(service_url, form_data, timeout_seconds=3):
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(service_url, data=form_data) as response:
                    if response.status != 200:
                        response.raise_for_status()
                    yield response
        except aiohttp.ServerTimeoutError:
            print_t(f"[Y] Timeout: {service_url}")

    async def detect(self, image: Image.Image, conf=0.2):
>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2
        config = {
            'robot_info': self.robot_info.robot_id,
            'service_type': 'yolo',
            'tracking_mode': self.enable_tracking,
            'image_id': 0,
            'conf': conf,
        }
        # ✅ FIX #4: JPEG فقط — content-type متطابق
        image_bytes = YoloClient.image_to_bytes(
            YoloClient.scale_image(image, self.target_image_width)
        )

        async with self.frame_id_lock:
            self.frame_id += 1
            config['image_id'] = self.frame_id
<<<<<<< HEAD

        try:
            response = requests.post(
                self.service_url,
                files={'image': ('frame.jpeg', image_bytes, 'image/jpeg')},
                data={'json_data': json.dumps(config)},
                timeout=3
            )
            if response.status_code != 200:
                print_t(f"[YOLO] Invalid status: {response.status_code}")
                return
            json_results = response.json()
=======
            form_data = aiohttp.FormData()
            form_data.add_field('image', image_bytes,
                                filename='frame.jpg',        # ✅ .jpg
                                content_type='image/jpeg')   # ✅ متطابق
            form_data.add_field('json_data', json.dumps(config), content_type='application/json')

        try:
            async with YoloClient.get_aiohttp_session_response(self.service_url, form_data) as response:
                data = await response.text()
                json_results = json.loads(data)
        except json.JSONDecodeError:
            print_t(f"[YOLO] Invalid json")
            return
>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2
        except Exception as e:
            print_t(f"[YOLO] Request failed: {str(e)}")
            return

        list_obj = self.cc_to_ps(json_results.get("result", []))

<<<<<<< HEAD
        image_np = np.array(image)
=======
        # ✅ FIX #3: FaceNet كل N فريم فقط — مش كل مرة
        self._face_match_counter += 1
        run_face_match = (self._face_match_counter % FACE_MATCH_EVERY_N_FRAMES == 0)

        image_np = np.array(image)
        new_cache: dict[tuple, int] = {}

>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2
        for obj in list_obj:
            if obj.name == 'person':
                bbox = {'x': obj.x, 'y': obj.y, 'width': obj.w, 'height': obj.h}
                obj.clothing_color = janah_cv.detect_clothing_color(image_np, bbox)
<<<<<<< HEAD
                obj.face_match = janah_cv.face_match(image_np, bbox)
                obj.age_estimate = janah_cv.estimate_age_from_size(bbox)
=======

                # cache key = موقع bbox مقرّب — مستقر عبر الفريمات بغض النظر عن ترتيب YOLO
                p = 10  # دقة 0.1
                cache_key = (round(obj.x * p) / p, round(obj.y * p) / p)

                if run_face_match:
                    score = janah_cv.face_match(image_np, bbox)
                    obj.face_match = score
                else:
                    obj.face_match = self._face_cache.get(cache_key, 0)

                new_cache[cache_key] = obj.face_match

        self._face_cache = new_cache
>>>>>>> 621e52efaf1a758afa2870b9a3e4fd98c22d73a2

        async with self._latest_result_lock:
            self._latest_result = (image, list_obj)