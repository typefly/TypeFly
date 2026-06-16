from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from contextlib import asynccontextmanager
from collections import defaultdict
import json, os
import asyncio, aiohttp

from .utils import print_t
from .robot_info import RobotInfo

DIR = os.path.dirname(os.path.abspath(__file__))

EDGE_SERVICE_IP = os.environ.get("EDGE_SERVICE_IP", "localhost")
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
import time, threading, itertools
import numpy as np

# --- Client-side tracking-by-detection tunables ---
# Sized for a modest 2-5 Hz detection feed with a handful of same-class objects.
N_CONFIRM = 2         # cumulative hits before a track is emitted (suppresses 1-frame false positives)
M_MAX_MISS = 3        # consecutive missed frames a track may coast before it is reaped
T_MAX_AGE = 1.0       # wall-clock seconds since last detection before a track is reaped
BASE_GATE = 0.12      # base association gate: max normalized centre distance for a match
MOTION_GATE_K = 1.5   # gate widening per unit of commanded ego-motion (handles camera pans)
SIZE_W = 0.25         # weight of the size-mismatch term in the association cost
EGO_HFOV_DEG = 70.0   # assumed horizontal FOV; converts a commanded yaw into a normalized x shift

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except ImportError:  # scipy is optional; a greedy fallback keeps deps light
    _HAVE_SCIPY = False


def _wall_clock() -> float:
    """Indirection over time.time so tests can drive frames deterministically."""
    return time.time()


def _make_filter() -> KalmanFilter:
    # Constant-velocity model: state (x, y, vx, vy), measurement (x, y). These are
    # the original ObjectTracker matrices, kept verbatim so filter behaviour is a
    # known quantity; only the *management* of filters around them has changed.
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.R *= 1      # measurement uncertainty
    kf.P *= 1000   # initial uncertainty
    kf.Q *= 0.01   # process uncertainty
    return kf


class Track:
    """A single tracked object instance, with client-owned identity.

    Identity is established here by spatial association, NOT by the server's
    ByteTrack id (which churns on exactly the glitches we must survive). A track
    is `tentative` until it accrues N_CONFIRM cumulative hits, then becomes
    `confirmed` and is emitted. A confirmed track keeps emitting a Kalman-
    predicted ("coasting") box through short detection dropouts, and is reaped
    once it has missed for M_MAX_MISS frames or T_MAX_AGE seconds.
    """
    _ids = itertools.count(1)

    def __init__(self, cls: str, x, y, w, h, d, now: float):
        self.cls = cls
        self.id = next(Track._ids)          # internal only; never surfaced in ObjectInfo.name
        self.kf_pos = _make_filter()
        self.kf_siz = _make_filter()
        self.kf_pos.x[0, 0], self.kf_pos.x[1, 0] = x, y   # seed mean with the first measurement
        self.kf_siz.x[0, 0], self.kf_siz.x[1, 0] = w, h
        self.depth = d
        self.hits = 1
        self.misses = 0
        self.confirmed = (N_CONFIRM <= 1)
        self.last_update = now

    def predict(self) -> None:
        self.kf_pos.predict()
        self.kf_siz.predict()

    def shift_prediction(self, sx: float) -> None:
        """Ego-motion compensation: nudge the predicted centre by a commanded
        horizontal image shift so association survives a camera pan."""
        self.kf_pos.x[0, 0] += sx

    @property
    def cx(self) -> float:
        return float(self.kf_pos.x[0, 0])

    @property
    def cy(self) -> float:
        return float(self.kf_pos.x[1, 0])

    @property
    def cw(self) -> float:
        return float(self.kf_siz.x[0, 0])

    @property
    def ch(self) -> float:
        return float(self.kf_siz.x[1, 0])

    @property
    def speed(self) -> float:
        return float(np.hypot(self.kf_pos.x[2, 0], self.kf_pos.x[3, 0]))

    def update(self, x, y, w, h, d, now: float) -> None:
        self.kf_pos.update((x, y))
        self.kf_siz.update((w, h))
        self.depth = d
        self.hits += 1
        self.misses = 0
        self.last_update = now
        if self.hits >= N_CONFIRM:
            self.confirmed = True

    def mark_missed(self) -> None:
        self.misses += 1

    def is_dead(self, now: float) -> bool:
        if self.misses > M_MAX_MISS:
            return True
        if now - self.last_update > T_MAX_AGE:
            return True
        return self.cw <= 0 or self.ch <= 0

    def to_object_info(self) -> ObjectInfo:
        return ObjectInfo(self.cls, self.cx, self.cy, self.cw, self.ch, self.depth)


class YoloClient():
    """
    Access the YOLO service through http POST request.
    You can enable tracking mode to track the objects across frames.
    """
    def __init__(self, robot_info: RobotInfo, enable_tracking: bool = True):
        self.robot_info = robot_info
        self.service_url = 'http://{}:{}/process'.format(EDGE_SERVICE_IP, EDGE_SERVICE_PORT)
        self.target_image_width = 640
        self.enable_tracking = enable_tracking
        self._latest_result_lock = asyncio.Lock()
        self._latest_result = (None, [])
        self.frame_id = 0
        self.frame_id_lock = asyncio.Lock()
        # Client-owned tracks: identity is by spatial association, not server id.
        self.tracks: list[Track] = []
        # Ego-motion hints fed by the robot's motion skills (see note_ego_motion).
        # Written from the plan thread, consumed from the observation thread.
        self._ego_lock = threading.Lock()
        self._pending_dx = 0.0   # normalized horizontal image shift (commanded yaw)
        self._pending_fwd = 0.0  # commanded forward-motion magnitude

        print_t(f"[Y] YoloClient initialized with service url: {self.service_url}, tracking: {enable_tracking}")

    @property
    def latest_result(self) -> tuple[Image.Image, list]:
        result = self._latest_result
        if result is None:
            return None
        image, objects = result
        # shallow copy of list to decouple from async updates
        return (image, list(objects))

    @staticmethod
    def scale_image(image: Image.Image, target_width: int) -> Image.Image:
        w, h = image.size
        if w <= target_width:
            return image
        scale = target_width / w
        new_size = (target_width, int(h * scale))
        # print_t(f"[Y] Scaling image from ({w}, {h}) to {new_size}")
        return image.resize(new_size, Image.LANCZOS)

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
    
    def note_ego_motion(self, d_yaw_deg: float = 0.0, d_forward: float = 0.0) -> None:
        """Record commanded robot motion so the next association step can predict
        where tracked objects will move in the image.

        ``d_yaw_deg`` follows the platform's ``_rotate`` convention (positive =
        rotate left). A left turn shifts a static object toward larger image-x,
        so the predicted centre is nudged by ``+d_yaw_deg / HFOV``. Forward
        motion mostly rescales boxes rather than translating their centres, so it
        only loosens the gate. Called from the plan thread; cc_to_ps consumes the
        hint from the observation thread, hence the lock."""
        with self._ego_lock:
            self._pending_dx += d_yaw_deg / EGO_HFOV_DEG
            self._pending_fwd += d_forward

    @staticmethod
    def _strip_track_id(name: str) -> str:
        """Defensive: a server still in tracking mode emits '{class}_{id}'. Strip
        a trailing _<digits> so client identity is class-based, not server-id
        based, regardless of the server's mode."""
        head, sep, tail = name.rpartition('_')
        return head if (sep and tail.isdigit()) else name

    def cc_to_ps(self, result: list[dict]) -> list[ObjectInfo]:
        """Convert raw YOLO detections to a de-duplicated list of ObjectInfo.

        When tracking is enabled the client owns identity: each frame's
        detections are associated per-class to existing tracks by motion-
        compensated proximity, and a tentative/confirmed/coasting lifecycle keeps
        a single stable entry per physical object across YOLO glitches (dropped
        frames, server track-id switches, 1-frame false positives)."""
        dets: dict[str, list[tuple]] = defaultdict(list)
        for obj in result:
            cls = self._strip_track_id(obj['name'])
            box = obj['box']
            x = (box['x1'] + box['x2']) / 2
            y = (box['y1'] + box['y2']) / 2
            w = box['x2'] - box['x1']
            h = box['y2'] - box['y1']
            if w <= 0 or h <= 0:
                continue
            d = obj['depth'] / 2 if 'depth' in obj else None
            dets[cls].append((x, y, w, h, d))

        # Consume the ego-motion hint regardless of mode so it can't accumulate.
        with self._ego_lock:
            sx, fwd = self._pending_dx, self._pending_fwd
            self._pending_dx = self._pending_fwd = 0.0

        if not self.enable_tracking:
            # No tracking: return detections directly (legacy passthrough).
            return [ObjectInfo(cls, *m) for cls, ms in dets.items() for m in ms]

        now = _wall_clock()
        ego_mag = abs(sx) + 0.5 * abs(fwd)

        # Predict every existing track one step, then apply ego-motion comp.
        for t in self.tracks:
            t.predict()
            t.shift_prediction(sx)

        # Associate per class: an apple detection may only match an apple track.
        for cls in set(dets) | {t.cls for t in self.tracks}:
            cls_tracks = [t for t in self.tracks if t.cls == cls]
            cls_dets = dets.get(cls, [])
            matches, un_tracks, un_dets = self._associate(cls_tracks, cls_dets, ego_mag)
            for ti, di in matches:
                cls_tracks[ti].update(*cls_dets[di], now=now)
            for ti in un_tracks:
                cls_tracks[ti].mark_missed()
            for di in un_dets:
                self.tracks.append(Track(cls, *cls_dets[di], now=now))

        # Reap dead tracks; emit one ObjectInfo per confirmed survivor.
        self.tracks = [t for t in self.tracks if not t.is_dead(now)]
        return [t.to_object_info() for t in self.tracks if t.confirmed]

    def _associate(self, tracks: list, dets: list, ego_mag: float):
        """One-to-one match detections to tracks within an adaptive gate.

        Returns (matches, unmatched_track_idx, unmatched_det_idx), where matches
        is a list of (track_idx, det_idx). Cost is centre distance plus a small
        size-mismatch term; pairs beyond the motion-widened gate are forbidden.
        Distinct nearby instances stay separate because assignment is one-to-one
        and cost-minimizing, not merged by the gate alone."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        INF = 1e6
        cost = np.full((len(tracks), len(dets)), INF)
        for i, t in enumerate(tracks):
            gate = BASE_GATE + MOTION_GATE_K * ego_mag + t.speed
            for j, (x, y, w, h, _d) in enumerate(dets):
                cd = float(np.hypot(t.cx - x, t.cy - y))
                if cd > gate:
                    continue  # forbidden pair
                sz = (abs(t.cw - w) + abs(t.ch - h)) / max(t.cw + t.ch, 1e-3)
                cost[i, j] = cd + SIZE_W * sz

        if _HAVE_SCIPY:
            rows, cols = linear_sum_assignment(cost)
            # Drop pairs the solver was forced to make across the gate (cost==INF).
            matches = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < INF]
        else:
            matches = []
            taken_t, taken_d = set(), set()
            for _c, i, j in sorted((cost[i, j], i, j)
                                   for i in range(len(tracks))
                                   for j in range(len(dets)) if cost[i, j] < INF):
                if i in taken_t or j in taken_d:
                    continue
                taken_t.add(i)
                taken_d.add(j)
                matches.append((i, j))

        matched_t = {i for i, _ in matches}
        matched_d = {j for _, j in matches}
        un_tracks = [i for i in range(len(tracks)) if i not in matched_t]
        un_dets = [j for j in range(len(dets)) if j not in matched_d]
        return matches, un_tracks, un_dets

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

    async def detect(self, image: Image.Image, conf=0.2):
        """
        Detect the objects in the image asynchronously using the YOLO service.
        """
        config = {
            'robot_info': self.robot_info.robot_id,
            'service_type': 'yolo',
            # Server runs plain detection (clean class names); the client owns
            # identity/tracking. Keeping this False also makes the pooled YOLO
            # worker stateless and avoids the reload_model() mode-flip cost.
            'tracking_mode': False,
            'image_id': 0,
            'conf': conf,
        }
        image_bytes = YoloClient.image_to_bytes(YoloClient.scale_image(image, self.target_image_width))

        async with self.frame_id_lock:
            self.frame_id += 1
            config['image_id'] = self.frame_id
            
            form_data = aiohttp.FormData()
            form_data.add_field('image', image_bytes, filename='frame.jpeg', content_type='image/jpeg')
            form_data.add_field('json_data', json.dumps(config), content_type='application/json')

        try:
            async with YoloClient.get_aiohttp_session_response(self.service_url, form_data) as response:
                data = await response.text()
                json_results = json.loads(data)
        except json.JSONDecodeError:
            print_t(f"[YOLO] Invalid json results: {data}")
            return
        except Exception as e:
            print_t(f"[YOLO] Request failed: {str(e)}")
            return
        
        list_obj = self.cc_to_ps(json_results.get("result", []))
        async with self._latest_result_lock:
            self._latest_result = (image, list_obj)