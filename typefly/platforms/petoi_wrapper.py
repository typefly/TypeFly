"""
TypeFly platform wrapper for a Petoi quadruped (Bittle / Nybble / Cub) running
the OpenCatESP32 firmware in this repo's sibling project `typego-petoi-firmware`.

Unlike the Go2 (ROS2 / cmd_vel), the Petoi is driven over plain HTTP:

  * Locomotion / control board — the OpenCatESP32 board exposes a JSON HTTP API
    on port 80 (see OpenCatESP32/src/httpServer.h). We use:
        POST /nav   {"vx": <m/s>, "vyaw": <rad/s>}  velocity -> nearest gait
        POST /up    stand up (valid posture before gaits)
        POST /rest  fold down / power-save
        GET  /version  reachability + build/model probe
    /nav maps a body-frame velocity to the nearest discrete gait: +vx -> wkF,
    -vx -> bk, +vyaw -> vtL (turn left in place), -vyaw -> vtR, and zero
    velocity -> balance (stop). Only the SIGN selects the gait; the magnitude
    just has to clear the firmware dead-zones, and gait speed is fixed by /speed.
    /nav returns as soon as the gait is loaded (the gait then runs continuously),
    so a distance or heading is achieved open-loop: hold the gait for a timed
    duration, then send a zero-velocity /nav to stop.

    NOTE: this targets the *deployed* firmware (build "Apr 30 2026", commit
    66cca52), which exposes /nav + /control but NOT /walk or /rotate (those were
    added later in cfa0d94). Flashing current firmware restores /rotate, whose
    IMU-closed-loop turn is far more accurate than the timed turn used here.

  * Camera — a *separate* Seeed XIAO ESP32S3 board running
    `esp32-xiao-cam-stream` serves a continuous MJPEG stream at
    `http://<camera-ip>/` (VGA 640x480 JPEG). OpenCV's VideoCapture decodes the
    multipart stream directly.

Direction conventions (kept consistent with RobotWrapper):
    _move(+dx) -> forward, _move(+dy) -> left
    _rotate(+deg) -> turn left (counter-clockwise)
Firmware matches these: /nav +vx -> wkF (forward); /nav +vyaw -> vtL (left /
CCW). Quadruped gaits have no lateral translation, so a sideways dy is emulated
by turning to face the target, walking, then turning back.

Config (typefly/config/robot_info.json):
    {
        "robot_id": "petoi1",
        "robot_type": "petoi",
        "extra": {
            "ip": "192.168.1.50",          # OpenCatESP32 control board (required)
            "port": 80,                     # optional, defaults to 80
            "camera_ip": "192.168.1.51",   # XIAO cam board (required for vision)
            "camera_port": 80,              # optional, defaults to 80
            "camera_url": "http://.../",   # optional, overrides camera_ip/port
            "walk_speed": 0.07,            # optional m/s, open-loop walk calibration
            "rotate_speed": 45.0           # optional deg/s, open-loop turn calibration
        }
    }
Remember to register the platform in llm_controller.py:
    elif robot_info.robot_type == "petoi":
        from .platforms.petoi_wrapper import PetoiWrapper
        self.robot = PetoiWrapper(robot_info)
"""

import time, json
import urllib.request, urllib.error
import threading
from typing import Any, Optional

import cv2
from PIL import Image
from overrides import overrides

from ..robot_wrapper import RobotWrapper, RobotObservation
from ..robot_info import RobotInfo
from ..yolo_client import YoloClient
from ..utils import print_t

# --- Defaults / firmware-derived constants ----------------------------------
DEFAULT_HTTP_PORT = 80
DEFAULT_CAMERA_PORT = 80
# wkF/bk advance at roughly this speed at the firmware's default /speed
# multiplier. Walking is open-loop (time x speed): we hold the gait for
# |distance| / speed seconds, so this constant sets that duration.
DEFAULT_WALK_SPEED_MPS = 0.07
# vtL/vtR spin in place at roughly this rate at the default /speed multiplier.
# Open-loop like walking (the deployed firmware has no IMU-closed-loop /rotate),
# so we hold the turn gait for |deg| / rotate_speed seconds. Surface- and
# battery-dependent — calibrate per robot, or flash firmware with /rotate.
DEFAULT_ROTATE_SPEED_DPS = 45.0
# /nav velocity magnitudes. Only the SIGN picks the gait; the magnitude just has
# to clear chooseNavSkill's dead-zones (LIN_DEAD=0.05 m/s, ANG_DEAD=0.1 rad/s).
NAV_LIN_VX = 0.1
NAV_ANG_VYAW = 0.5
# Distances/headings smaller than these are no-ops.
MIN_MOVE_M = 0.02
MIN_ROTATE_DEG = 1.0
# Let the robot settle into the balance posture between commands.
SETTLE_DELAY = 0.3

# --- /control "euler" body-pose gestures (radians) ---------------------------
# The firmware exposes POST /control {"command":"euler","roll","pitch","yaw"},
# which sets an *absolute* body orientation (radians) without walking. All
# gesture/look skills go through this.
#
# Per-axis limits are enforced by the firmware (OpenCatESP32 README "/euler") and
# OUT-OF-RANGE VALUES ARE REJECTED by the board, so we clamp to them. They are
# asymmetric. Sign conventions, also from the firmware:
#   * yaw  + = head pans LEFT (matches orienting()/nav "+ = left").
#   * pitch is inverted vs intuition: NEGATIVE = head UP (rear squat),
#     POSITIVE = head DOWN. Range is asymmetric (more down-travel than up).
#   * roll + = lean (IMU-stabilised).
EULER_ROLL_LIMITS = (-0.4, 0.4)
EULER_PITCH_LIMITS = (-0.2, 0.5)
EULER_YAW_LIMITS = (-1.0, 1.0)
# Hold each pose briefly so the motion is visible before the next one.
EULER_POSE_HOLD = 0.25
GESTURE_REPEATS = 3
# nod -> dip the head down then up, repeatedly (both within the pitch range).
NOD_PITCH_DOWN_RAD = 0.3    # +pitch = head down
NOD_PITCH_UP_RAD = -0.15    # -pitch = head up (>= EULER_PITCH_LIMITS[0] = -0.2)
# shake_head -> yaw left/right ("no" gesture), within +-1.0.
SHAKE_YAW_RAD = 0.5
# look_object maps an object's 0-1 frame offset from center to head angles.
# yaw: +yaw = left, matching orienting()'s (0.5 - x). pitch: an object ABOVE the
# frame center (y < 0.5) -> head up (negative); BELOW (y > 0.5) -> head down
# (positive), using the asymmetric up/down swings below (both within range).
LOOK_YAW_RANGE_RAD = 0.8    # +-0.4 swing at frame edges (within +-1.0)
LOOK_PITCH_UP_RAD = 0.2     # max head-up magnitude  -> maps to -0.2
LOOK_PITCH_DOWN_RAD = 0.4   # max head-down magnitude -> maps to +0.4
LOOK_DURATION_S = 5.0       # look_object follows the object for this long


def _clamp(value: float, limits: tuple[float, float]) -> float:
    """Clamp a commanded euler angle (radians) to its firmware axis limits."""
    lo, hi = limits
    return max(lo, min(hi, value))


class PetoiObservation(RobotObservation):
    """
    Pulls frames from the XIAO MJPEG camera stream and runs YOLO detection.

    The Petoi exposes no odometry/orientation HTTP endpoint, so _position and
    _orientation stay at their zero defaults (the common vision/movement skills
    do not depend on them).
    """
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        self.yolo_client = YoloClient(robot_info)

        extra = robot_info.extra or {}
        self.camera_url: Optional[str] = extra.get("camera_url")
        if not self.camera_url:
            camera_ip = extra.get("camera_ip")
            if camera_ip:
                camera_port = extra.get("camera_port", DEFAULT_CAMERA_PORT)
                self.camera_url = f"http://{camera_ip}:{camera_port}/"
        if not self.camera_url:
            print_t("[Petoi] No 'camera_ip'/'camera_url' in robot_info.extra — "
                    "vision disabled.")

        self.capture_thread = threading.Thread(target=self._capture_spin, daemon=True)

    def _capture_spin(self):
        # Create the capture and read in the same thread (like the other
        # wrappers). MJPEG streams can drop, so reopen on failure until stopped.
        while self.running:
            if not self.camera_url:
                time.sleep(0.5)
                continue
            cap = cv2.VideoCapture(self.camera_url)
            if not cap.isOpened():
                print_t(f"[Petoi] camera stream not available at {self.camera_url}, "
                        "retrying...")
                cap.release()
                time.sleep(1.0)
                continue
            try:
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        print_t("[Petoi] camera read failed, reconnecting...")
                        break
                    # OpenCV decodes JPEG to BGR; TypeFly expects an RGB PIL image.
                    self._image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(1)
            finally:
                cap.release()
            if self.running:
                time.sleep(0.5)

    @overrides
    def _start(self):
        self.capture_thread.start()

    @overrides
    def _stop(self):
        self.capture_thread.join()

    @overrides
    async def process_image(self, image: Image.Image):
        await self.yolo_client.detect(image)

    @overrides
    def fetch_processed_result(self) -> dict[str, Any]:
        _, object_list = self.yolo_client.latest_result
        return {
            "yolo": object_list
        }


class PetoiWrapper(RobotWrapper):
    camera_moves_with_body = True  # body-mounted forward camera rotates/moves with the quadruped
    def __init__(self, robot_info: RobotInfo):
        super().__init__(robot_info, PetoiObservation(robot_info))

        extra = robot_info.extra or {}
        ip = extra.get("ip")
        if not ip:
            raise ValueError("Petoi robot_info.extra must contain 'ip' "
                             "(OpenCatESP32 control board address)")
        port = extra.get("port", DEFAULT_HTTP_PORT)
        self.base_url = f"http://{ip}:{port}"
        self.walk_speed = float(extra.get("walk_speed", DEFAULT_WALK_SPEED_MPS))
        self.rotate_speed = float(extra.get("rotate_speed", DEFAULT_ROTATE_SPEED_DPS))

        # Petoi-specific expressive skills driven by the firmware /control "euler"
        # command (body pose, no walking). See the gesture methods below.
        self.skillset.add_skill(self.nod, "Nod the head up and down (a 'yes' gesture)")
        self.skillset.add_skill(self.shake_head, "Shake the head left and right (a 'no' gesture)")
        self.skillset.add_skill(self.look_object, "Tilt the head to look toward an object without moving the body")

    # --- HTTP helpers --------------------------------------------------------
    def _post(self, path: str, payload: Optional[dict] = None,
              timeout: float = 10.0) -> Optional[dict]:
        """POST JSON to the control board. Returns parsed JSON, or None on error."""
        url = f"{self.base_url}{path}"
        data = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"})
        return self._request(req, path, timeout)

    def _get(self, path: str, timeout: float = 5.0) -> Optional[dict]:
        """GET JSON from the control board. Returns parsed JSON, or None on error."""
        req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        return self._request(req, path, timeout)

    def _request(self, req: urllib.request.Request, path: str,
                 timeout: float) -> Optional[dict]:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            result = json.loads(body) if body else {}
            if isinstance(result, dict) and result.get("ok") is False:
                print_t(f"[Petoi] {path} returned error: {result}")
            return result
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")
            print_t(f"[Petoi] {path} -> HTTP {e.code}: {detail}")
        except (urllib.error.URLError, OSError, ValueError) as e:
            print_t(f"[Petoi] {path} request failed: {e}")
        return None

    # --- Lifecycle -----------------------------------------------------------
    @overrides
    def start(self) -> bool:
        version = self._get("/version", timeout=5.0)
        if version is None:
            print_t(f"[Petoi] Could not reach control board at {self.base_url}")
            return False
        print_t(f"[Petoi] Connected to {self.base_url}: {version}")
        # Stand up so the gaits have a valid starting posture. /up is blocking,
        # so it returns once the robot has finished standing.
        self._post("/up", timeout=10.0)
        self.obs.start()
        return True

    @overrides
    def stop(self) -> bool:
        self.obs.stop()
        # Fold the legs / power-save the servos before exiting.
        self._post("/rest", timeout=10.0)
        return True

    @overrides
    def stop_motion(self) -> None:
        # Zero velocity -> firmware 'balance' gait (stop in place, stay standing).
        self._post("/nav", {})

    @overrides
    def emergency_shutdown(self) -> None:
        # Reserved for fatal paths: fold legs / power-save the servos.
        self._post("/rest", timeout=10.0)

    # --- Movement primitives -------------------------------------------------
    def _walk(self, distance: float):
        """Open-loop forward (+) / backward (-) walk of `distance` metres.

        The deployed firmware has no /walk endpoint, so replicate it client-side:
        start the wkF (forward) / bk (back) gait via /nav, hold it for
        |distance| / walk_speed seconds, then stop with a zero-velocity /nav.
        """
        if abs(distance) < MIN_MOVE_M:
            return
        vx = NAV_LIN_VX if distance > 0 else -NAV_LIN_VX
        # /nav returns once the gait is loaded; the gait then runs on its own.
        self._post("/nav", {"vx": vx})
        time.sleep(abs(distance) / self.walk_speed)
        self._post("/nav", {})  # vx, vyaw below dead-zone -> balance (stop)
        time.sleep(SETTLE_DELAY)

    @overrides
    def _move(self, dx: float, dy: float):
        """
        Move dx (forward+/back-) and dy (left+/right-) metres.

        The quadruped gait cannot strafe, so lateral motion is emulated by
        turning to face the target direction, walking, then turning back to the
        original heading (/rotate is IMU closed-loop, so the heading recovers).
        """
        print_t(f"-> Move by ({dx}, {dy}) m")
        if abs(dx) >= MIN_MOVE_M:
            self._walk(dx)
        if abs(dy) >= MIN_MOVE_M:
            turn = 90.0 if dy > 0 else -90.0  # +dy = left -> turn left first
            self._rotate(turn)
            self._walk(abs(dy))
            self._rotate(-turn)

    @overrides
    def _rotate(self, deg: float):
        """Turn left (+) / right (-) by `deg` degrees, open-loop (timed).

        The deployed firmware has no IMU-closed-loop /rotate, so time the
        in-place vtL/vtR turn gait: start it via /nav, hold for
        |deg| / rotate_speed seconds, then stop. Calibrate `rotate_speed`
        (deg/s) for your surface, or flash firmware with /rotate for accuracy.
        """
        print_t(f"-> Rotate by {deg} degrees")
        if abs(deg) < MIN_ROTATE_DEG:
            return
        # +deg (left/CCW) -> positive vyaw -> vtL; negative -> vtR.
        vyaw = NAV_ANG_VYAW if deg > 0 else -NAV_ANG_VYAW
        self._post("/nav", {"vyaw": vyaw})
        time.sleep(abs(deg) / self.rotate_speed)
        self._post("/nav", {})  # vx, vyaw below dead-zone -> balance (stop)
        time.sleep(SETTLE_DELAY)

    # --- Expressive body-pose skills (firmware /control "euler") --------------
    def _euler(self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> bool:
        """Set an absolute body orientation (radians) via /control 'euler'.

        Each angle is clamped to its firmware axis limits (the board rejects
        out-of-range values), and the call is routed through the plan policy
        ('action': counts against the budget + honors a cancel) so a plan can't
        spam poses or run past cancellation. Returns True if the board accepted
        the command (False if the HTTP request failed — _post already logs why).
        """
        self._policy_guard("action")
        result = self._post("/control", {
            "command": "euler",
            "roll": _clamp(roll, EULER_ROLL_LIMITS),
            "pitch": _clamp(pitch, EULER_PITCH_LIMITS),
            "yaw": _clamp(yaw, EULER_YAW_LIMITS),
        })
        time.sleep(EULER_POSE_HOLD)
        return result is not None

    def _level(self) -> None:
        """Command a level (all-zero) posture directly, bypassing the policy
        guard so it still runs from a finally on the cancel/error cleanup path
        (the guard would re-raise PlanCancelled and skip the reset)."""
        self._post("/control", {"command": "euler", "roll": 0.0, "pitch": 0.0, "yaw": 0.0})

    def nod(self):
        """Nod the head up and down (a 'yes' gesture) by oscillating euler pitch."""
        print_t("-> Nod")
        try:
            for _ in range(GESTURE_REPEATS):
                self._euler(pitch=NOD_PITCH_DOWN_RAD)  # head down
                self._euler(pitch=NOD_PITCH_UP_RAD)    # head up
        finally:
            self._level()  # always end level, even if cancelled mid-gesture

    def shake_head(self):
        """Shake the head left and right (a 'no' gesture) by oscillating euler yaw."""
        print_t("-> Shake head")
        try:
            for _ in range(GESTURE_REPEATS):
                self._euler(yaw=SHAKE_YAW_RAD)   # +yaw = left
                self._euler(yaw=-SHAKE_YAW_RAD)  # -yaw = right
        finally:
            self._level()  # always end level, even if cancelled mid-gesture

    def look_object(self, object_name: str) -> bool:
        """Track `object_name` with the head for ~LOOK_DURATION_S seconds.

        Repeatedly reads the object's normalized frame position (0-1, 0.5 =
        center) and updates the euler yaw/pitch so the head follows it in place,
        until the duration elapses or the object leaves the view. Returns True if
        the object was in sight at least once, else False.
        """
        print_t(f"-> Look at {object_name} for {LOOK_DURATION_S:.0f}s")
        deadline = time.monotonic() + LOOK_DURATION_S
        tracked = False
        while time.monotonic() < deadline:
            self._policy_poll()  # stay cancellable across the tracking window
            x = self.object_x(object_name)
            y = self.object_y(object_name)
            # object_x/object_y yield a 'not in sight' string when the object is
            # lost; stop tracking (a flicker simply ends the look early).
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                if not tracked:
                    print_t(f"-> look_object: {object_name} is not in sight")
                break
            # +yaw = left (CCW) like orienting(); object left of center -> +yaw.
            yaw = (0.5 - x) * LOOK_YAW_RANGE_RAD
            # Below center (y>0.5) -> look down (+pitch); above -> up (-pitch).
            # Asymmetric: more down-travel than up (firmware pitch range).
            v = (y - 0.5) * 2.0  # -1 (top) .. +1 (bottom)
            pitch = v * (LOOK_PITCH_DOWN_RAD if v >= 0 else LOOK_PITCH_UP_RAD)
            self._euler(pitch=pitch, yaw=yaw)  # paces the loop via EULER_POSE_HOLD
            tracked = True
        return tracked
