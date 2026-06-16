from PIL import Image
import queue, io, base64, time
from typing import Optional
import threading
import traceback

from .robot_wrapper import RobotWrapper
from .llm_planner import LLMPlanner
from .utils import print_t
from .robot_info import RobotInfo
from .plan_execution import (
    SkillNamespace,
    PlanPolicy,
    PlanValidationError,
    PlanCancelled,
    PlanLimitExceeded,
    extract_plan_json,
    validate_plan,
)

_USER_LOG_QUEUE = queue.Queue()

# Wall-clock budget for a single plan. Because we cannot forcibly kill a thread,
# this is enforced cooperatively at skill-call boundaries (see PlanPolicy); it
# bounds plans built from the validator's bounded loops + clamped skill calls.
PLAN_TIMEOUT_SEC = 120.0
# Bounded re-planning when the model returns invalid/unsafe output.
MAX_PLAN_ATTEMPTS = 3


class LLMController():
    def __init__(self, robot_info: RobotInfo, plan_timeout: float = PLAN_TIMEOUT_SEC):
        self.controller_func = [
            self._user_log,
            self._probe
        ]
        RobotWrapper.set_controller_func(self.controller_func)

        if robot_info.robot_type == "virtual":
            from .platforms.virtual_robot_wrapper import VirtualRobotWrapper
            self.robot = VirtualRobotWrapper(robot_info)
        elif robot_info.robot_type == "tello":
            from .platforms.tello_wrapper import TelloWrapper
            self.robot = TelloWrapper(robot_info)
        elif robot_info.robot_type == "go2":
            from .platforms.go2_wrapper import Go2Wrapper
            self.robot = Go2Wrapper(robot_info)
        elif robot_info.robot_type == "petoi":
            from .platforms.petoi_wrapper import PetoiWrapper
            self.robot = PetoiWrapper(robot_info)
        else:
            raise ValueError(f"Unknown robot type: {robot_info.robot_type}")
        self.planner = LLMPlanner(self.robot)

        self.plan_timeout = plan_timeout
        # Single worker consumes instructions sequentially, so two instructions
        # can never drive the robot's actuators concurrently. A new instruction
        # cancels the running plan (cooperatively) before its own runs.
        self._instruction_queue: "queue.Queue[str]" = queue.Queue()
        self._cancel_event = threading.Event()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    def _user_log(self, msg: str | Image.Image) -> bool:
        if isinstance(msg, Image.Image):
            buffer = io.BytesIO()
            msg.save(buffer, format="JPEG")
            encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
            _USER_LOG_QUEUE.put(f'<img src="data:image/jpeg;base64,{encoded_img}" />')
        else:
            text = str(msg).strip('\'')
            _USER_LOG_QUEUE.put(f'[ROBOT] {text}')
            print_t(f'[ROBOT] {text}')
        return True

    def _probe(self, query: str, robot_info: RobotInfo) -> str:
        return self.planner.probe(query, robot_info)

    def start_controller(self):
        self.robot.start()
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="plan-worker", daemon=True
        )
        self._worker_thread.start()

    def stop_controller(self):
        self._running = False
        self._cancel_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
        self.robot.stop()

    def fetch_robot_pov(self, overlay: bool=True) -> Optional[Image.Image]:
        from .yolo_client import YoloClient
        image = self.robot.obs.image
        yolo_results = self.robot.obs.image_process_result.get("yolo", [])
        if overlay:
            YoloClient.plot_results_ps(image, yolo_results)
        return image

    def put_instruction(self, user_instruction: str):
        """Queue an instruction. Preempts any plan currently running so the new
        one is not racing the old one for the robot."""
        self._cancel_event.set()
        self._instruction_queue.put(user_instruction)

    # ------------------------------------------------------------------ #
    # Worker / plan lifecycle
    # ------------------------------------------------------------------ #
    def _worker_loop(self):
        while self._running:
            try:
                instruction = self._instruction_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            # If several instructions were queued in a burst, keep only the most
            # recent one.
            while True:
                try:
                    instruction = self._instruction_queue.get_nowait()
                except queue.Empty:
                    break
            # The cancel set by put_instruction was meant for the *previous* plan;
            # clear it before running this one.
            self._cancel_event.clear()
            try:
                self._execute_instruction(instruction)
            except Exception as e:  # pragma: no cover - worker must never die
                print_t(f"[P] Worker error: {e}")
                print_t(traceback.format_exc())
                _USER_LOG_QUEUE.put(f'[ERROR] {e}')
            finally:
                _USER_LOG_QUEUE.put('#end')

    def _execute_instruction(self, user_instruction: str):
        error_message = None  # rejection reason fed back into the next attempt
        for attempt in range(MAX_PLAN_ATTEMPTS):
            if self._cancel_event.is_set():
                _USER_LOG_QUEUE.put('[ROBOT] Instruction cancelled.')
                return

            raw = self.planner.plan(user_instruction, error_message=error_message)
            print_t(f"[P] Plan (attempt {attempt + 1}/{MAX_PLAN_ATTEMPTS}): {raw}")

            try:
                plan_obj = extract_plan_json(raw)
                program_str = plan_obj["plan"]
                code = validate_plan(program_str, self.robot.skillset.skills.keys())
            except PlanValidationError as e:
                # Bounded, *corrective* retry: feed the rejection reason back so the
                # model can fix it rather than (likely) repeating the same output.
                print_t(f"[P] Rejected plan: {e}")
                error_message = [str(e)]
                continue

            self._run_validated_plan(code)
            return

        _USER_LOG_QUEUE.put(
            '[ROBOT] Sorry, I could not produce a valid plan for that request.'
        )

    def _run_validated_plan(self, code):
        policy = PlanPolicy(
            cancel_event=self._cancel_event,
            deadline=time.monotonic() + self.plan_timeout,
        )
        self.robot.set_policy(policy)
        namespace = SkillNamespace(self.robot)
        try:
            exec(code, namespace)
        except PlanCancelled as e:
            print_t(f"[P] Plan stopped: {e}")
            _USER_LOG_QUEUE.put(f'[ROBOT] Plan stopped ({e}).')
            self.robot.stop_motion()
        except PlanLimitExceeded as e:
            print_t(f"[P] Plan limit exceeded: {e}")
            _USER_LOG_QUEUE.put(f'[ROBOT] Plan stopped: {e}.')
            self.robot.stop_motion()
        except Exception as e:
            print_t(f"[P] Error executing plan: {e}")
            print_t(f"[P] Traceback: {traceback.format_exc()}")
            _USER_LOG_QUEUE.put(f'[ROBOT] Plan failed: {e}.')
            self.robot.stop_motion()
        finally:
            self.robot.clear_policy()
