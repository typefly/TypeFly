# -*- coding: utf-8 -*-
from PIL import Image
import queue, io, base64, ast, signal
from typing import Optional
import threading
import json
import builtins

from .yolo_client import YoloClient
from .robot_wrapper import RobotWrapper
from .llm_planner import LLMPlanner
from .utils import print_t
from .robot_info import RobotInfo

_USER_LOG_QUEUE = queue.Queue()

# ✅ FIX #5: Signal خاص لـ re_plan
class RePlanSignal(Exception):
    """يُرمى عندما يستدعي الـ LLM re_plan()"""
    pass

# ✅ FIX #4: حد أقصى لوقت التنفيذ
PLAN_EXECUTION_TIMEOUT = 120  # ثانية — كافية للبحث الدوراني الكامل (12 دوران × ~4s)

# ✅ FIX #4: builtins آمنة فقط
SAFE_BUILTINS = {
    'True': True, 'False': False, 'None': None,
    'int': int, 'float': float, 'str': str, 'bool': bool,
    'len': len, 'range': range, 'print': print,
    'list': list, 'dict': dict, 'tuple': tuple,
    'min': min, 'max': max, 'abs': abs, 'round': round,
}


class LLMController():
    def __init__(self, robot_info: RobotInfo):
        self.controller_func = [self._user_log, self._probe]
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
        elif robot_info.robot_type == "pod":
            from .platforms.pod_wrapper import PodWrapper
            self.robot = PodWrapper(robot_info)
        elif robot_info.robot_type == "tello_sim":
            from .platforms.tello_wrapper_janah import TelloSimWrapper
            self.robot = TelloSimWrapper(robot_info)
        elif robot_info.robot_type == "airsim":
            from .platforms.airsim_platform import AirSimDronePlatform
            self.robot = AirSimDronePlatform(robot_info)
            print(f"[LLM Controller] ✅ AirSim platform initialized")

        self.planner = LLMPlanner(self.robot)
        self.current_plan_loop_thread = None
        self._stop_flag = threading.Event()  # يُستخدم لإيقاف plan شغالة

        self.missing_child_info = {
            'name': '', 'age': '', 'clothing_color': '',
            'last_location': '', 'description': '', 'photo_path': ''
        }

    def _user_log(self, msg) -> bool:
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

    def stop_controller(self):
        self.robot.stop()

    def fetch_robot_pov(self, overlay: bool = True) -> Optional[Image.Image]:
        image = self.robot.obs.image
        yolo_results = self.robot.obs.image_process_result.get("yolo", [])
        if overlay:
            YoloClient.plot_results_ps(image, yolo_results)
        return image

    def _validate_plan_ast(self, program_str: str) -> tuple[bool, str]:
        """
        AST validation قبل exec():
        - يرفض: import, open, exec, eval, __
        - يرفض: while True / while 1 (infinite loops)
        - يرفض: تعشيش أعمق من MAX_DEPTH
        - يرفض: أكثر من MAX_CALLS استدعاء للـ skills
        """
        FORBIDDEN = {'import', 'open', 'exec', 'eval', '__import__',
                     'compile', 'globals', 'locals', 'vars', 'getattr',
                     'setattr', 'delattr', 'hasattr', 'type', 'object'}
        MAX_DEPTH = 4    # حد التعشيش — SAR plans خطية عادةً depth=1
        MAX_CALLS = 50   # حد عدد skill calls في plan واحدة

        def _get_depth(node, current=0):
            """احسب أعمق مستوى تعشيش في الـ AST"""
            if not hasattr(node, '_fields'):
                return current
            max_child = current
            for child in ast.iter_child_nodes(node):
                max_child = max(max_child, _get_depth(child, current + 1))
            return max_child

        def _is_infinite_while(node) -> bool:
            """اكتشف while True / while 1"""
            if not isinstance(node, ast.While):
                return False
            test = node.test
            # while True
            if isinstance(test, ast.Constant) and test.value is True:
                return True
            # while 1
            if isinstance(test, ast.Constant) and test.value == 1:
                return True
            # while True (Python <3.8 NameConstant)
            if isinstance(test, ast.Name) and test.id == 'True':
                return True
            return False

        try:
            tree = ast.parse(program_str)

            call_count = 0
            for node in ast.walk(tree):
                # منع الـ imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    return False, "❌ import غير مسموح في الـ plan"

                # منع while True نهائياً
                if _is_infinite_while(node):
                    return False, "❌ while True/while 1 ممنوع — استخدم scan() أو re_plan()"

                # منع استدعاء دوال خطرة
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN:
                        return False, f"❌ {node.func.id}() غير مسموح"
                    call_count += 1

                # منع الـ dunder attributes
                if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
                    return False, f"❌ {node.attr} غير مسموح"

            # step counter
            if call_count > MAX_CALLS:
                return False, f"❌ عدد الـ calls ({call_count}) يتجاوز الحد ({MAX_CALLS})"

            # max depth
            depth = _get_depth(tree)
            if depth > MAX_DEPTH:
                return False, f"❌ تعشيش الكود ({depth}) يتجاوز الحد ({MAX_DEPTH})"

            return True, "OK"
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

    def _execute_with_timeout(self, program_str: str, exec_namespace: dict) -> Optional[str]:
        """
        تنفيذ مع timeout + stop_flag
        يرجع error message أو None إذا نجح
        """
        error_result = [None]
        stop = self._stop_flag  # reference للـ flag الحالي

        def _run():
            try:
                exec(program_str, exec_namespace)
            except RePlanSignal:
                error_result[0] = '__replan__'
            except Exception as e:
                import traceback
                error_result[0] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        # انتظر مع فحص stop_flag كل 0.5 ثانية
        elapsed = 0
        while t.is_alive() and elapsed < PLAN_EXECUTION_TIMEOUT:
            if stop.is_set():
                print_t("[P] ⛔ Execution interrupted by stop_flag")
                return "__stopped__"
            t.join(timeout=0.5)
            elapsed += 0.5

        if t.is_alive():
            return f"Timeout: plan exceeded {PLAN_EXECUTION_TIMEOUT}s"

        return error_result[0]

    def plan_loop(self, user_instruction: str):
        """
        plan_loop مع stop_flag + re_plan + error feedback
        """
        MAX_REPLANS = 3
        error_history = []
        replan_count = 0

        while replan_count <= MAX_REPLANS:

            # تحقق من إشارة الإيقاف قبل كل محاولة
            if self._stop_flag.is_set():
                print_t("[P] ⛔ Plan stopped by new instruction")
                return

            plan_raw = self.planner.plan(
                user_instruction,
                error_message=error_history if error_history else None,
                missing_child_info=self.missing_child_info
            )
            print_t(f"[P] Plan (attempt {replan_count+1}): {plan_raw}")

            if self._stop_flag.is_set():
                print_t("[P] ⛔ Plan stopped after LLM response")
                return

            # نظّف الـ JSON
            if '```json' in plan_raw:
                plan_raw = plan_raw.split('```json')[1].split('```')[0]

            try:
                plan = json.loads(plan_raw)
            except json.JSONDecodeError as e:
                print_t(f"[P] Invalid JSON: {e}")
                error_history.append(f"JSON parse error: {e}")
                replan_count += 1
                continue

            program_str = plan.get('plan', '')
            if not program_str:
                print_t("[P] Empty plan")
                break

            valid, ast_msg = self._validate_plan_ast(program_str)
            if not valid:
                print_t(f"[P] AST validation failed: {ast_msg}")
                error_history.append(f"AST validation failed: {ast_msg}")
                replan_count += 1
                continue

            exec_namespace = self._build_namespace()
            error = self._execute_with_timeout(program_str, exec_namespace)

            if error == '__replan__':
                print_t(f"[P] re_plan() called — replanning ({replan_count+1}/{MAX_REPLANS})")
                error_history.append("re_plan() was called by the plan")
                replan_count += 1
                continue

            elif error == '__stopped__':
                print_t("[P] ⛔ Plan was stopped")
                return

            elif error:
                print_t(f"[P] Execution error: {error}")
                error_history.append(error)
                replan_count += 1
                continue

            else:
                break

        if replan_count > MAX_REPLANS:
            print_t(f"[P] Max replans ({MAX_REPLANS}) reached")
            _USER_LOG_QUEUE.put(f"[ROBOT] تعذّر تنفيذ الخطة بعد {MAX_REPLANS} محاولات")

        _USER_LOG_QUEUE.put('#end')

    def _build_namespace(self) -> dict:
        """
        ✅ FIX #4 #5: بناء namespace آمن مع re_plan حقيقية
        """
        robot = self.robot

        class SafeNamespace(dict):
            def __init__(self, robot):
                super().__init__()
                self._robot = robot
                # أضف كل الـ skills
                for skill_name in robot.skillset.skills.keys():
                    self[skill_name] = robot.skillset.get_skill(skill_name)
                # ✅ FIX #4: builtins محدودة فقط
                self['__builtins__'] = SAFE_BUILTINS
                # ✅ FIX #5: re_plan ترمي exception حقيقي
                self['re_plan'] = lambda: (_ for _ in ()).throw(RePlanSignal())

            def __getitem__(self, key):
                if key in self._robot.skillset.skills:
                    return self._robot.skillset.get_skill(key)
                if key in self:
                    return super().__getitem__(key)
                if key in SAFE_BUILTINS:
                    return SAFE_BUILTINS[key]
                raise NameError(f"name '{key}' is not defined")

        return SafeNamespace(robot)

    def put_instruction(self, user_instruction: str):
        # أوقف أي plan شغالة أولاً
        if self.current_plan_loop_thread and self.current_plan_loop_thread.is_alive():
            print_t("[P] ⛔ إيقاف الخطة السابقة...")
            self._stop_flag.set()
            self.current_plan_loop_thread.join(timeout=3.0)

        # reset الـ flag للأمر الجديد
        self._stop_flag = threading.Event()

        self.current_plan_loop_thread = threading.Thread(
            target=self.plan_loop,
            args=(user_instruction,),
            daemon=True
        )
        self.current_plan_loop_thread.start()

    def set_missing_child_info(self, info: dict):
        """✅ FIX #10: يُستدعى من webui بعد upload"""
        self.missing_child_info.update(info)
        print_t(f"[SAR] 📋 Child info updated: {info.get('name')}, {info.get('age')}y, {info.get('clothing_color')}")
