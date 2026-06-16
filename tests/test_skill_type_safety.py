"""Regression: a generated plan can chain skills so that a bool/number reaches
an object-name skill — e.g. `goto(scan('x'))`, or the documented
`obj = scan_description(...); goto(obj)` when probe answers a yes/no query with
'True'. That used to crash the plan thread with

    AttributeError: 'bool' object has no attribute 'strip'

inside get_obj_info. Object-name skills must degrade gracefully (treat a
non-string as "not in sight") instead of crashing, and scan_description must
honor its contract of returning the object *name* (a string) or False."""

import types

import pytest

# Needs the control-path deps (robot_wrapper -> yolo_client -> cv2/numpy/...).
try:
    from typefly.robot_wrapper import RobotWrapper
except Exception:  # pragma: no cover - skip when heavy deps absent
    pytest.skip("control-path deps not installed", allow_module_level=True)


class _Obj:
    name = "bottle"
    x = 0.5
    y = 0.5
    w = 0.1
    h = 0.1
    depth = 1.0


class _Obs:
    image = None
    image_process_result = {"yolo": [_Obj()]}


class _NoDepthObj(_Obj):
    # The YOLO webcam/petoi serving pipeline emits no depth key, so
    # ObjectInfo.depth is None for every detected object on the default robot.
    depth = None


class _NoDepthObs:
    image = None
    image_process_result = {"yolo": [_NoDepthObj()]}


class _FakeRobot(RobotWrapper):
    def __init__(self):
        super().__init__(types.SimpleNamespace(robot_id="t", robot_type="test"), _Obs())

    def start(self):
        return True

    def stop(self):
        return True

    def _move(self, dx, dy):
        pass

    def _rotate(self, deg):
        pass


def _make_robot(probe_reply="bottle"):
    """probe_reply is the *raw* string the probe LLM would emit (pre
    evaluate_value): 'True' -> bool True, 'bottle' -> the object name."""
    RobotWrapper.set_controller_func([
        lambda msg: True,                          # _user_log
        lambda query, robot_info: probe_reply,     # _probe
    ])
    return _FakeRobot()


@pytest.mark.parametrize("bad", [True, False, 1, 3.14, None])
def test_get_obj_info_tolerates_non_string(bad):
    assert _make_robot().get_obj_info(bad) is None


def test_is_visible_non_string_is_false():
    assert _make_robot().is_visible(True) is False


def test_goto_non_string_returns_false_without_crashing():
    # The exact regression: goto(True) used to raise AttributeError.
    assert _make_robot().goto(True) is False


def test_orienting_non_string_returns_false():
    assert _make_robot().orienting(True) is False


def test_object_x_non_string_does_not_crash():
    # Falls through to the "not in sight" message rather than crashing on regex.
    assert "not in sight" in str(_make_robot().object_x(True))


def test_scan_description_yes_no_answer_is_not_a_bool_target():
    # probe answers the yes/no form with 'True'; scan_description must NOT hand
    # that bool to goto -- it returns False, which goto handles gracefully.
    robot = _make_robot(probe_reply="True")
    result = robot.scan_description("Any edible object here?")
    assert result is False
    assert robot.goto(result) is False  # safe to feed back into goto


def test_scan_description_returns_named_object():
    robot = _make_robot(probe_reply="bottle")
    result = robot.scan_description("Any drinkable object here?")
    assert isinstance(result, str)
    # The named object is in view: goto reaches it and reports success.
    assert robot.goto(result) is True


def test_object_dist_in_sight_but_no_depth_does_not_crash():
    # Object IS found, but the default vision pipeline provides no depth (None).
    # object_dist used to do `None * 100` -> TypeError; it must degrade instead.
    robot = _make_robot()
    robot.obs = _NoDepthObs()
    result = robot.object_dist("bottle")
    assert isinstance(result, str)  # descriptive message, not a crash or bogus number


def test_object_dist_non_string_arg_is_not_amplified():
    # Chained: object_dist(is_visible('x')) -> object_dist(False). get_obj_info
    # returns None, so depth_info is the "not in sight" STRING; object_dist must
    # not return that string multiplied by 100 (a multi-KB "distance").
    robot = _make_robot()
    result = robot.object_dist(False)
    assert isinstance(result, str)
    assert len(result) < 200
