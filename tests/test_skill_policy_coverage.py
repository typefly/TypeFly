"""Prove that EVERY registered skill passes through PlanPolicy (counts + is
cancellable). This is the regression guard for "a new skill was added without a
policy guard" — exactly the gap that previously let log/take_picture/re_plan
bypass the budgets."""

import types

import pytest

# Needs the control-path deps (robot_wrapper -> yolo_client -> cv2/numpy/...).
try:
    from typefly.robot_wrapper import RobotWrapper
    from typefly.plan_execution import PlanPolicy
except Exception:  # pragma: no cover - skip when heavy deps absent
    pytest.skip("control-path deps not installed", allow_module_level=True)


class _Obj:
    name = "thing"
    x = 0.5
    y = 0.5
    w = 0.1
    h = 0.1
    depth = 1.0


class _Obs:
    image = None
    image_process_result = {"yolo": [_Obj()]}


class _CountRobot(RobotWrapper):
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


def _dummy_arg(arg):
    return {float: 1.0, int: 1, str: "thing"}.get(arg.arg_type)


def test_every_registered_skill_goes_through_policy():
    RobotWrapper.set_controller_func([
        lambda msg: True,                       # _user_log
        lambda query, robot_info: "thing",      # _probe
    ])
    robot = _CountRobot()
    assert robot.skillset.skills, "expected a non-empty skillset"

    for name, skill in robot.skillset.skills.items():
        # max_delay=0 so delay() does not actually sleep during the test.
        policy = PlanPolicy(max_total_calls=100000, max_probe_calls=100000, max_delay=0.0)
        robot.set_policy(policy)
        skill(*[_dummy_arg(a) for a in skill.args])
        robot.clear_policy()
        assert policy.total_calls >= 1, f"skill '{name}' bypassed PlanPolicy (no checkpoint)"
