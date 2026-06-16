"""End-to-end: validate + exec a plan against a fake robot, exactly as the
controller does, asserting both functionality and the safety guarantees."""

import threading

import pytest

from typefly.plan_execution import PlanPolicy, PlanValidationError, PlanCancelled, PlanLimitExceeded


def test_benign_plan_executes(robot, run_plan):
    calls = run_plan(robot, "log('hello')\nmove_forward(1.0)")
    assert ("log", "hello") in calls
    assert ("move_forward", 1.0) in calls


def test_control_flow_executes(robot, run_plan):
    plan = (
        "if is_visible('person'):\n"
        "    log('seen')\n"
        "else:\n"
        "    log('not seen')"
    )
    calls = run_plan(robot, plan)
    assert ("is_visible", "person") in calls
    assert ("log", "seen") in calls  # FakeRobot.is_visible returns True


def test_malicious_plan_never_executes(robot, run_plan):
    with pytest.raises(PlanValidationError):
        run_plan(robot, "__import__('os').system('echo pwned')")
    assert robot.calls == []  # rejected before any skill ran


def test_reflection_escape_rejected(robot, run_plan):
    with pytest.raises(PlanValidationError):
        run_plan(robot, "log((1).__class__.__bases__)")
    assert robot.calls == []


def test_policy_clamps_runaway_move(robot, run_plan):
    calls = run_plan(robot, "move_forward(999)", policy=PlanPolicy(max_distance=5.0))
    assert calls == [("move_forward", 5.0)]


def test_bounded_loop_runs(robot, run_plan):
    calls = run_plan(robot, "for _ in range(3):\n    rotate_right(30)", policy=PlanPolicy())
    assert calls == [("rotate_right", 30.0)] * 3


def test_precancelled_plan_does_not_actuate(robot, run_plan):
    ev = threading.Event()
    ev.set()
    with pytest.raises(PlanCancelled):
        run_plan(robot, "move_forward(1.0)", policy=PlanPolicy(cancel_event=ev))
    assert robot.calls == []  # cancelled at the checkpoint, before the actuator


def test_total_call_budget_halts_plan(robot, run_plan):
    with pytest.raises(PlanLimitExceeded):
        run_plan(
            robot,
            "for _ in range(10):\n    rotate_right(10)",
            policy=PlanPolicy(max_total_calls=2),
        )
    # Only the budgeted number of actuator calls got through.
    assert len(robot.calls) == 2
