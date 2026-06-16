"""Shared test fixtures for the trust-boundary suite.

These tests deliberately exercise only the security boundary (plan_execution +
skillset), so they import no heavy deps (cv2/openai/ROS). FakeRobot mirrors the
RobotWrapper policy-guard pattern without being one.
"""

import os
import sys

import pytest

# Make the repo importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typefly.skillset import SkillSet  # noqa: E402
from typefly.plan_execution import SkillNamespace, validate_plan  # noqa: E402


class FakeRobot:
    """Minimal RobotWrapper stand-in: records skill calls and applies a policy."""

    def __init__(self):
        self.calls = []  # (skill_name, *args)
        self._policy = None
        self.skillset = SkillSet.get_common_skillset([
            (self.move_forward, "Move forward by a dist (m)"),
            (self.rotate_right, "Rotate right by a deg"),
            (self.delay, "Wait for seconds"),
            (self.probe, "Query the LLM for reasoning"),
            (self.is_visible, "Check if object is visible"),
            (self.log, "Print text to user"),
        ])

    # --- policy hooks (same semantics as RobotWrapper) ---
    def set_policy(self, policy):
        self._policy = policy

    def clear_policy(self):
        self._policy = None

    def _guard(self, kind, value=None):
        if self._policy is None:
            return value
        return self._policy.checkpoint(kind, value)

    # --- skills (each param needs a type annotation for SkillItem) ---
    def move_forward(self, dist: float):
        dist = self._guard("move", dist)
        self.calls.append(("move_forward", dist))

    def rotate_right(self, deg: float):
        deg = self._guard("rotate", deg)
        self.calls.append(("rotate_right", deg))

    def delay(self, sec: float):
        sec = self._guard("delay", sec)
        self.calls.append(("delay", sec))

    def probe(self, query: str):
        self._guard("probe")
        self.calls.append(("probe", query))
        return query

    def is_visible(self, object_name: str) -> bool:
        self.calls.append(("is_visible", object_name))
        return True

    def log(self, message: str):  # intentionally unguarded, like RobotWrapper.log
        self.calls.append(("log", message))


@pytest.fixture
def robot():
    return FakeRobot()


@pytest.fixture
def run_plan():
    """Validate + exec a plan against a robot, exactly as the controller does."""
    def _run(robot, program_str, policy=None):
        code = validate_plan(program_str, robot.skillset.skills.keys())
        if policy is not None:
            robot.set_policy(policy)
        try:
            exec(code, SkillNamespace(robot))
        finally:
            robot.clear_policy()
        return robot.calls

    return _run
