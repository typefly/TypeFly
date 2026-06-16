"""Comment 3: invalid-plan retries must be corrective — the rejection reason is
fed back into the next planning prompt."""

import pytest

try:
    from typefly.llm_planner import build_replan_feedback
except Exception:  # pragma: no cover - skip when heavy deps absent
    pytest.skip("planner deps not installed", allow_module_level=True)


def test_no_feedback_on_first_attempt():
    # Empty so the first attempt's prompt is unchanged.
    assert build_replan_feedback() == ""
    assert build_replan_feedback(None, None) == ""
    assert build_replan_feedback([], []) == ""


def test_feedback_carries_the_rejection_reason():
    fb = build_replan_feedback(["call to 'os' is not an allowed skill"])
    assert "os" in fb
    assert "rejected" in fb.lower()
    # It should also remind the model of the constraints it violated.
    assert "skill" in fb.lower()


def test_feedback_includes_execution_history():
    fb = build_replan_feedback(["bad"], ["moved forward 1m", "scan failed"])
    assert "scan failed" in fb
