import threading
import time

import pytest

from typefly.plan_execution import PlanPolicy, PlanCancelled, PlanLimitExceeded


def test_clamp_move():
    p = PlanPolicy(max_distance=5.0)
    assert p.checkpoint("move", 999) == 5.0
    assert p.checkpoint("move", -999) == -5.0
    assert p.checkpoint("move", 2.0) == 2.0


def test_clamp_rotate():
    p = PlanPolicy(max_angle=360.0)
    assert p.checkpoint("rotate", 99999) == 360.0
    assert p.checkpoint("rotate", -99999) == -360.0


def test_clamp_delay():
    p = PlanPolicy(max_delay=30.0)
    assert p.checkpoint("delay", 99999) == 30.0
    assert p.checkpoint("delay", -5) == 0.0


def test_total_call_limit():
    p = PlanPolicy(max_total_calls=3)
    for _ in range(3):
        p.checkpoint("vision")
    with pytest.raises(PlanLimitExceeded):
        p.checkpoint("vision")


def test_probe_limit():
    p = PlanPolicy(max_probe_calls=2, max_total_calls=100)
    p.checkpoint("probe")
    p.checkpoint("probe")
    with pytest.raises(PlanLimitExceeded):
        p.checkpoint("probe")


def test_cancel_event_trips_checkpoint_and_poll():
    ev = threading.Event()
    ev.set()
    p = PlanPolicy(cancel_event=ev)
    with pytest.raises(PlanCancelled):
        p.checkpoint("move", 1.0)
    with pytest.raises(PlanCancelled):
        p.poll()


def test_deadline_trips():
    p = PlanPolicy(deadline=time.monotonic() - 1.0)
    with pytest.raises(PlanCancelled):
        p.checkpoint("move", 1.0)


def test_non_numeric_value_passes_through():
    p = PlanPolicy()
    # A non-numeric arg is left for the skill to handle, not crashed on.
    assert p.checkpoint("move", "oops") == "oops"
