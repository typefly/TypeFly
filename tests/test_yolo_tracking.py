"""Deterministic unit tests for the client-side object tracker (yolo_client).

`cc_to_ps` is a pure function of the detection list, so these tests drive it
directly with synthetic detection dicts and a fake clock -- no YOLO model, no
network, no event loop. Each test maps to one reliability scenario the tracker
must handle (Sn tags below).

Run with either:
    pytest tests/test_yolo_tracking.py
    python  tests/test_yolo_tracking.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typefly import yolo_client as yc
from typefly.yolo_client import (
    YoloClient, ObjectInfo,
    N_CONFIRM, M_MAX_MISS, T_MAX_AGE, BASE_GATE, MOTION_GATE_K, SIZE_W, EGO_HFOV_DEG,
)
from typefly.robot_info import RobotInfo


# --- helpers ----------------------------------------------------------------

class FakeClock:
    """Callable clock the tracker reads via yolo_client._wall_clock."""
    def __init__(self, t0=1000.0):
        self.t = t0

    def __call__(self):
        return self.t

    def tick(self, dt=0.25):
        self.t += dt
        return self.t


def new_client(clock):
    """Fresh tracking client with the fake clock installed."""
    yc._wall_clock = clock
    return YoloClient(RobotInfo("test-robot", "virtual", {}), enable_tracking=True)


def det(name, x, y, w=0.2, h=0.2, depth=None):
    """Build one raw YOLO detection dict (centre/size -> x1y1x2y2 box)."""
    box = {'x1': x - w / 2, 'y1': y - h / 2, 'x2': x + w / 2, 'y2': y + h / 2}
    d = {'name': name, 'confidence': 0.9, 'box': box}
    if depth is not None:
        d['depth'] = depth
    return d


def feed(client, clock, dets, dt=0.25):
    """Advance one frame and run the tracker."""
    clock.tick(dt)
    return client.cc_to_ps(dets)


def names(objs):
    return sorted(o.name for o in objs)


# --- tests ------------------------------------------------------------------

def test_id_switch_collapses_to_one():
    """S2: an upstream track-id switch must NOT spawn a duplicate ghost."""
    clock = FakeClock()
    c = new_client(clock)
    feed(c, clock, [det('apple_3', 0.5, 0.5)])                 # tentative
    out = feed(c, clock, [det('apple_3', 0.5, 0.5)])           # confirmed
    assert names(out) == ['apple'], out
    tid = c.tracks[0].id
    # Server "switches" the id; same physical apple, same place.
    out = feed(c, clock, [det('apple_7', 0.51, 0.5)])
    assert names(out) == ['apple'], f"id-switch produced a duplicate: {names(out)}"
    assert len([t for t in c.tracks if t.cls == 'apple']) == 1
    assert c.tracks[0].id == tid, "id-switch must reuse the same client track"


def test_dropout_coasts():
    """S1: a short detection dropout must NOT make the object disappear."""
    clock = FakeClock()
    c = new_client(clock)
    feed(c, clock, [det('apple', 0.5, 0.5)])
    feed(c, clock, [det('apple', 0.5, 0.5)])                   # confirmed
    out = feed(c, clock, [])                                   # miss 1
    assert names(out) == ['apple'], "object dropped on a single missed frame"
    out = feed(c, clock, [])                                   # miss 2
    assert names(out) == ['apple'], "object dropped after 2 missed frames"
    out = feed(c, clock, [det('apple', 0.52, 0.5)])           # re-detect
    assert names(out) == ['apple']
    assert len(c.tracks) == 1, "re-detection must re-snap, not spawn a new track"


def test_one_frame_false_positive_suppressed():
    """S4: a 1-frame spurious detection must never be emitted."""
    clock = FakeClock()
    c = new_client(clock)
    assert feed(c, clock, [det('apple', 0.5, 0.5)]) == []      # tentative, not emitted
    for _ in range(M_MAX_MISS + 2):
        assert feed(c, clock, []) == [], "1-frame false positive leaked into output"


def test_two_distinct_apples_not_merged():
    """S3: two genuinely distinct same-class instances stay separate."""
    for sep in (0.24, 0.15):                                   # 0.15 is just above BASE_GATE
        clock = FakeClock()
        c = new_client(clock)
        x1, x2 = 0.40 - sep / 2, 0.40 + sep / 2
        feed(c, clock, [det('apple', x1, 0.5), det('apple', x2, 0.5)])
        out = feed(c, clock, [det('apple', x1, 0.5), det('apple', x2, 0.5)])
        assert names(out) == ['apple', 'apple'], f"sep={sep} merged: {names(out)}"
        xs = sorted(o.x for o in out)
        assert abs(xs[0] - x1) < 0.05 and abs(xs[1] - x2) < 0.05


def test_departed_object_expires_by_time():
    """S5: an object that leaves must expire (wall-clock bound)."""
    clock = FakeClock()
    c = new_client(clock)
    feed(c, clock, [det('apple', 0.5, 0.5)])
    feed(c, clock, [det('apple', 0.5, 0.5)])                   # confirmed
    out = feed(c, clock, [], dt=T_MAX_AGE + 0.1)               # jump past T_MAX_AGE
    assert out == [], "object did not expire after T_MAX_AGE"
    assert c.tracks == []


def test_departed_object_expires_by_misses_on_fast_feed():
    """S5: on a fast feed, expiry is bounded by frame count, not wall-clock."""
    clock = FakeClock()
    c = new_client(clock)
    feed(c, clock, [det('apple', 0.5, 0.5)], dt=0.05)
    feed(c, clock, [det('apple', 0.5, 0.5)], dt=0.05)         # confirmed
    out = None
    for _ in range(M_MAX_MISS + 1):                           # all within 1.0s
        out = feed(c, clock, [], dt=0.05)
    assert out == [], "object did not expire by M_MAX_MISS on a fast feed"
    assert clock.t - 1000.0 < T_MAX_AGE                       # proves it was the miss bound


def test_fast_pan_no_drop_no_dup():
    """S6: a camera pan must not drop or duplicate the target (ego-motion comp)."""
    shift = 20.0 / EGO_HFOV_DEG                               # a 20deg left turn
    x0 = 0.40

    # ego-comp ON: tracker follows the panned detection as the SAME track.
    clock = FakeClock()
    c = new_client(clock)
    feed(c, clock, [det('apple', x0, 0.5)])
    feed(c, clock, [det('apple', x0, 0.5)])                   # confirmed
    c.note_ego_motion(d_yaw_deg=20.0)                         # rotate_left(20)
    out = feed(c, clock, [det('apple', x0 + shift, 0.5)])
    assert names(out) == ['apple'], f"pan produced wrong count: {names(out)}"
    assert abs(out[0].x - (x0 + shift)) < 0.08, "ego-comp failed to follow the pan"

    # CONTROL, ego-comp OFF: a fixed gate drops then DUPLICATES the target.
    clock = FakeClock()
    c = new_client(clock)
    feed(c, clock, [det('apple', x0, 0.5)])
    feed(c, clock, [det('apple', x0, 0.5)])                   # confirmed
    out = feed(c, clock, [det('apple', x0 + shift, 0.5)])     # no ego hint
    assert len(out) == 1 and out[0].x < 0.55, "control should see only a stale box"
    out = feed(c, clock, [det('apple', x0 + shift, 0.5)])
    assert len(out) == 2, "control must duplicate without ego-comp (demonstrates the fix)"


def test_depth_passthrough():
    """Depth is carried through, halved, and None when absent (go2 vs webcam)."""
    clock = FakeClock()
    c = new_client(clock)
    feed(c, clock, [det('apple', 0.5, 0.5, depth=2.0)])
    out = feed(c, clock, [det('apple', 0.5, 0.5, depth=2.0)])
    assert out[0].depth == 1.0, out[0].depth

    clock = FakeClock()
    c = new_client(clock)
    feed(c, clock, [det('apple', 0.5, 0.5)])
    out = feed(c, clock, [det('apple', 0.5, 0.5)])
    assert out[0].depth is None


def test_cumulative_confirm_on_entry():
    """A glitch on entry (detect->miss->detect) must still confirm (cumulative)."""
    clock = FakeClock()
    c = new_client(clock)
    assert feed(c, clock, [det('apple', 0.5, 0.5)]) == []      # hit 1, tentative
    assert feed(c, clock, []) == []                            # miss, still tentative
    out = feed(c, clock, [det('apple', 0.5, 0.5)])            # hit 2 -> confirmed
    assert names(out) == ['apple'], "cumulative hits must confirm despite a gap"


def test_passthrough_when_tracking_disabled():
    """enable_tracking=False returns detections directly (legacy path), names clean."""
    clock = FakeClock()
    yc._wall_clock = clock
    c = YoloClient(RobotInfo("test-robot", "virtual", {}), enable_tracking=False)
    out = c.cc_to_ps([det('apple_7', 0.5, 0.5), det('chair', 0.2, 0.3)])
    assert names(out) == ['apple', 'chair'], names(out)


def test_constants_pinned():
    """Guard the tuned defaults against silent regression."""
    assert (N_CONFIRM, M_MAX_MISS, T_MAX_AGE) == (2, 3, 1.0)
    assert (BASE_GATE, MOTION_GATE_K, SIZE_W, EGO_HFOV_DEG) == (0.12, 1.5, 0.25, 70.0)


# --- standalone runner (no pytest required) ---------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
