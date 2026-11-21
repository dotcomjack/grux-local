#!/usr/bin/env python3
"""A GPU shortage is waited out, never given up on in minutes (capacity.py, train_run.box_loop, train_run.exit_code).

Measured 2026-11-20: a 4B and a 9B run were each reclaimed three times overnight, then got "No spot GPU capacity" on
five creates in about 3 minutes and quit with exit 0; capacity came back about 7 hours later.

  1. CapacityWait doubles its gap from FIRST_GAP up to MOST_GAP, stops at its deadline, and starts over after reset;
  2. is_capacity names the provider's shortage error and not a bad request;
  3. box_loop: 20 shortages then a box that runs ends "done", with every shortage settled at 0 hours;
  4. box_loop: a shortage that outlasts --capacity-wait-hours ends "no capacity" with a STOP line, $0 spent;
  5. box_loop: reclaimed boxes still stop at --max-boxes; a non-capacity create error still uses a box;
  6. exit_code: 6 for no capacity, 7 for out of boxes, 0 otherwise.
No box, no meter, no model: everything here is a stub. Prints PASS or FAIL per check; exit 1 on any FAIL.

  python3 recipe/s2/test_capacity.py
"""
import argparse
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
try:
    import daytona  # noqa: F401
except ImportError:   # the box SDK is not needed: no box is created here
    sys.modules["daytona"] = types.SimpleNamespace(**{n: object for n in (
        "CreateSandboxFromImageParams", "Daytona", "GpuType", "Image", "ListSandboxesQuery", "Resources",
        "SessionExecuteRequest")})
import capacity  # noqa: E402
import train_run as t  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


class Clock:
    def __init__(self):
        self.now, self.slept = 1000.0, []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


class NoSpot(Exception):
    pass


SHORT = NoSpot("Failed to create sandbox: No spot GPU capacity is available for this request. Please try again later.")
SHORT.__class__.__name__ = "DaytonaServiceUnavailableError"

# --------------------------------------------------------------------------- 1. the wait
c = Clock()
w = capacity.CapacityWait(1.0, first=30, most=600, clock=c, sleep=c.sleep)
oks = [w.wait() for _ in range(6)]
check("the gap doubles from 30 s and holds at 600 s", c.slept == [30, 60, 120, 240, 480, 600], str(c.slept))
check("... and every wait inside the hour says try again", all(oks))
while w.wait():
    pass
check("the wait stops at its deadline, never past it", abs(c.now - 1000.0 - 3600) < 1e-6, f"{c.now - 1000.0}")
w.reset()
check("reset starts the next shortage from 30 s with a fresh deadline", w.wait() and c.slept[-1] == 30, str(c.slept[-3:]))

# --------------------------------------------------------------------------- 2. which errors are a shortage
check("the provider's no-spot error is a shortage", capacity.is_capacity(SHORT))
check("a bad request is not a shortage", not capacity.is_capacity(ValueError("image not found: grux-train:9")))


# --------------------------------------------------------------------------- 3 to 5. the box loop
def loop(plan, wait_hours=12.0, max_boxes=8, meter=False, refuse_after=None, refuse_settle=False):
    """Run box_loop over `plan`: each item is an Exception to raise from the box or a (status, cost) to return.
    The box clock moves 0.5 h on every read, so a settle that used it instead of 0 reads 0.5000. The meter refuses
    every reserve after `refuse_after`."""
    run = tempfile.mkdtemp()
    a = argparse.Namespace(cap=100.0, meter="fake" if meter else "", gpu="H200", meter_lane="test", charter="",
                           capacity_wait_hours=wait_hours, max_boxes=max_boxes)
    calls, settles = [], []
    loop.settles = settles   # readable even when box_loop raises
    plan = list(plan)

    def box(*x):
        calls.append(1)
        step = plan.pop(0) if plan else ("done", 1.0)
        if isinstance(step, BaseException):
            raise step
        return step

    reserves = []

    def fake_meter(args):   # the real meter's rule (daytona_spend.check_hours): hours must be above 0
        if args[0] == "settle":
            settles.append(float(args[3]))
            if float(args[3]) <= 0 or refuse_settle:
                return types.SimpleNamespace(returncode=1, stdout="", stderr="hours must be a finite number above 0")
        if args[0] == "reserve":
            reserves.append(1)
            if refuse_after is not None and len(reserves) > refuse_after:
                return types.SimpleNamespace(returncode=3, stdout="", stderr="week cap reached")
        return types.SimpleNamespace(returncode=0, stdout=f"rid-{len(calls)}\n", stderr="")

    box_t = [0.0]

    def box_clock():
        box_t[0] += 1800.0
        return box_t[0]

    ck = Clock()
    waiter = capacity.CapacityWait(wait_hours, clock=ck, sleep=ck.sleep)
    saved = t.time.sleep
    t.time.sleep = lambda s: None
    try:
        status, spent = t.box_loop(a, run, "grux-m", "base-m", 0.0, box=box, meter=fake_meter, waiter=waiter,
                                   clock=box_clock)
    finally:
        t.time.sleep = saved
    loop.spent_json = json.load(open(os.path.join(run, "spent.json"))) if os.path.exists(os.path.join(run, "spent.json")) else None
    return status, spent, len(calls), settles, open(os.path.join(run, "run.log")).read()


st, spent, n, settles, lg = loop([SHORT] * 20 + [("done", 4.0)], meter=True)
check("20 shortages then a box that runs: done (the old loop quit after 8)", st == "done" and n == 21, f"{st} {n}")
check("... every shortage settled at the meter's floor (0.0001 h, which it accepts), the box at its real time",
      settles[:20] == [0.0001] * 20 and len(settles) == 21 and settles[20] == 0.5, f"{settles[:3]} {settles[20:]}")
check("... and $0 spent on the shortages", abs(spent - 4.0) < 1e-9, str(spent))

st, spent, n, settles, lg = loop([SHORT] * 10000, wait_hours=1.0)
check("a shortage past --capacity-wait-hours ends no capacity", st.startswith("no capacity"), st)
check("... with a STOP line in run.log and $0 spent", "STOP: no capacity" in lg and spent == 0.0, lg[-200:])

st, spent, n, settles, lg = loop([("evicted", 1.0)] * 20, max_boxes=8)
check("reclaimed boxes still stop at --max-boxes", st == "evicted" and n == 8, f"{st} {n}")

st, spent, n, settles, lg = loop([ValueError("image not found")] * 20, max_boxes=3)
check("a create error that is not a shortage still uses a box", n == 3 and st == "evicted", f"{st} {n}")

st, spent, n, settles, lg = loop([SHORT, SHORT, ("evicted", 1.0), SHORT, ("done", 1.0)], max_boxes=2)
check("shortages around a reclaimed box do not use boxes", st == "done" and n == 5, f"{st} {n}")

CREATED = NoSpot("503 Service Unavailable from the toolbox proxy")
CREATED.box_created = True   # train_box marks an error raised after the create returned
check("an error after the box was created is not a shortage", not capacity.is_capacity(CREATED))
st, spent, n, settles, lg = loop([CREATED] * 20, max_boxes=3, meter=True)
check("... it uses a box and settles its real time", n == 3 and settles == [0.5, 0.5, 0.5], f"{n} {settles}")

st, spent, n, settles, lg = loop([SHORT] * 50, meter=True, refuse_after=4)
check("a meter refusal in the middle of a wait stops the run, not 0", st == "meter refused" and t.exit_code(st) == 8
      and n == 4 and settles == [0.0001] * 4, f"{st} {n} {settles}")

st, spent, n, settles, lg = loop([SHORT] * 50, meter=True, refuse_settle=True)
check("a settle the meter refuses stops the run at once (no new reservation)", st == "settle refused"
      and t.exit_code(st) == 8 and n == 1 and "STOP: the spend meter refused the settle" in lg, f"{st} {n}")
st, spent, n, settles, lg = loop([("evicted", 20.0)], meter=True, refuse_settle=True)
check("a refused settle still writes the box's cost to spent.json (the next run's --cap starts right)",
      st == "settle refused" and loop.spent_json == {"spent": 20.0}, f"{st} {loop.spent_json}")

try:
    loop([KeyboardInterrupt()], meter=True)
    ki = "not raised"
except KeyboardInterrupt:
    ki = "raised"
check("Ctrl-C mid-box settles its real time, then is re-raised (no reservation left open)",
      ki == "raised" and loop.settles == [0.5], f"{ki} {loop.settles}")
check("no settle is ever sent below the floor", capacity.settle_hours(0.0, False) == capacity.SETTLE_FLOOR_HOURS
      and capacity.settle_hours(0.0, True) == capacity.SETTLE_FLOOR_HOURS and capacity.settle_hours(2.0, False) == 2.0)

big = capacity.CapacityWait(1e9, first=30, most=600, clock=Clock(), sleep=lambda s: None)
big.misses = 5000
check("the gap stays at its cap after thousands of misses (no overflow)", big.wait() and big.misses == 5001)

# --------------------------------------------------------------------------- 6. exit codes
check("exit 0 only for done: 6 no capacity, 7 out of boxes, 8 anything else unfinished",
      [t.exit_code(x) for x in ("done", "no capacity: no box came up in 12 h", "evicted", "meter refused",
                                "not started", "failed: boot")] == [0, 6, 7, 8, 8, 8])

print("SELF-TEST", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
sys.exit(1 if FAILS else 0)
