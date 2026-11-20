"""Waiting out a GPU shortage without giving up the run.

Measured 2026-11-20: two training runs were each reclaimed three times overnight, then got "No spot GPU capacity"
on five creates in about 3 minutes and quit, because a create that never came up used one of 8 attempts. Capacity
came back about 7 hours later. A create that never came up costs nothing, so it must not use an attempt: it waits
here instead, with a growing gap, until a wall-clock deadline. The spend bound stays --cap.

    w = CapacityWait(hours=12)
    if is_capacity(err) and w.wait(log):   # True: slept, try again; False: the deadline passed
"""
import time

FIRST_GAP, MOST_GAP = 30.0, 600.0   # seconds; tests shrink these
CAPACITY_WORDS = ("capacity", "serviceunavailable", "service unavailable", "no spot")


def is_capacity(e):
    """True when a create failed because the provider had no box to give (as opposed to a bad request). An error
    raised after the box was created (train_box marks it `box_created`) is a box that billed, never a shortage."""
    if getattr(e, "box_created", False):
        return False
    text = f"{type(e).__name__} {e}".lower()
    return any(w in text for w in CAPACITY_WORDS)


class CapacityWait:
    """A wall-clock budget for creates that never came up. The gap doubles from `first` to `most` seconds; `reset()`
    after a box that came up starts the next shortage from `first` again, with a fresh deadline."""

    def __init__(self, hours, first=None, most=None, clock=time.time, sleep=time.sleep):
        self.first = FIRST_GAP if first is None else first
        self.most = MOST_GAP if most is None else most
        self.hours, self.clock, self.sleep = hours, clock, sleep
        self.misses, self.started = 0, None

    def reset(self):
        self.misses, self.started = 0, None

    def waited_hours(self):
        return 0.0 if self.started is None else (self.clock() - self.started) / 3600

    def wait(self, log=lambda s: None):
        """Sleep before the next create. False (no sleep) once `hours` have passed since this shortage began."""
        now = self.clock()
        if self.started is None:
            self.started = now
        left = self.started + self.hours * 3600 - now
        if left <= 0:
            return False
        gap = min(self.most, self.first * (2 ** min(self.misses, 30)), left)   # capped: no overflow
        self.misses += 1
        log(f"no GPU capacity (miss {self.misses}, {self.waited_hours():.2f} h of {self.hours:g} h); "
            f"next create in {gap:.0f} s, $0 meanwhile")
        self.sleep(gap)
        return True
