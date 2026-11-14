"""A spend meter around every box: reserve before the create, settle after with the real hours.

The meter is any CLI with `reserve --lane L --hours H --gpu G` (prints an id; exit 3 refuses) and
`settle <id> --hours H`. The box classes refuse to start without the id `reserve` returned, so a box that was never
paid for cannot exist, and each box gets its own reservation.

  with Reserved(meter, lane, hours, gpu) as rid:
      with OllamaBox(..., reservation=rid) as box: ...
"""
import os
import subprocess
import sys
import time

# Every GPU reservation names a swing charter (the owner's rule, 2026-11-14: never a swing on one or two jobs).
# The charter file comes from GRUX_SWING_CHARTER; the meter refuses without one (exit 5).
CHARTER = os.path.expanduser(os.environ.get("GRUX_SWING_CHARTER", ""))


class Refused(Exception):
    pass


def reserve(meter, lane, hours, gpu, purpose="train"):
    if not meter:
        raise Refused("no spend meter given (--meter); every box needs its own reservation, so none is created")
    r = subprocess.run([sys.executable, meter, "reserve", "--lane", lane, "--hours", f"{max(hours, 0.01):.3f}",
                        "--gpu", gpu.replace("-", ""), "--purpose", purpose]
                       + (["--charter", CHARTER] if CHARTER else []), capture_output=True, text=True)
    rid = r.stdout.strip()
    if r.returncode != 0 or not rid:
        raise Refused(f"the spend meter refused the box (exit {r.returncode}): {r.stderr.strip()[:200] or 'no id'}")
    return rid


def settle(meter, rid, hours):
    r = subprocess.run([sys.executable, meter, "settle", rid, "--hours", f"{hours:.4f}"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"settle failed: {r.stderr.strip()[:200]}"


class Reserved:
    """Reserve on enter; settle on exit with the hours actually held, whatever happened inside."""

    def __init__(self, meter, lane, hours, gpu, purpose="train"):
        self.meter, self.lane, self.hours, self.gpu, self.purpose = meter, lane, hours, gpu, purpose
        self.rid = None

    def __enter__(self):
        self.rid = reserve(self.meter, self.lane, self.hours, self.gpu, self.purpose)
        self.t0 = time.time()
        return self.rid

    def __exit__(self, *exc):
        self.settled = settle(self.meter, self.rid, (time.time() - self.t0) / 3600)
        return False
