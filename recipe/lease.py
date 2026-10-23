"""A Daytona box that cannot outlive the process that made it.

Three layers, so cleanup is never a step anyone has to remember:

1. Server side, the one that survives kill -9, a crash or a sleeping laptop: the box is created
   with a short wall-clock TTL and a heartbeat thread re-arms it (`set_ttl` re-anchors from now).
   When the process stops beating, Daytona destroys the box within LEASE_MINUTES, in any state.
2. Process side: a normal end, an uncaught exception, SIGINT, SIGTERM and SIGHUP all delete the
   box at once (atexit plus signal handlers).
3. Backstop: a separate scheduled sweeper lists the org every 15 minutes and deletes any labelled
   box that has no TTL, is stopped, or is past its maximum age (not part of this recipe).

A box costs about $6.53 an hour on an H200 (GPU plus 16 vCPU and 192 GB RAM, measured
2026-10-22), so the worst a dead process can waste is about LEASE_MINUTES of that.
"""
import atexit
import signal
import sys
import threading

LEASE_MINUTES = 20
BEAT_SECONDS = 300

_live = {}            # sandbox id -> (daytona client, sandbox)
_lock = threading.Lock()
_installed = False


def _delete_all(reason):
    with _lock:
        items = list(_live.items())
        _live.clear()
    for sid, (d, sb) in items:
        try:
            d.delete(sb, wait=False, timeout=60)
            print(f"lease: deleted {sid[:8]} ({reason})", file=sys.stderr, flush=True)
        except Exception as e:  # the server-side TTL is still armed
            print(f"lease: delete {sid[:8]} failed ({e}); its TTL ends it", file=sys.stderr, flush=True)


def _install():
    """atexit always; signal handlers only from the main thread, the one place Python allows them.
    Measured 2026-10-22: a Lease made in a worker thread raised here and left its box unheld. A
    threaded caller runs `install()` once from the main thread first; the TTL holds either way."""
    global _installed
    if _installed:
        return
    atexit.register(_delete_all, "process exit")
    if threading.current_thread() is not threading.main_thread():
        print("lease: not on the main thread; signal handlers wait for install() (TTL still armed)",
              file=sys.stderr, flush=True)
        return
    _installed = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        prior = signal.getsignal(sig)

        def handler(signum, frame, prior=prior):
            _delete_all(signal.Signals(signum).name)
            if callable(prior):
                prior(signum, frame)
            raise SystemExit(128 + signum)
        signal.signal(sig, handler)


install = _install


class Lease:
    """Holds one sandbox: registers it for deletion on exit and keeps its TTL armed."""

    def __init__(self, daytona, sandbox, minutes=LEASE_MINUTES, beat=BEAT_SECONDS):
        self.d, self.sb, self.minutes, self.beat = daytona, sandbox, minutes, beat
        self._stop = threading.Event()
        _install()
        with _lock:
            _live[sandbox.id] = (daytona, sandbox)
        self.sb.set_ttl(self.minutes)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.beat):
            try:
                self.sb.set_ttl(self.minutes)
            except Exception as e:  # a missed beat is fine; the TTL still holds
                print(f"lease: heartbeat failed ({e})", file=sys.stderr, flush=True)

    def release(self):
        """Stop beating and delete now (normal end)."""
        self._stop.set()
        with _lock:
            _live.pop(self.sb.id, None)
        self.d.delete(self.sb, wait=True, timeout=180)
