#!/usr/bin/env python3
"""eval_local.py stops itself before it can take the Mac it runs on down.

Measured 2026-11-14 on the Mini (16 GB, 4 GB of disk free): held-out v3 at its 65,536 window ran the Mac out of
swap at 1:15 PM ("no paging space"); at 2:21 PM macOS killed system services by the dozen, the login session reset,
and every LaunchAgent on the Mini stopped until a password was typed. The run scored nothing (12 of 12 skipped).

  1. the readers parse the real sysctl and memory_pressure lines (swap in M and in G, the free percentage);
  2. the verdict trips on a critical kernel pressure level and on a free percentage under the floor, and not on a
     healthy reading;
  3. the disk preflight refuses a disk with no room to swap into above a 16,384 window, and leaves the 16,384
     runs that held on the Mini alone;
  4. the watchdog, fed a critical reading, unloads the model and stops the run with exit 7 and a written reason;
     fed a healthy one, it does neither;
  5. when the unload fails (a starved Ollama that never answers), the runner is still killed by pid and the run
     still stops; the pid finder picks only the runner serving this exact model;
  6. memory is read from footprint, which counts the GPU memory resident size misses (9B: 0.15 GB rss, 7.13 GB).
Live proof against a real runner: live_guard_check.py, on a Mac with Ollama.
No model runs. Prints PASS or FAIL per check; exit 1 on any FAIL.

  python3 recipe/test_eval_local_guard.py
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_local as e  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name)
    fails += 0 if ok else 1


check("swap free parses in M", e.parse_swap_free_mb(
    "vm.swapusage: total = 1536.00M  used = 176.06M  free = 1359.94M  (encrypted)") == 1359.94)
check("swap free parses in G", e.parse_swap_free_mb(
    "vm.swapusage: total = 4.00G  used = 3.50G  free = 0.50G  (encrypted)") == 512.0)
check("free percentage parses", e.parse_free_pct(
    "The system has 17179869184 (4194304 pages with a page size of 4096).\n...\n"
    "System-wide memory free percentage: 83%") == 83)
check("an unreadable line reads as None, never as healthy", e.parse_free_pct("garbage") is None)

check("healthy reading passes", e.mem_verdict(pressure_level=1, free_pct=60) is None)
check("critical kernel pressure trips", e.mem_verdict(pressure_level=4, free_pct=60) is not None)
check("free percentage under the floor trips", e.mem_verdict(pressure_level=1, free_pct=e.MEM_FREE_PCT_MIN - 1) is not None)
check("unknown readings do not trip by themselves", e.mem_verdict(pressure_level=None, free_pct=None) is None)

check("disk with 4 GB free is refused at held-out v3's window", e.disk_verdict(4.2, 65536) is not None)
check("disk with room passes", e.disk_verdict(e.DISK_FREE_GB_MIN + 1, 65536) is None)
check("a 16,384 window run is not blocked by the disk floor (the Mini's small-tier gates)",
      e.disk_verdict(4.2, 16384) is None)

calls = {"unload": 0, "exit": None}
run = tempfile.mkdtemp()
abort_path = os.path.join(run, "x.abort.json")


def fake_exit(code):
    calls["exit"] = code
    raise SystemExit(code)


try:
    e.guard_tick(read=lambda: (4, 50), unload=lambda: calls.__setitem__("unload", calls["unload"] + 1),
                 abort_path=abort_path, exit=fake_exit)
except SystemExit:
    pass
check("critical reading unloads the model", calls["unload"] == 1)
check("critical reading stops the run with exit 7", calls["exit"] == 7)
check("critical reading writes the reason", os.path.exists(abort_path) and "pressure" in json.load(open(abort_path))["why"])

e.ABORTING.clear()
calls = {"unload": 0, "exit": None}
e.guard_tick(read=lambda: (1, 70), unload=lambda: calls.__setitem__("unload", 1),
             abort_path=os.path.join(run, "y.abort.json"), exit=fake_exit)
check("healthy reading leaves the run alone", calls == {"unload": 0, "exit": None} and not e.ABORTING.is_set())

calls = {"kill": 0, "exit": None}
try:
    e.guard_tick(read=lambda: (4, 50), unload=lambda: (_ for _ in ()).throw(TimeoutError("timed out")),
                 abort_path=os.path.join(run, "z.abort.json"), exit=fake_exit,
                 kill=lambda: (calls.__setitem__("kill", calls["kill"] + 1), [4242])[1])
except SystemExit:
    pass
z = json.load(open(os.path.join(run, "z.abort.json")))
check("a failed unload still kills the runner", calls["kill"] == 1 and z["killed"] == [4242])
check("a failed unload still stops the run with exit 7", calls["exit"] == 7)
check("a failed unload is written down, not hidden", z["unload"].startswith("failed"))

PS = """  101 /opt/homebrew/opt/ollama/bin/ollama serve
49207 /opt/homebrew/Cellar/ollama/0.40.8/libexec/ollama runner --model gruxai.com/grux/grux-local:9b --port 62645
49300 /opt/homebrew/Cellar/ollama/0.40.8/libexec/ollama runner --model gruxai.com/grux/grux-local:2b --port 62700
49400 /usr/bin/python3 eval_local.py --model gruxai.com/grux/grux-local:9b
"""
check("pid finder picks the runner for this exact model", e.parse_runner_pids(PS, "gruxai.com/grux/grux-local:9b") == [49207])
check("pid finder never picks the server or the eval itself", e.parse_runner_pids(PS, "gruxai.com/grux/grux-local:2") == [])

check("footprint parses in MB", e.parse_footprint_gb("    phys_footprint: 7128 MB\n    phys_footprint_peak: 7647 MB") == 7.13)
check("footprint parses in GB", e.parse_footprint_gb("    phys_footprint: 12.5 GB") == 12.5)
check("no footprint reads as None, never as zero", e.parse_footprint_gb("footprint: command not found") is None)

calls = {"exit": None}
try:
    e.guard_tick(read=lambda: (4, 50), unload=lambda: None, abort_path="/nonexistent-dir/x.abort.json",
                 exit=fake_exit, kill=lambda: [])
except SystemExit:
    pass
check("an abort file that cannot be written (full disk) still stops the run with exit 7", calls["exit"] == 7)

def boom():
    raise ValueError("unreadable")
try:
    e.guard_step("no-such-model", os.path.join(run, "s.abort.json"), read=boom)
    check("a reader that throws skips the tick instead of killing the guard thread", True)
except Exception:
    check("a reader that throws skips the tick instead of killing the guard thread", False)

rp = os.path.join(run, "r.jsonl")
open(rp, "w").write('{"case": "a"}\n{"case": "b"}\n{"case": "c", "steps": [1, 2')
check("a row cut off by an abort is skipped on resume", e.load_done(rp) == {"a", "b"})
check("the next row starts on its own line", open(rp).read().endswith("\n"))

check("footprint's own high-water mark parses",
      e.parse_footprint_gb("    phys_footprint: 7128 MB\n    phys_footprint_peak: 7647 MB", "phys_footprint_peak") == 7.65)
check("a malformed number reads as None, never crashes", e.parse_footprint_gb("phys_footprint: 7.1.2 MB") is None)

order = []
e.ABORTING.clear()
e.PEAK["gb"] = 9.9
try:
    e.guard_tick(read=lambda: (4, 50), unload=lambda: order.append("unload"), abort_path=os.path.join(run, "o.abort.json"),
                 exit=fake_exit, kill=lambda: (order.append("kill"), [7])[1])
except SystemExit:
    pass
check("the runner is killed before Ollama is asked (a starved Ollama cannot delay the kill)", order == ["kill", "unload"])
check("the abort file records the run's peak", json.load(open(os.path.join(run, "o.abort.json")))["peak_runner_footprint_gb"] == 9.9)
try:
    e.post("/api/chat", {})
    check("after a breach no request can reload the model", False)
except RuntimeError:
    check("after a breach no request can reload the model", True)
held = {"exit": None}
e.hold_if_aborting(limit_s=0.3, exit=lambda code: held.__setitem__("exit", code))
check("a guard that never reaches its exit is backstopped: the run stops with exit 7", held["exit"] == 7)
e.ABORTING.clear()
t0 = __import__("time").time()
e.hold_if_aborting(limit_s=5)
check("with no breach, writing a row is never delayed", __import__("time").time() - t0 < 0.1)

import time as _t
e.BEAT["t"] = _t.time()
check("a guard that just ticked counts as alive", e.guard_alive())
e.BEAT["t"] = _t.time() - e.GUARD_STALE_S - 1
check("a guard silent past the limit stops the run", not e.guard_alive())

print("SELF-TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
