#!/usr/bin/env python3
"""eval_local.py stops itself before it can take the Mac it runs on down.

Measured 2026-11-14 on the Mini (16 GB, 4 GB of disk free): held-out v3 at its 65,536 window ran the Mac out of
swap at 1:15 PM ("no paging space"); at 2:21 PM macOS killed system services by the dozen, the login session reset,
and every LaunchAgent on the Mini stopped until a password was typed. The run scored nothing (12 of 12 skipped).

  1. the readers parse the real sysctl and memory_pressure lines (swap in M and in G, the free percentage);
  2. the verdict trips on a critical kernel pressure level and on a free percentage under the floor, and not on a
     healthy reading;
  3. the disk preflight refuses a disk with no room to swap into;
  4. the watchdog, fed a critical reading, unloads the model and stops the run with exit 7 and a written reason;
     fed a healthy one, it does neither.
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

check("disk with 4 GB free is refused", e.disk_verdict(4.2) is not None)
check("disk with room passes", e.disk_verdict(e.DISK_FREE_GB_MIN + 1) is None)

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

calls = {"unload": 0, "exit": None}
e.guard_tick(read=lambda: (1, 70), unload=lambda: calls.__setitem__("unload", 1),
             abort_path=os.path.join(run, "y.abort.json"), exit=fake_exit)
check("healthy reading leaves the run alone", calls == {"unload": 0, "exit": None})

print("SELF-TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
