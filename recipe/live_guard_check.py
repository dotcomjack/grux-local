#!/usr/bin/env python3
"""Live proof of eval_local's guard against a real Ollama runner. Run on a Mac with Ollama and nothing else loaded
(it refuses when /api/ps shows a model: the Mini's Ollama is shared with other loops).

Two runs, each in a child process so the guard's real os._exit is what ends it:
  unload: the floor is forced above 100%, so the real guard thread trips on its first look, asks Ollama to unload,
          kills the runner by pid and exits 7;
  kill:   Ollama's address is pointed at a closed port after the load, so the unload fails the way it would against
          a starved Ollama, and only the pid kill can free the memory.
Each run must end with exit 7, an abort file naming the reason, and the runner's pid gone.

  python3 recipe/live_guard_check.py --model gruxai.com/grux/grux-local:2b
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_local as e  # noqa: E402


def child(model, mode, abort_path):
    e.post("/api/generate", {"model": model, "prompt": "ok", "stream": False,
                             "options": {"num_predict": 2, "num_ctx": 4096}}, timeout=180)
    pids = e.runner_pids(model)
    print(json.dumps({"pids": pids, "footprint_gb": e.runner_footprint_gb(model)}), flush=True)
    e.MEM_FREE_PCT_MIN = 101
    if mode == "kill":
        e.BASE = "http://127.0.0.1:9"
    e.start_guard(model, abort_path)
    time.sleep(30)
    sys.exit(0)   # reaching here means the guard never fired


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--child", choices=["unload", "kill"])
    ap.add_argument("--abort")
    a = ap.parse_args()
    if a.child:
        return child(a.model, a.child, a.abort)
    if e.post_get("/api/ps").get("models"):
        sys.exit("REFUSED a model is already loaded; this Ollama is in use")
    fails = 0
    for mode in ("unload", "kill"):
        abort = os.path.join(tempfile.mkdtemp(), mode + ".abort.json")
        p = subprocess.run([sys.executable, __file__, "--model", a.model, "--child", mode, "--abort", abort],
                           capture_output=True, text=True, timeout=300)
        first = json.loads(p.stdout.splitlines()[0]) if p.stdout.strip() else {"pids": [], "footprint_gb": None}
        pids = first["pids"]
        time.sleep(2)
        got = json.load(open(abort)) if os.path.exists(abort) else {}
        checks = [
            ("exit 7", p.returncode == 7),
            ("a runner was loaded", len(pids) == 1),
            ("footprint read live", (first["footprint_gb"] or 0) > 1),
            ("abort file names the reason", "floor" in got.get("why", "")),
            ("runner pid gone", bool(pids) and not alive(pids[0])),
        ]
        if mode == "kill":
            checks += [("unload failed as staged", got.get("unload", "").startswith("failed")),
                       ("the pid kill freed it", got.get("killed") == pids)]
        for name, ok in checks:
            print(("PASS " if ok else "FAIL ") + mode + ": " + name)
            fails += 0 if ok else 1
        print(mode, json.dumps({"rc": p.returncode, "pids": pids, "footprint_gb": first["footprint_gb"],
                                "unload": got.get("unload"), "killed": got.get("killed")}))
    print("LIVE", "PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
