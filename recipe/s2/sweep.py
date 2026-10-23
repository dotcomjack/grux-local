#!/usr/bin/env python3
"""S2 sweep: one trained model per Mac memory tier, chosen from parallel bets (card
recipe/README.md). Each config is its own train_run.py process on its
own leased spot box.

Stateless, so it can be restarted at any time without touching a running run: every 15 s it reads
which runs have a live process (ps), which are done (status.json), and starts the next not-done,
not-live config in priority order while fewer than --width are live. A run that stops without
"done" is started again (it resumes from its own pulled checkpoint), at most --tries launches.

Width is the box count spot capacity actually holds: measured 2026-10-22 evening, 10 live runs lost
their newest boxes every few minutes; about 8 held.

  DAYTONA_API_KEY=<your key> .venv/bin/python s2/sweep.py \\
    --sweep s2/results/sweep-<stamp> --data s2/results/data-<stamp>/train.jsonl [--width 8] [--only 4b-r16-e1]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# model, $ cap per run, examples per forward, token ceiling per forward (9B smaller to fit an 80 GB H100),
# checkpoint every N steps (about 10 minutes of box time at the measured step times)
BASES = {"4b": ("Qwen/Qwen3.5-4B", 30.0, 8, 48000, 12), "9b": ("Qwen/Qwen3.5-9B", 45.0, 4, 24000, 12),
         "2b": ("Qwen/Qwen3.5-2B", 15.0, 8, 48000, 20)}
CONFIGS = [(size, rank, ep) for size in ("4b", "9b", "2b") for rank in (16, 64) for ep in (1, 2)]


def name(size, rank, ep):
    return f"{size}-r{rank}-e{ep}"


def live_runs(root):
    out = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True).stdout
    return {os.path.basename(l.split("--run ", 1)[1].split(" ", 1)[0].rstrip("/"))
            for l in out.splitlines() if "train_run.py --run " + root in l}


def done(root, n):
    st = os.path.join(root, n, "status.json")
    return os.path.exists(st) and json.load(open(st)).get("status") == "done"


def launches(root, n, add=False):
    p = os.path.join(root, n, "launches")
    k = int(open(p).read()) if os.path.exists(p) else 0
    if add:
        open(p, "w").write(str(k + 1))
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--only", default="", help="comma list of config names")
    a = ap.parse_args()
    root = os.path.abspath(a.sweep)
    os.makedirs(root, exist_ok=True)
    todo = [c for c in CONFIGS if not a.only or name(*c) in a.only.split(",")]
    json.dump({"configs": [name(*c) for c in todo], "data": os.path.abspath(a.data), "width": a.width,
               "bases": BASES}, open(os.path.join(root, "sweep.json"), "w"), indent=1)
    while True:
        live = live_runs(root)
        waiting = [c for c in todo if not done(root, name(*c)) and name(*c) not in live
                   and launches(root, name(*c)) < a.tries]
        if not live and not waiting:
            break
        for size, rank, ep in waiting:
            if len(live) >= a.width:
                break
            n = name(size, rank, ep)
            run = os.path.join(root, n)
            model, cap, micro, mtok, every = BASES[size]
            cmd = [sys.executable, os.path.join(HERE, "train_run.py"), "--run", run, "--model", model,
                   "--data", a.data, "--rank", str(rank), "--epochs", str(ep), "--cap", str(cap),
                   "--tag", f"grux-{n}", "--no-pull", "--max-minutes", "600", "--micro", str(micro),
                   "--micro-tokens", str(mtok), "--accum", str(32 // micro), "--save-every", str(every)]
            if not (rank == 16 and ep == 1):
                cmd.append("--skip-base")       # the r16-e1 run of each size scores its base once
            os.makedirs(run, exist_ok=True)
            launches(root, n, add=True)
            subprocess.Popen(cmd, stdout=open(os.path.join(run, "stdout.log"), "a"), stderr=subprocess.STDOUT,
                             start_new_session=True)  # a launcher restart never takes a run down with it
            live.add(n)
            print(time.strftime("%H:%M:%S"), "started", n, f"(launch {launches(root, n)})", flush=True)
            time.sleep(20)                      # stagger creates; spot capacity is contended
        time.sleep(15)
    left = [name(*c) for c in todo if not done(root, name(*c))]
    print(time.strftime("%H:%M:%S"), "SWEEP DONE", "not done:", left or "none", flush=True)


if __name__ == "__main__":
    main()
