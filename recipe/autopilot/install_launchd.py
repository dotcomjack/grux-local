#!/usr/bin/env python3
"""Install the Grux Local lane so the weekly job outlives any checkout, then load its two LaunchAgents.

Copies the recipe (and results/v1.json, the published rows the reproduction reads) to ~/.grux/autopilot/grux-local/
lane, seeds the stable results home ~/.grux/autopilot/grux-local/results with the repo's results/autopilot/*.json
(never overwriting), copies the training data named by --train-data to a stable path, self-tests the installed copy,
and writes:
  <label>            Mondays 3:00 AM, run.py
  <label>-deadman    Mondays 6:00 AM, deadman.py (red page if the run did not report)
This machine's own settings live in ~/.grux/autopilot/grux-local/local.json, never in the repo:
  {"label": "<launchd label>", "key_command": [argv that puts DAYTONA_API_KEY in the environment, then runs the
   rest, ending in "--"], "telegram_env": "<file with TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID>"}
The Daytona key is read at run time by key_command; no key is written into a plist.
Re-run after any change to the lane. Refuses while a weekly run is in progress.

  python3 recipe/autopilot/install_launchd.py [--train-data <train.jsonl>] [--load]
"""
import argparse
import glob
import json
import os
import plistlib
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
HOME = os.path.expanduser("~/.grux/autopilot/grux-local")
LANE = os.path.join(HOME, "lane")
RESULTS = os.path.join(HOME, "results")
TRAIN = os.path.join(HOME, "train-data", "train.jsonl")
VOCAB = os.path.join(HOME, "vocab")
GRUX_VOCAB = os.environ.get("GRUX_BANNED_VOCAB", "")   # Grux's own Tests/GruxTests/banned-vocabulary.json


def install_vocab(src):
    """Grux's banned vocabulary, copied beside the lane (it never enters this public repo). A local extra.txt in the
    same folder, if present, adds terms; it is never written by this installer."""
    os.makedirs(VOCAB, exist_ok=True)
    src = os.path.expanduser(src)
    if src and os.path.exists(src):
        shutil.copy2(src, os.path.join(VOCAB, "banned-vocabulary.json"))
    if not os.path.exists(os.path.join(VOCAB, "banned-vocabulary.json")):
        raise SystemExit(f"no identity vocabulary at {src} or {VOCAB}; the training data scan cannot run")
    return os.path.join(VOCAB, "banned-vocabulary.json") + ":?" + os.path.join(VOCAB, "extra.txt")


def install_train_data(src, vocab):
    """Copy the training file to its pinned path WITHOUT the rows that carry an identity term; refuse a secret shape.
    s2/train_run.py scans again before any box and refuses on any hit."""
    sys.path.insert(0, os.path.join(REPO, "recipe", "s2"))
    import data_scan
    terms = data_scan.load_terms(vocab)
    hits = data_scan.scan(src, terms)
    bad = [(r, k) for r, k, _ in hits if k != "identity term"]
    if bad:
        raise SystemExit(f"training data {src} carries a secret shape or an unreadable row; not installed: {bad[:5]}")
    drop = {n for _, _, n in hits}
    os.makedirs(os.path.dirname(TRAIN), exist_ok=True)
    import json
    kept = dropped = 0
    with open(src, encoding="utf-8") as fin, open(TRAIN + ".part", "w", encoding="utf-8") as fout:
        for n, line in enumerate(fin, 1):
            if n in drop or not line.strip():
                dropped += 1
                continue
            fout.write(line)
            kept += 1
    if kept == 0:
        os.remove(TRAIN + ".part")
        raise SystemExit(f"every row of {src} was dropped ({dropped}); nothing pinned, the old data is untouched")
    os.replace(TRAIN + ".part", TRAIN)
    return kept, dropped


def running():
    out = subprocess.run(["pgrep", "-f", os.path.join(LANE, "recipe", "autopilot")], capture_output=True, text=True)
    return [p for p in out.stdout.split() if p and int(p) != os.getpid()]


def copy_lane():
    tmp = LANE + ".new"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(os.path.join(REPO, "recipe"), os.path.join(tmp, "recipe"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "results"))
    os.makedirs(os.path.join(tmp, "results"))
    for f in ("v1.json", "gate.py"):
        shutil.copy2(os.path.join(REPO, "results", f), os.path.join(tmp, "results", f))
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    open(os.path.join(tmp, "VERSION"), "w").write(sha + "\n")
    old = LANE + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(LANE):
        os.replace(LANE, old)
    os.replace(tmp, LANE)
    shutil.rmtree(old, ignore_errors=True)
    return sha


def seed_results():
    os.makedirs(os.path.join(RESULTS, "gate-requests"), exist_ok=True)
    n = 0
    for p in glob.glob(os.path.join(REPO, "results", "autopilot", "20*.json")):   # results and verdicts, never a watch report
        dest = os.path.join(RESULTS, os.path.basename(p))
        if not os.path.exists(dest):
            shutil.copy2(p, dest)
            n += 1
    return n


def plist(label, args, when, env):
    log = os.path.expanduser(f"~/Library/Logs/{label}.log")
    return {"Label": label, "ProgramArguments": args, "StartCalendarInterval": when, "WorkingDirectory": LANE,
            "StandardOutPath": log, "StandardErrorPath": log,
            "EnvironmentVariables": {"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin", **env}}


def load(label, path):
    job = f"gui/{os.getuid()}/{label}"
    subprocess.run(["launchctl", "bootout", job], capture_output=True)
    for _ in range(10):                      # bootout is asynchronous; a bootstrap straight after can fail with EIO
        if subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", path], capture_output=True).returncode == 0:
            return True
        time.sleep(1)
    return subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", path]).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=os.path.join(HOME, "venv", "bin", "python"))
    ap.add_argument("--train-data", default="", help="round 2 train.jsonl to copy to a stable path")
    ap.add_argument("--vocab", default=GRUX_VOCAB, help="Grux's banned-vocabulary.json (or GRUX_BANNED_VOCAB); "
                    "kept from the last install when omitted")
    ap.add_argument("--load", action="store_true")
    a = ap.parse_args()
    local_p = os.path.join(HOME, "local.json")
    if not os.path.exists(local_p):
        raise SystemExit(f"write {local_p} first (label, key_command, telegram_env); see this file's docstring")
    local = json.load(open(local_p))
    LABEL, key_command = local["label"], local["key_command"]
    DEADMAN = LABEL + "-deadman"
    if running():
        raise SystemExit(f"a lane run is in progress ({' '.join(running())}); install after it ends")
    sha = copy_lane()
    seeded = seed_results()
    vocab = install_vocab(a.vocab)
    if a.train_data:
        kept, dropped = install_train_data(os.path.expanduser(a.train_data), vocab)
        print(f"training data pinned at {TRAIN}: {kept} rows kept, {dropped} dropped for an identity term")
    env = {"GRUX_LOCAL_AUTOPILOT_RESULTS": RESULTS, "GRUX_DATA_SCAN_VOCAB": vocab}
    if os.path.exists(TRAIN):
        env["GRUX_LOCAL_TRAIN_DATA"] = TRAIN
    lane_py = os.path.join(LANE, "recipe", "autopilot")
    r = subprocess.run([a.python, os.path.join(lane_py, "watch.py"), "--self-test"], capture_output=True, text=True,
                       env={**os.environ, **env})
    if r.returncode != 0:
        raise SystemExit("the installed lane failed its self-test; nothing loaded:\n" + r.stdout + r.stderr)
    jobs = {
        LABEL: plist(LABEL, [*key_command, a.python, os.path.join(lane_py, "run.py")], {"Weekday": 1, "Hour": 3, "Minute": 0}, env),
        DEADMAN: plist(DEADMAN, ["/usr/bin/python3", os.path.join(lane_py, "deadman.py")],
                       {"Weekday": 1, "Hour": 6, "Minute": 0}, env),
    }
    for label, body in jobs.items():
        path = os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")
        with open(path, "wb") as f:
            plistlib.dump(body, f)
        print(f"wrote {path}")
        if a.load:
            print(f"loaded {label}: {load(label, path)}")
    print(f"installed lane {sha[:7]} at {LANE}; results home {RESULTS} ({seeded} seeded); "
          f"train data {'at ' + TRAIN if os.path.exists(TRAIN) else 'not installed'}")


if __name__ == "__main__":
    main()
