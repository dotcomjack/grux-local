#!/usr/bin/env python3
"""The weekly Grux Local autopilot run: watch, score, decide, and on a win train and hand to the Mini gate.

  1. watch.py finds new bases (the seen-state keeps a build from being scored twice);
  2. score.py scores every new candidate stock, one spot box each, in parallel, beside each involved tier's base
     when that base has no complete row for the digest the registry serves today;
  3. decide.py gives win, tie or lose per candidate, with the numbers;
  4. train.py runs only on a win (and not with --no-train).
Writes results/autopilot/week-<date>.json. The launchd job (install_launchd.py) runs this on Mondays at 3:00 AM.

  dcj-secret run dcj/daytona/grux-api-key DAYTONA_API_KEY -- <venv>/bin/python recipe/autopilot/run.py [--no-train]
"""
import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import watch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def base_row_current(tier):
    """True when a complete stock row exists for the digest the registry serves for the tier's base today."""
    tag = common.base_config()["tiers"][tier]["ollama_tag"]
    name, size = tag.split(":")
    digest = watch.registry_build(name, size)["digest"]
    for p in glob.glob(os.path.join(common.RESULTS, f"*-{common.stem(tag)}.json")):
        r = json.load(open(p))
        if r.get("complete") and r.get("digest") == digest:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true")
    a = ap.parse_args()
    date = common.today()
    week = {"date": date, "steps": []}
    wpath = os.path.join(common.RESULTS, f"watch-{date}.json")
    if os.path.exists(wpath):                           # a second run the same day keeps the first report
        wpath = os.path.join(common.RESULTS, f"watch-{date}-{dt.datetime.now():%H%M}.json")
    rep = watch.run(wpath)
    week["watch_report"] = os.path.relpath(wpath, common.REPO)
    week["watch"] = rep["counts"]
    new = [c for c in rep["candidates"] if c["new"]]
    models = [c["id"] for c in new]
    for tier in sorted({c["tier"] for c in new}):
        if not base_row_current(tier):
            models.append(common.base_config()["tiers"][tier]["ollama_tag"])
    week["scored"] = models
    if models:
        r = subprocess.run([sys.executable, os.path.join(HERE, "score.py"), "--models", ",".join(models), "--date", date])
        week["steps"].append({"score_exit": r.returncode})
    week["decisions"] = []
    for c in new:
        res = os.path.join(common.RESULTS, f"{date}-{common.stem(c['id'])}.json")
        if not os.path.exists(res):
            week["decisions"].append({"candidate": c["id"], "verdict": None, "why": "no result written"})
            continue
        r = subprocess.run([sys.executable, os.path.join(HERE, "decide.py"), res], capture_output=True, text=True)
        print(r.stdout, r.stderr, flush=True)
        dpath = os.path.join(common.RESULTS, f"{date}-decide-{common.stem(c['id'])}.json")
        d = json.load(open(dpath)) if os.path.exists(dpath) else {}
        week["decisions"].append({"candidate": c["id"], "verdict": d.get("verdict"), "line": d.get("line") or r.stdout.strip()})
        if d.get("verdict") == "win" and not a.no_train:
            t = subprocess.run([sys.executable, os.path.join(HERE, "train.py"), dpath])
            week["steps"].append({"train": c["id"], "exit": t.returncode})
    common.write_json(os.path.join(common.RESULTS, f"week-{date}.json"), week)
    print(json.dumps(week, indent=1))


if __name__ == "__main__":
    main()
