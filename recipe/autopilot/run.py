#!/usr/bin/env python3
"""The weekly Grux Local autopilot run: watch, score, decide, and on a win train and hand to the Mini gate.

  1. watch.py finds new bases (the seen-state keeps a build from being scored twice);
  2. score.py scores every candidate with no verdict and no complete result (at most 15 a run, the rest carry
     over), one spot box each, in parallel, beside each involved tier's base when that base has no complete row
     for the digest the registry serves today;
  3. decide.py gives win, tie or lose per candidate with a complete result, including one scored in an earlier
     run that never got a verdict; a build still incomplete after two runs is a lose;
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


MAX_PER_RUN = 15           # new candidates scored per run (under the $150 swing line); the rest carry to next week
MAX_INCOMPLETE_RUNS = 2    # a build still not scored completely after this many runs is a lose, and stops retrying


def complete_result(digest, model):
    """The newest complete, fully scored result for this build, or None. A result whose box listed no digest is
    matched by model name, and only from today's run (the watch's digest for today's build is then the only one
    known; an older undigested result could be another build)."""
    best = None
    for p in sorted(glob.glob(os.path.join(common.RESULTS, "20*.json"))):
        if "-decide-" in p:
            continue
        r = json.load(open(p))
        if (r.get("digest") == digest or (r.get("digest") is None and r.get("model") == model
                                         and r.get("date") == common.today())) and r.get("complete") and not any(v.get("unscored") for v in r["sets"].values()):
            best = p
    return best


_BASE_CURRENT = {}


def base_row_current(tier):
    """True when a complete stock row exists for the digest the registry serves for the tier's base today, False when
    none does, None when the registry could not be read (unknown: never rescore on a blip, never decide either).
    One registry read per tier per run."""
    if tier not in _BASE_CURRENT:
        _BASE_CURRENT[tier] = _base_row_current(tier)
    return _BASE_CURRENT[tier]


def _base_row_current(tier):
    tag = common.base_config()["tiers"][tier]["ollama_tag"]
    name, size = tag.split(":")
    try:
        digest = watch.registry_build(name, size)["digest"]
    except Exception as e:  # noqa: BLE001 (a registry blip: unknown, never crash the week)
        print(f"registry read for {tag} failed ({str(e)[:120]}); base row unknown this run", flush=True)
        return None
    import decide                                        # the row decide.py will compare against must be today's build
    _, row = decide.baseline_result(tier)
    return bool(row) and row.get("digest") == digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true")
    a = ap.parse_args()
    date = common.today()
    week = {"date": date, "steps": [], "decisions": []}
    wpath = os.path.join(common.RESULTS, f"watch-{date}.json")
    if os.path.exists(wpath):                           # a second run the same day keeps the first report
        wpath = os.path.join(common.RESULTS, f"watch-{date}-{dt.datetime.now():%H%M}.json")
    rep = watch.run(wpath)
    week["watch_report"] = os.path.relpath(wpath, common.REPO)
    week["watch"] = rep["counts"]
    pending = [c for c in rep["candidates"] if c["new"]]        # every build with no verdict yet
    state = json.load(open(watch.STATE_PATH)) if os.path.exists(watch.STATE_PATH) else {"seen": {}}
    attempts = state.setdefault("incomplete_runs", {})
    to_decide, to_score = [], []
    for c in pending:
        res = complete_result(c["digest"], c["id"])
        if res:
            to_decide.append((c, res))                           # scored earlier, never decided (killed, base failed)
        else:
            to_score.append(c)
    to_score.sort(key=lambda c: c.get("updated") or "", reverse=True)             # newest first ...
    to_score.sort(key=lambda c: c["tier"] != common.base_config()["headline_tier"])  # ... headline tier ahead (stable)
    week["carried_over"] = [c["id"] for c in to_score[MAX_PER_RUN:]]
    to_score = to_score[:MAX_PER_RUN]
    models = [c["id"] for c in to_score]
    for tier in sorted({c["tier"] for c in pending}):
        if base_row_current(tier) is False:
            models.append(common.base_config()["tiers"][tier]["ollama_tag"])
    week["scored"] = models
    if models:
        r = subprocess.run([sys.executable, os.path.join(HERE, "score.py"), "--models", ",".join(models), "--date", date])
        week["steps"].append({"score_exit": r.returncode})
        _BASE_CURRENT.clear()                             # a base rescored just now may be current
    for c in to_score:
        res = os.path.join(common.RESULTS, f"{date}-{common.stem(c['id'])}.json")
        done = complete_result(c["digest"], c["id"])
        if done:
            to_decide.append((c, done))
            continue
        if not os.path.exists(res):                      # refused or crashed before a box ran: not an attempt
            week["decisions"].append({"candidate": c["id"], "verdict": None, "why": "no box ran this week; next week retries"})
            continue
        runs = attempts.setdefault(c["digest"], [])
        if date not in runs:
            runs.append(date)
        if len(runs) >= MAX_INCOMPLETE_RUNS:
            r = subprocess.run([sys.executable, os.path.join(HERE, "decide.py"), res, "--unscoreable", str(len(runs)),
                                "--digest", c["digest"]], capture_output=True, text=True)
            week["decisions"].append({"candidate": c["id"], "verdict": "lose", "line": (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[0]})
        else:
            week["decisions"].append({"candidate": c["id"], "verdict": None,
                                      "why": f"not scored completely (run {len(runs)} of {MAX_INCOMPLETE_RUNS}); next week retries"})
    common.write_json(watch.STATE_PATH, state)
    for c, res in to_decide:
        if base_row_current(c["tier"]) is not True:      # never a permanent verdict against a stale or unknown base
            week["decisions"].append({"candidate": c["id"], "verdict": None,
                                      "why": f"the {c['tier']} base row is not proven current for today's registry "
                                             "build; next run"})
            continue
        r = subprocess.run([sys.executable, os.path.join(HERE, "decide.py"), res, "--digest", c["digest"]],
                           capture_output=True, text=True)
        print(r.stdout, r.stderr, flush=True)
        dpath = os.path.join(common.RESULTS, f"{json.load(open(res))['date']}-decide-{common.stem(c['id'])}.json")
        d = json.load(open(dpath)) if os.path.exists(dpath) else {}
        if d.get("candidate_digest") != c["digest"]:
            d = {}
        week["decisions"].append({"candidate": c["id"], "verdict": d.get("verdict"),
                                  "line": d.get("line") or (r.stdout + r.stderr).strip()[-300:]})
        if d.get("verdict") == "win" and not a.no_train:
            t = subprocess.run([sys.executable, os.path.join(HERE, "train.py"), dpath])
            week["steps"].append({"train": c["id"], "exit": t.returncode})
    common.write_json(os.path.join(common.RESULTS, f"week-{date}.json"), week)
    print(json.dumps(week, indent=1))


if __name__ == "__main__":
    main()
