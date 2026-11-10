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
MAX_BASE_RESCORES = 2      # base rescores per registry build; past it the lane says so loudly and stops paying


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
_BASE_DIGEST = {}


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
        _BASE_DIGEST[tier] = digest
    except Exception as e:  # noqa: BLE001 (a registry blip: unknown, never crash the week)
        print(f"registry read for {tag} failed ({str(e)[:120]}); base row unknown this run", flush=True)
        return None
    import decide                                        # the row decide.py will compare against must be today's build
    _, row = decide.baseline_result(tier)
    return bool(row) and row.get("digest") == digest


def main():
    """The week, with a heartbeat written at the end whatever happens (the board and the dead-man check read it)."""
    beat = {"lane": common.LANE, "last_run_at": None, "ok": False, "summary": "run.py started but did not finish"}
    try:
        prior = json.load(open(common.HEARTBEAT))
    except (OSError, ValueError):
        prior = {}
    common.write_json(common.HEARTBEAT, {**prior, "lane": common.LANE,
                                         "running": {"pid": os.getpid(), "started_at": common.now_iso()}})
    try:
        week = week_run()
        alerts = list(week.get("alerts") or [])
        for st in week["steps"]:
            if st.get("score_exit") == 3:
                alerts.append("the spend meter refused at least one scoring box (exit 3); a refused model retries next week")
            elif st.get("score_exit") == 4:
                alerts.append("scoring refused over the $150 swing line (exit 4): no box was created")
            elif st.get("score_exit") not in (None, 0):
                alerts.append(f"score.py exited {st['score_exit']}")
            if st.get("train") and st.get("exit") not in (None, 0):
                alerts.append(f"train.py for {st['train']} exited {st['exit']} (see its TRAIN REFUSED line in the log)")
        standing = standing_verdicts()
        cands = ", ".join(f"{m} {v}" for m, v in standing.items()) or "none"
        beat.update(ok=not alerts,
                    summary=(f"{week['watch'].get('candidates', 0)} candidate(s): {cands}; "
                             f"{week['watch'].get('new_candidates', 0)} without a verdict, "
                             f"{week['watch'].get('filtered', 0)} filtered; scored {len(week['scored'])} this run"
                             + (f"; ALERTS: {' | '.join(alerts)}" if alerts else "")))
    except BaseException as e:
        beat["summary"] = f"run.py failed: {type(e).__name__}: {str(e)[:300]}"
        raise
    finally:
        beat["last_run_at"] = common.now_iso()
        common.write_json(common.HEARTBEAT, beat)        # no "running" key: the run is over


def standing_verdicts():
    """Each current candidate (today's watch report) with its latest verdict, or 'pending'."""
    import glob as g
    reports = sorted(g.glob(os.path.join(common.RESULTS, f"watch-{common.today()}*.json")), key=os.path.getmtime)
    if not reports:
        return {}
    out = {}
    for c in json.load(open(reports[-1])).get("candidates", []):
        v = "pending"
        for p in sorted(g.glob(os.path.join(common.RESULTS, f"*-decide-{common.stem(c['id'])}.json"))):
            d = json.load(open(p))
            if d.get("candidate_digest") == c["digest"] and d.get("verdict"):
                v = d["verdict"]
        out[c["id"]] = v
    return out


def week_run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true")
    a = ap.parse_args()
    date = common.today()
    week = {"date": date, "steps": [], "decisions": []}
    wpath = os.path.join(common.RESULTS, f"watch-{date}.json")
    if os.path.exists(wpath):                           # a second run the same day keeps the first report
        wpath = os.path.join(common.RESULTS, f"watch-{date}-{dt.datetime.now():%H%M}.json")
    rep = watch.run(wpath)
    week["watch_report"] = wpath.replace(os.path.expanduser("~"), "~", 1)
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
    rescores = state.setdefault("base_rescores", {})
    for tier in sorted({c["tier"] for c in pending}):
        if base_row_current(tier) is False:
            key = f"{tier}|{_BASE_DIGEST.get(tier)}"
            if len(rescores.get(key, [])) >= MAX_BASE_RESCORES:
                msg = (f"BASE ROW STUCK: the {tier} base was scored {MAX_BASE_RESCORES} times for registry build "
                       f"{_BASE_DIGEST.get(tier)} and its newest complete row still does not carry that digest; "
                       "not rescoring again, its candidates wait for a person")
                print(msg, flush=True)
                week.setdefault("alerts", []).append(msg)
                continue
            rescores.setdefault(key, []).append(date)
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
    return week


if __name__ == "__main__":
    main()
