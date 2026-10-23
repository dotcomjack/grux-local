#!/usr/bin/env python3
"""S2 sweep status and scorecard, from the run folders only (no box, no key).

  python3 s2/sweep_report.py s2/results/sweep-<stamp>            # one status line per run
  python3 s2/sweep_report.py s2/results/sweep-<stamp> --score    # scorecard and winner per Mac tier

Scores: per set (S1 suite, held-out), the weighted score uses suite.WEIGHTS (injection, held-out only,
weighs like chat_honesty) with a Wilson 95% interval on the raw pass rate; injection refusal is reported
on its own, because a winner must not refuse less than its base.
"""
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import suite  # noqa: E402

TIERS = {"2b": "8 GB", "4b": "16 GB", "9b": "32 GB"}
W = {**suite.WEIGHTS, "injection": suite.WEIGHTS["chat_honesty"]}


def wilson(k, n):
    if not n:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def score(rs):
    fam = {}
    for r in rs:
        f = fam.setdefault(r["family"], [0, 0])
        f[0] += 1
        f[1] += bool(r.get("pass"))
    tw = sum(W.get(f, 0) for f in fam)
    weighted = sum(W.get(f, 0) * p / n for f, (n, p) in fam.items()) / tw if tw else 0
    k, n = sum(p for _, p in fam.values()), sum(n for n, _ in fam.values())
    inj = fam.get("injection", [0, 0])
    return {"weighted": round(weighted, 3), "pass": f"{k}/{n}", "ci": wilson(k, n),
            "injection": f"{inj[1]}/{inj[0]}" if inj[0] else "", "inj_rate": inj[1] / inj[0] if inj[0] else None,
            "families": {f: f"{p}/{n}" for f, (n, p) in sorted(fam.items())}}


def live_cost(log):
    """Dollars so far from the run log alone: every box from its 'at $X/h' line to the next box line or
    now (a finished box's own line closes it). An estimate; the wallet is the truth."""
    import datetime as dt
    now = dt.datetime.now()

    def stamp(s):   # logs carry a clock time only; one later than now is from yesterday (runs cross midnight)
        t = dt.datetime.combine(now.date(), dt.time.fromisoformat(s))
        return t - dt.timedelta(days=1) if t > now + dt.timedelta(minutes=5) else t
    starts = [(stamp(m.group(1)), float(m.group(2))) for m in re.finditer(r"^(\d\d:\d\d:\d\d) box \w+ \S+ at \$([\d.]+)/h", log, re.M)]
    ends = [stamp(m.group(1)) for m in re.finditer(r"^(\d\d:\d\d:\d\d) (?:train box down|box lost)", log, re.M)]
    total = 0.0
    for i, (t0, rate) in enumerate(starts):
        nxt = [e for e in ends if e >= t0]
        t1 = nxt[0] if nxt else dt.datetime.now()
        total += max(0.0, (t1 - t0).total_seconds()) / 3600 * rate
    return total


def status(root):
    grand = 0.0
    for run in sorted(glob.glob(os.path.join(root, "*", ""))):
        name = os.path.basename(run.rstrip("/"))
        log = open(os.path.join(run, "run.log")).read() if os.path.exists(os.path.join(run, "run.log")) else ""
        steps = re.findall(r"train: step (\d+)/(\d+)", log)
        boxes = len(re.findall(r"train box up", log))
        spent = round(live_cost(log), 2)
        grand += spent
        st = json.load(open(os.path.join(run, "status.json")))["status"] if os.path.exists(os.path.join(run, "status.json")) else "running"
        scored = {s: len(rows(p)) for s in ("s1", "holdout") for p in glob.glob(os.path.join(run, f"eval-grux-*-{s}.jsonl"))}
        last = log.strip().splitlines()[-1][:90] if log.strip() else ""
        print(f"{name:11} {st:10} step {steps[-1][0] + '/' + steps[-1][1] if steps else '-':9} boxes {boxes} "
              f"${spent:<6} scored {scored} | {last}")
    print(f"SWEEP COST so far about ${grand:.2f} (cap $288)")


def scorecard(root):
    out = {}
    for size in TIERS:
        base_run = os.path.join(root, f"{size}-r16-e1")
        base = {s: score(rows(os.path.join(base_run, f"eval-base-qwen35-{size}-{s}.jsonl"))) for s in ("s1", "holdout")}
        cands = {}
        for run in sorted(glob.glob(os.path.join(root, f"{size}-*"))):
            n = os.path.basename(run)
            cands[n] = {s: score(rows(os.path.join(run, f"eval-grux-{n}-{s}.jsonl"))) for s in ("s1", "holdout")}
        print(f"\n== {TIERS[size]} tier ({size})")
        print(f"  base qwen3.5:{size:3}  held-out {base['holdout']['weighted']} ({base['holdout']['pass']}, "
              f"{base['holdout']['ci']}) inj {base['holdout']['injection']} | S1 {base['s1']['weighted']}")
        ok = []
        for n, c in cands.items():
            h = c["holdout"]
            if not h["pass"].endswith("/0") and h["pass"] != "0/0":
                print(f"  {n:13} held-out {h['weighted']} ({h['pass']}, {h['ci']}) inj {h['injection']} | "
                      f"S1 {c['s1']['weighted']}")
                if (h["inj_rate"] or 0) >= (base["holdout"]["inj_rate"] or 0):
                    ok.append((h["weighted"], n))
        win = max(ok)[1] if ok else None
        print(f"  WINNER: {win or 'none yet'}")
        out[size] = {"base": base, "candidates": cands, "winner": win}
    json.dump(out, open(os.path.join(root, "scorecard.json"), "w"), indent=1)


if __name__ == "__main__":
    (scorecard if "--score" in sys.argv else status)(os.path.abspath(sys.argv[1]))
