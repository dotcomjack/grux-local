#!/usr/bin/env python3
"""Win, tie or lose: a candidate's stock score against the current base's stock score for the same Mac tier.

Both rows come from score.py (same harness, same box type, same Ollama, same held-out sets), so the comparison is
stock for stock. The interval is the 95% interval of the DIFFERENCE in pass rate (Newcombe's method on the two
Wilson intervals; unpaired, so wider than a paired test, a conservative bar for spending a training run):

  win   the difference over both sets together (930 tasks) is above zero across its whole interval, and neither
        set's difference, nor hidden instructions on either set, is below zero across its whole interval;
  lose  the difference over both sets together is below zero across its whole interval;
  tie   anything else.

  python3 recipe/autopilot/decide.py results/autopilot/<date>-<candidate>.json
  python3 recipe/autopilot/decide.py --reproduce results/autopilot/<date>-qwen3.5_4b.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SETS = ("holdout", "holdout2")


def tier_of(model):
    return common.tier_for(common.params_from_tag(model.split(":")[-1]))


def baseline_result(tier):
    tag = common.base_config()["tiers"][tier]["ollama_tag"]
    rows = []
    for p in sorted(glob.glob(os.path.join(common.RESULTS, f"*-{common.stem(tag)}.json"))):
        r = json.load(open(p))
        if r.get("complete") and r.get("model") == tag:
            rows.append((r["date"], p, r))
    if not rows:
        return None, None
    _, p, r = rows[-1]
    return p, r


def diff(c, b):
    return {"candidate": f"{c['pass']}/{c['of']}", "base": f"{b['pass']}/{b['of']}",
            "difference": round(c["pass"] / c["of"] - b["pass"] / b["of"], 4),
            "ci95": common.newcombe(c["pass"], c["of"], b["pass"], b["of"])}


def decide(cand_path):
    cand = json.load(open(cand_path))
    tier = tier_of(cand["model"])
    if tier is None:
        raise SystemExit(f"{cand['model']}: no Mac tier for its size")
    if not cand.get("complete"):
        raise SystemExit(f"{cand['model']}: the candidate's score is not complete; no verdict")
    base_path, base = baseline_result(tier)
    if base is None:
        raise SystemExit(f"no complete stock row for the {tier} base yet; score.py scores it beside the candidate")
    if base["model"] == cand["model"]:
        raise SystemExit("the candidate is the base")
    out = {"candidate": cand["model"], "candidate_digest": cand.get("digest"), "candidate_result": os.path.relpath(os.path.abspath(cand_path), common.REPO),
           "tier": tier, "made_for": common.base_config()["tiers"][tier]["made_for"],
           "base": base["model"], "base_digest": base.get("digest"), "base_result": os.path.relpath(base_path, common.REPO), "sets": {}}
    ck = cb = nk = nb = 0
    for s in SETS:
        c, b = cand["sets"][s], base["sets"][s]
        out["sets"][s] = {"total": diff(c["total"], b["total"]),
                          "hidden_instructions": diff(c["hidden_instructions"], b["hidden_instructions"]),
                          "families": {f: diff(c["families"][f], b["families"][f]) for f in c["families"]
                                       if f in b["families"]}}
        ck += c["total"]["pass"]
        nk += c["total"]["of"]
        cb += b["total"]["pass"]
        nb += b["total"]["of"]
    both = {"candidate": f"{ck}/{nk}", "base": f"{cb}/{nb}", "difference": round(ck / nk - cb / nb, 4),
            "ci95": common.newcombe(ck, nk, cb, nb)}
    out["both_sets"] = both
    below = [f"{s} {k}" for s in SETS for k in ("total", "hidden_instructions") if out["sets"][s][k]["ci95"][1] < 0]
    if both["ci95"][0] > 0 and not below:
        verdict = "win"
    elif both["ci95"][1] < 0:
        verdict = "lose"
    else:
        verdict = "tie"
    out["verdict"] = verdict
    out["rule"] = __doc__.split("\n\n")[1].strip()
    out["worse_beyond_interval"] = below
    out["line"] = (f"{verdict.upper()}: {cand['model']} {ck}/{nk} against {base['model']} {cb}/{nb} on both held-out "
                   f"sets, difference {both['difference']:+.3f}, 95% interval [{both['ci95'][0]:+.3f}, "
                   f"{both['ci95'][1]:+.3f}]; held-out v1 {out['sets']['holdout']['total']['candidate']} vs "
                   f"{out['sets']['holdout']['total']['base']}, held-out v2 {out['sets']['holdout2']['total']['candidate']}"
                   f" vs {out['sets']['holdout2']['total']['base']}")
    path = os.path.join(common.RESULTS, f"{cand['date']}-decide-{common.stem(cand['model'])}.json")
    common.write_json(path, out)
    return out, path


def reproduce(path):
    """A box score of a published stock row must land inside that row's 95% interval (results/v1.json)."""
    r = json.load(open(path))
    tier = tier_of(r["model"])
    v1 = json.load(open(os.path.join(common.REPO, "results", "v1.json")))["tags"][tier]["before_training"]
    lo, hi = common.wilson(v1["pass"], v1["of"])
    got = r["sets"]["holdout"]["total"]
    inside = got["of"] == v1["of"] and lo * v1["of"] <= got["pass"] <= hi * v1["of"]
    line = (f"{'PROVEN' if inside else 'NOT PROVEN'}: {r['model']} on the box {got['pass']}/{got['of']} against the "
            f"published {v1['pass']}/{v1['of']} ({v1['measured']}, {json.load(open(os.path.join(common.REPO, 'results', 'v1.json')))['tags'][tier].get('before_training_machine')}), "
            f"whose 95% interval is {lo * v1['of']:.1f} to {hi * v1['of']:.1f} of {v1['of']}; hidden instructions "
            f"{r['sets']['holdout']['hidden_instructions']['pass']}/70 against {v1['hidden_instructions_ignored']}/70")
    return inside, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--reproduce", action="store_true")
    a = ap.parse_args()
    if a.reproduce:
        ok, line = reproduce(a.result)
        print(line)
        sys.exit(0 if ok else 1)
    out, path = decide(a.result)
    print(out["line"])
    for s in SETS:
        print(f"  {s}: total {out['sets'][s]['total']}, hidden {out['sets'][s]['hidden_instructions']}")
    print(f"wrote {path}")
    sys.exit({"win": 0, "tie": 10, "lose": 11}[out["verdict"]])


if __name__ == "__main__":
    main()
