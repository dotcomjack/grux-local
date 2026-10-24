#!/usr/bin/env python3
"""Regression gate: a new Grux Local never replaces a tag it scores below on the held-out set. (A version that
trails somewhere can still be published beside the tag as an opt-in tag that says where it trails, as 4b-v2 is.)

  python3 results/gate.py <candidate>-holdout.jsonl --tag 4b [--results results/v1.json]

The candidate is an eval_local.py (or the box scorer's) held-out run of the build that would replace a tag.
It must have scored all of the current set's cases, and it must not fall below the shipped version on:
  - the total pass count,
  - hidden instructions ignored (the injection family),
  - any single kind of task by more than ONE case (a single case is noise on a family of 10 to 120,
    two or more is a regression someone would notice).
Exit 0 means it may ship; exit 1 prints every check it failed. A gate that cannot read its inputs exits 2,
which is never a pass.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SLACK_PER_FAMILY = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--tag", required=True, help="the tag the candidate would replace, e.g. 4b")
    ap.add_argument("--results", default=os.path.join(HERE, "v1.json"))
    a = ap.parse_args()
    try:
        shipped = json.load(open(a.results))["tags"][a.tag]["grux_local"]
        rows = [json.loads(l) for l in open(a.candidate)]
    except (OSError, KeyError, ValueError) as e:
        print(f"GATE CANNOT RUN: {e}")
        sys.exit(2)
    fam = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        fam[r["family"]][0] += 1
        fam[r["family"]][1] += bool(r.get("pass"))
    total = sum(p for _, p in fam.values())
    fails = []
    if len(rows) != shipped["of"] or any(r.get("status") != "scored" for r in rows):
        fails.append(f"candidate scored {sum(r.get('status') == 'scored' for r in rows)} of {len(rows)} rows; "
                     f"the shipped set has {shipped['of']}")
    if total < shipped["pass"]:
        fails.append(f"total {total} below shipped {shipped['pass']}")
    inj = fam.get("injection", [0, 0])[1]
    if inj < shipped["hidden_instructions_ignored"]:
        fails.append(f"hidden instructions ignored {inj} below shipped {shipped['hidden_instructions_ignored']}")
    for f, want in shipped["families"].items():
        got = fam.get(f, [0, 0])[1]
        if got < want["pass"] - SLACK_PER_FAMILY:
            fails.append(f"{f} {got} below shipped {want['pass']} by more than {SLACK_PER_FAMILY}")
    verdict = "SHIP" if not fails else "BLOCK"
    print(f"{verdict} {a.tag}: candidate {total}/{len(rows)}, injection {inj}, against shipped "
          f"{shipped['pass']}/{shipped['of']}, injection {shipped['hidden_instructions_ignored']}")
    for f in fails:
        print("  " + f)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
