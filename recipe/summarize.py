#!/usr/bin/env python3
"""Summarize an eval_local.py run the way results/v1.json reports it.

  python3 recipe/summarize.py <run dir>/<model>-holdout.jsonl

Prints the pass count out of the total, hidden instructions ignored out of 70, chat p95 (the same index rule
as results/build_v1.py) and the count per kind of task.
"""
import collections
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
unscored = [r for r in rows if r.get("status") != "scored"]
fam = collections.defaultdict(lambda: [0, 0])
for r in rows:
    fam[r["family"]][0] += 1
    fam[r["family"]][1] += bool(r.get("pass"))
chat = sorted(r["s"] for r in rows if r["family"].startswith("chat_") or r["family"] == "injection")
p95 = round(chat[int(0.95 * (len(chat) - 1))], 2) if chat else None
inj = fam.get("injection", [0, 0])
print(f"pass {sum(v[1] for v in fam.values())}/{len(rows)} | hidden instructions ignored {inj[1]}/{inj[0]} | "
      f"chat p95 {p95} s | unscored {len(unscored)}")
print({f: f"{v[1]}/{v[0]}" for f, v in sorted(fam.items())})
