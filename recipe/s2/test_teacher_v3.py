#!/usr/bin/env python3
"""The teacher pass over the v3 training bank keeps exactly what the scorer passed, with the teacher's own answer as
the target. No box: a stub box replays holdout_v3's perfect answers.

  1. eval_run's v3 job keeps each step's raw answer when asked (train_v3) and leaves held-out rows unchanged;
  2. a retry pass skips the jobs an earlier pass already passed, and only a retry pass changes the seed;
  3. export_v3 writes one row per step of each passing job: the request rebuilt from the bank, the raw answer as the
     target; a failed job and a job whose answer misses a step are not exported.

  GRUX_SRC_V3=<Grux v3.3.1 Sources/Grux> python3 recipe/s2/test_teacher_v3.py
"""
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
try:
    import daytona  # noqa: F401
except ImportError:
    sys.modules["daytona"] = types.SimpleNamespace(**{n: object for n in (
        "CreateSandboxFromImageParams", "Daytona", "GpuType", "Image", "ListSandboxesQuery", "Resources",
        "SessionExecuteRequest")})
import eval_run  # noqa: E402
import export_v3  # noqa: E402
import holdout_v3 as h  # noqa: E402
import train_bank_v3 as tb  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name)
    fails += 0 if ok else 1


cases = tb.kept()[:60]
by_fam = {}
for c in tb.kept():
    by_fam.setdefault(c["family"], c)
sample = list({c["id"]: c for c in list(by_fam.values()) + cases[:20]}.values())   # each job once


class ReplayBox:
    """Answers each step with holdout_v3's perfect answer for that step."""

    def __init__(self, case):
        self.raws, self.k, self.cid = h.perfect_raws(case), 0, case["id"]

    def post(self, path, body, timeout=600):
        SEEN.setdefault(self.cid, []).append(body)
        r = self.raws[self.k]
        self.k += 1
        return r


SEEN = {}   # case id -> the request bodies the box received, step by step
eval_run.V3 = h
eval_run.KEEP_RAW = True
rows = [eval_run.one(ReplayBox(c), "teacher", c) for c in sample]
check("every sampled bank job passes with the perfect answers", all(r["pass"] for r in rows))
check("each passing row keeps one raw answer per step",
      all(len(r["raw"]) == len(c["steps"]) for r, c in zip(rows, sample)))
eval_run.KEEP_RAW = False
plain = eval_run.one(ReplayBox(sample[0]), "teacher", sample[0])
check("a held-out row carries no raw answers", "raw" not in plain)

run = tempfile.mkdtemp()
p1 = os.path.join(run, "t-train_v3-pass1.jsonl")
with open(p1, "w") as f:
    for i, (r, c) in enumerate(zip(rows, sample)):
        r = dict(r)
        if i == 1:
            r["pass"] = False                      # a failed job
        if i == 2:
            r["raw"] = []                                            # an answer with no steps
        if i == next(j for j, c in enumerate(sample) if len(c["steps"]) > 1 and j > 3):
            r["raw"] = r["raw"][:-1]                                 # a multi-step answer missing its last step
        if i == 3:
            r["job_sha"] = "0" * 64                                  # an answer to a job that has since changed
        f.write(json.dumps(r) + "\n")
check("a retry pass skips the jobs an earlier pass passed",
      sample[1]["id"] not in eval_run.passed_ids([p1]) and sample[0]["id"] in eval_run.passed_ids([p1]))

out = os.path.join(run, "train.jsonl")
s = export_v3.export(run, out)
got = [json.loads(l) for l in open(out)]
ids = {g["id"].split("#")[0] for g in got}
check("a failed job is not exported", sample[1]["id"] not in ids)
multi = next(i for i, c in enumerate(sample) if len(c["steps"]) > 1 and i > 3)
c0 = sample[0]
r0 = [g for g in got if g["id"].startswith(c0["id"] + "#")]
check("a job whose answer misses a step is not exported", sample[multi]["id"] not in ids and sample[2]["id"] not in ids)
check("an answer whose job changed since the run is not exported", sample[3]["id"] not in ids and s["dropped_job_changed"] == 1)
check("a passing job becomes one row per step", len(r0) == len(c0["steps"]))
check("each row's request is the one the box received at that step",
      all(g["messages"] == SEEN[c0["id"]][k]["messages"] and g["tools"] == SEEN[c0["id"]][k]["tools"]
          for k, g in enumerate(r0)))
check("each row's target is the teacher's raw answer",
      all(g["target"]["tool_calls"] == rows[0]["raw"][k]["tool_calls"] for k, g in enumerate(r0)))
check("the export summary counts what it wrote", s["rows"] == len(got) and s["jobs_kept"] == len(ids))

print("SELF-TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
