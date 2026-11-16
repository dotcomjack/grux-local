#!/usr/bin/env python3
"""train_run scores a trained model on held-out v1, v2 and v3, `--passes` times, through a stub Ollama client.
No model and no box: every answer here is scripted.

  1. v3 jobs run step by step: a client that replays holdout_v3.perfect_raws passes every sampled job, and each row
     carries its per-step facts;
  2. a client that answers nothing useful fails every sampled v3 job (the stub is not a fake pass);
  3. each pass writes its own file: pass 1 keeps the name readers use (eval-<model>-<set>.jsonl), pass k adds -pass<k>;
  4. a rerun skips what is already scored; a client error is retried once, then written as skipped (a fail);
  5. v1 and v2 are scored with their own request and scorer;
  6. the box's Ollama serves as many requests at once as train_run sends (--parallel).
Prints PASS or FAIL per check; exit 1 on any FAIL.

  GRUX_SRC_V3=<Grux v3.3.1 Sources/Grux> python3 recipe/s2/test_train_run_scoring.py
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
except ImportError:   # the box SDK is not needed to score through a stub client
    sys.modules["daytona"] = types.SimpleNamespace(**{n: object for n in (
        "CreateSandboxFromImageParams", "Daytona", "GpuType", "Image", "ListSandboxesQuery", "Resources",
        "SessionExecuteRequest")})
import holdout_v3 as h  # noqa: E402
import train_run as t  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


V3 = []
seen_fams = set()
for c in h.cases():   # one job of each family, and at least one with more than one step
    if c["family"] not in seen_fams or (len(c["steps"]) > 1 and not any(len(x["steps"]) > 1 for x in V3)):
        V3.append(c)
        seen_fams.add(c["family"])
V1 = t.holdout.cases()[:3]
V2 = t.holdout_v2.cases()[:3]
MODEL = "m"
KEY = {}
for c in V3:
    for k, r in enumerate(h.perfect_raws(c)):
        KEY[json.dumps(h.request_v3(c, MODEL, k)["body"]["messages"])] = r


class Perfect:
    """Replays the perfect answer for the exact v3 request it receives; an empty reply for anything else."""
    def __init__(self, fail_first=0):
        self.calls, self.fail_first = 0, fail_first

    def post(self, path, body, timeout=None):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ConnectionError("stub down")
        return KEY.get(json.dumps(body.get("messages")), {"message": {"role": "assistant", "content": ""}})


class Mute:
    def post(self, path, body, timeout=None):
        return {"message": {"role": "assistant", "content": ""}}


class Down:
    def post(self, path, body, timeout=None):
        raise ConnectionError("stub down")


MODULE_ORDER = list(t.SETS)
check("the module scores v3 first", MODULE_ORDER[0] == "holdout_v3", str(MODULE_ORDER))
STUB = {"holdout": (lambda: V1, t.holdout.request_holdout, t.holdout.score_holdout),
        "holdout_v2": (lambda: V2, t.holdout_v2.request_v2, t.holdout_v2.score_v2),
        "holdout_v3": (lambda: V3, None, None)}
t.SETS = {n: STUB[n] for n in MODULE_ORDER if n in STUB}   # the module's order, with small samples
t.time.sleep = lambda s: None


def rows(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


run = tempfile.mkdtemp()
t.score_model(Perfect(), MODEL, run, 2, passes=3)
p1, p3 = t.eval_path(run, MODEL, "holdout_v3", 1), t.eval_path(run, MODEL, "holdout_v3", 3)
check("pass 1 keeps the name readers use", os.path.basename(p1) == "eval-m-holdout_v3.jsonl")
check("pass 3 adds -pass3", os.path.basename(p3) == "eval-m-holdout_v3-pass3.jsonl")
for k in (1, 2, 3):
    r = rows(t.eval_path(run, MODEL, "holdout_v3", k))
    check(f"v3 pass {k}: every sampled job scored and passed with the perfect answers",
          len(r) == len(V3) and all(x["pass"] for x in r), str([(x["case"], x["why"]) for x in r if not x["pass"]][:3]))
check("the sample has a multi-step job", any(len(c["steps"]) > 1 for c in V3))
r1 = {x["case"]: x for x in rows(p1)}
check("each v3 row carries one fact per step it ran",
      all(len(r1[c["id"]]["steps"]) == len(c["steps"]) for c in V3))
for name, cs in (("holdout", V1), ("holdout_v2", V2)):
    for k in (1, 2, 3):
        check(f"{name} pass {k}: every case scored", len(rows(t.eval_path(run, MODEL, name, k))) == len(cs))
check("every row records the think setting it was scored with",
      all(x.get("think") is t.suite.THINK and "think" in x for x in rows(p1)))
check("v1 and v2 rows carry no step facts",
      all("steps" not in x for n in ("holdout", "holdout_v2") for x in rows(t.eval_path(run, MODEL, n, 1))))

again = Perfect()
t.score_model(again, MODEL, run, 2, passes=3)
check("a rerun skips everything already scored", again.calls == 0)

order, real_path = [], t.eval_path
t.eval_path = lambda r, m, n, k: order.append((n, k)) or real_path(r, m, n, k)
t.score_model(Mute(), MODEL, tempfile.mkdtemp(), 2, passes=2)
t.eval_path = real_path
check("v3 first, then each set with its passes together (one model load per context size)",
      order == [("holdout_v3", 1), ("holdout_v3", 2), ("holdout", 1), ("holdout", 2), ("holdout_v2", 1), ("holdout_v2", 2)],
      str(order))

mute = tempfile.mkdtemp()
t.score_model(Mute(), MODEL, mute, 2, passes=1)
check("a client that answers nothing fails every v3 job",
      len(rows(t.eval_path(mute, MODEL, "holdout_v3", 1))) == len(V3)
      and not any(x["pass"] for x in rows(t.eval_path(mute, MODEL, "holdout_v3", 1))))
check("passes=1 writes no pass 2 file", not os.path.exists(t.eval_path(mute, MODEL, "holdout_v3", 2)))

flaky = tempfile.mkdtemp()
t.SETS = {"holdout_v3": (lambda: V3[:1], None, None)}
t.score_model(Perfect(fail_first=1), MODEL, flaky, 1, passes=1)
check("a client error is retried once", [x["pass"] for x in rows(t.eval_path(flaky, MODEL, "holdout_v3", 1))] == [True])
down = tempfile.mkdtemp()
t.score_model(Down(), MODEL, down, 1, passes=1)
dr = rows(t.eval_path(down, MODEL, "holdout_v3", 1))
check("a job that errors twice is written as skipped, a fail, so the pass file covers every case",
      [(x["status"], x["pass"]) for x in dr] == [("skipped", False)])
again = Perfect()
t.score_model(again, MODEL, down, 1, passes=1)
check("a skipped job is not rerun into a second row", again.calls == 0 and len(rows(t.eval_path(down, MODEL, "holdout_v3", 1))) == 1)

import re  # noqa: E402
import train_box  # noqa: E402
src = open(t.__file__).read()
default_parallel = re.search(r'"--parallel", type=int, default=(\d+)', src)
check("the box serves as many requests at once as train_run sends",
      default_parallel is not None and f"OLLAMA_NUM_PARALLEL={default_parallel.group(1)} " in train_box.BOOT)

print("SELF-TEST", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
sys.exit(1 if FAILS else 0)
