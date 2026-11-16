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
import train_prompt  # noqa: E402
eval_run.TRAIN_PROMPT = train_prompt   # train_v3 runs always carry the job's own tools
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

# decision 1: the short prompt; decision 2: the hint reaches the teacher only
import train_prompt as tp  # noqa: E402
full = len(h.wire_tools())
ok_named, ok_size, sizes = True, True, []
for c in tb.kept()[::40]:
    names = tp.tool_names_for(c)
    want = set(tp._named(c, set(h.tool_names())))
    ok_named &= want <= set(names)
    ok_size &= len(names) == max(tp.TOOLS_PER_JOB, len(want)) and len(set(names)) == len(names)
    body = tp.student_body(c, 0)
    sizes.append((len(json.dumps(body["tools"])) + sum(len(m["content"] or "") for m in body["messages"])) / 4)
check("every tool a job names is offered, planted and forbidden ones included", ok_named)


def named_plain(c):
    """Collected without train_prompt: every call, alternative, forbidden, allowed and target tool, by plain loops."""
    out = set()
    if isinstance(c.get("target"), str):
        out.add(c["target"])
    for st in c["steps"]:
        if st.get("call"):
            out.add(st["call"])
        out |= set((st.get("alt") or {}).keys())
        out |= set(st.get("forbid") or [])
        rep = st.get("reply") or {}
        out |= set(rep.get("forbid") or []) | set(rep.get("allow") or [])
    return out & set(h.tool_names())


miss = [(c["id"], sorted(named_plain(c) - set(tp.tool_names_for(c)))) for c in tb.kept()
        if named_plain(c) - set(tp.tool_names_for(c))]
check(f"every call, alternative, forbidden and allowed tool is offered, read independently ({len(miss)} jobs miss one)",
      not miss)
check(f"each job offers {tp.TOOLS_PER_JOB} distinct tools, not {full}", ok_size)
sizes.sort()
check(f"a training prompt is about 12,000 tokens, not 31,000 (median {int(sizes[len(sizes) // 2])}, max {int(sizes[-1])})",
      sizes[len(sizes) // 2] < 14000 and sizes[-1] < 16000)
inj = next(c for c in tb.kept() if c["family"] == "injection")
other = next(c for c in tb.kept() if c["family"] == "chain")
raw_body = h.request_v3(inj, "t", 0)["body"]
hinted = tp.teacher_body(raw_body, inj, hint=True)
check("the hint reaches the teacher on an injection job", hinted["messages"][0]["content"].endswith(tp.HINTS["injection"]))
check("the hint never reaches the training row", tp.HINTS["injection"] not in json.dumps(tp.student_body(inj, 0)))
check("a family without a hint is sent unchanged", tp.teacher_body(h.request_v3(other, "t", 0)["body"], other, hint=True)
      ["messages"] == h.request_v3(other, "t", 0)["body"]["messages"])
check("the tool choice is the same on every call (seeded by the job id)", tp.tool_names_for(inj) == tp.tool_names_for(inj))

eval_run.TRAIN_PROMPT, eval_run.KEEP_RAW, eval_run.HINT_NOW = tp, True, True
SEEN.clear()
hrow = eval_run.one(ReplayBox(inj), "teacher", inj)
check("on a hint pass the box received the hint and only this job's tools",
      SEEN[inj["id"]][0]["messages"][0]["content"].endswith(tp.HINTS["injection"])
      and len(SEEN[inj["id"]][0]["tools"]) == len(tp.tool_names_for(inj)))
check("a hint-pass row records the hint and the tools it was offered", hrow["hint"] and hrow["tools_sha"] == tp.tools_sha(inj))
hd = tempfile.mkdtemp()
with open(os.path.join(hd, "t-train_v3-pass3.jsonl"), "w") as f:
    f.write(json.dumps(hrow) + "\n")
    old = dict(hrow, case=other["id"], job_sha=h.case_sha([other]))
    old.pop("tools_sha")
    f.write(json.dumps(old) + "\n")                      # a row taught on the full tool list
s4 = export_v3.export(hd, os.path.join(hd, "train.jsonl"))
got4 = [json.loads(l) for l in open(os.path.join(hd, "train.jsonl"))]
check("a hinted job exports without the hint, on its own tools",
      s4["jobs_kept"] == 1 and s4["hinted_jobs"] == 1 and tp.HINTS["injection"] not in json.dumps(got4)
      and got4[0]["tools"] == tp.student_body(inj, 0)["tools"])
check("a row taught on the full tool list is never exported", s4["dropped_job_changed"] == 1)
eval_run.HINT_NOW = False

# the teacher pass's prefill setting reaches the box on every step and never the training row
SEEN.clear()
eval_run.NUM_BATCH = 2048
nrow = eval_run.one(ReplayBox(c0), "teacher", c0)
eval_run.NUM_BATCH = 0
check("with --num-batch every step the box received carries it",
      len(SEEN[c0["id"]]) == len(c0["steps"]) and all(b["options"].get("num_batch") == 2048 for b in SEEN[c0["id"]]))
check("the setting keeps the request's own options (window, seed, reply length)",
      all({k: v for k, v in b["options"].items() if k != "num_batch"} == h.request_v3(c0, "teacher", k)["body"]["options"]
          for k, b in enumerate(SEEN[c0["id"]])))
SEEN.clear()
eval_run.one(ReplayBox(c0), "teacher", c0)
check("without it no step carries num_batch", all("num_batch" not in b["options"] for b in SEEN[c0["id"]]))
check("a row answered with it exports the same row", export_v3.rows_for(c0, nrow) == export_v3.rows_for(c0, rows[0]))
argv, sys.argv = sys.argv, ["eval_run.py", "--run", tempfile.mkdtemp(), "--set", "holdout_v3", "--models", "m",
                            "--num-batch", "2048"]
try:
    eval_run.main()
    refused = ""
except SystemExit as e:
    refused = str(e.code)
sys.argv = argv
check("a scoring run refuses --num-batch (its gate compares the baseline's runtime)", "train_v3 only" in refused)
eval_run.NUM_BATCH = 0

print("SELF-TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
