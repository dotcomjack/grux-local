#!/usr/bin/env python3
"""The rejection-sampling path end to end at $0: eval_run.py --set train_local on a stub box, then export_local.py.
No Daytona box, no registry, no model: a stub box answers each request with a reply chosen by its seed.

  1. eval_run --set train_local, through main(): --passes 3 samples every job three times, pass k at seed 7 + k - 1,
     one window at a time, only the jobs 3.3.1 sends (the rest listed in the manifest); every sample keeps its
     request as sent, its reply as returned and its score; --temperature reaches every request, without it the
     app's 0.5 does, and on any other set it is refused;
  2. export_local turns the passing samples into rows: at most --per-case distinct replies a job (the same words in
     other spacing are one reply, the same tool with other arguments are two), paired only with the job and the
     request a sample answered, never a prefilled turn, never a row TrainGate or the data scan flags; jobs with no
     pass are listed by family; one model's samples only; a refused export leaves nothing a launcher could train on;
  3. the output passes contamination --gate (its .gate.json, nothing dropped) and data_scan.check, and every row
     renders through box_train.render_target (read out of box_train.py, which imports torch at the top). The other
     half of box_train's parsing, tokenizer.apply_chat_template over messages and tools, needs the model's tokenizer
     and is NOT run here.

  GRUX_SRC_V3=<Grux v3.3.1 Sources/Grux> [GRUX_DATA_SCAN_VOCAB=<file>] python3 recipe/s2/test_export_local.py
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, RECIPE)
try:
    import daytona  # noqa: F401
except ImportError:   # no box is made here, so a stand-in for the SDK is enough
    sys.modules["daytona"] = types.SimpleNamespace(**{n: object for n in (
        "CreateSandboxFromImageParams", "Daytona", "GpuType", "Image", "ListSandboxesQuery", "Resources",
        "SessionExecuteRequest")})
import contamination as ct  # noqa: E402
import data_scan  # noqa: E402
import eval_run  # noqa: E402
import export_local  # noqa: E402
import holdout_v4  # noqa: E402
import suite  # noqa: E402
import train_local as tl  # noqa: E402

MODEL = "gruxai.com/grux/grux-local:2b"
VOCAB = os.environ.get("GRUX_DATA_SCAN_VOCAB") or os.path.join(
    os.path.dirname(os.path.dirname(os.environ.get("GRUX_SRC_V3", "") or ".")), "Tests", "GruxTests",
    "banned-vocabulary.json")
src = open(os.path.join(HERE, "box_train.py")).read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "render_target")
ns = {"json": json}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "box_train.py", "exec"), ns)
render_target = ns["render_target"]
fails = 0


def check(label, ok, detail=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + label + ("" if ok else f"  [{str(detail)[:300]}]"))
    fails += 0 if ok else 1


check("data_scan has an identity vocabulary to read (it fails closed without one)", data_scan.load_terms(VOCAB), VOCAB)
CASES = tl.cases()
GONE = tl.not_sent(CASES)
SENT = [c for c in CASES if c["id"] not in GONE]
BY_TEXT = {c["utterance"]: c for c in SENT}


def utterance(body):
    return body["messages"][-1]["content"].split("THE USER'S MESSAGE:\n", 1)[1]


def variant(c, req, seed, j):
    """The stub model's reply: the ideal at seed 7; at seed 8 a second distinct passing reply for even jobs and the
    same reply in other spacing for odd ones; a failing reply at seed 9; job 1 fails at every seed."""
    good = tl.ideal_reply(c, req)["message"]
    if j == 1 or seed == 9:
        return {"role": "assistant", "content": ""}
    if seed == 7:
        return good
    if j % 2:   # suite._norm reads it as the same words
        return dict(good, content="  " + (good["content"] or "").upper() + " !")
    if good.get("tool_calls"):
        calls = [dict(good["tool_calls"][0], function=dict(good["tool_calls"][0]["function"], arguments=dict(
            good["tool_calls"][0]["function"]["arguments"], note="from the stub")))]
        return dict(good, tool_calls=calls)
    return dict(good, content=good["content"] + " Hope that helps.")


class StubBox:
    """eval_run's OllamaBox at $0: loads any model, answers /api/chat with variant()."""
    order = {}
    posts = []

    def __init__(self, *a, **k):
        self.info = {"id": "stub", "gpu_type": "H100", "boxes_left": []}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def pull(self, model, timeout=3600, deadline=None):
        return "success", 0.0

    def get(self, path, timeout=30):
        if path == "/api/tags":
            return {"models": [{"name": MODEL, "digest": "sha256:stub"}]}
        if path == "/api/ps":
            return {"models": [{"name": MODEL, "runner": "stub", "details": {}}]}
        return {"version": "stub"}

    def post(self, path, body, timeout=600):
        if path != "/api/chat":
            return {"done": True}
        c = BY_TEXT[utterance(body)]
        StubBox.posts.append((c["id"], body["options"]["seed"], body["options"]["num_ctx"],
                              body["options"]["temperature"]))
        req = tl.request_local(c, MODEL)
        return {"message": variant(c, req, body["options"]["seed"], StubBox.order.get(c["id"], 0)),
                "done_reason": "stop", "eval_count": 9, "prompt_eval_count": 99}

    def drop(self, model):
        pass

    def evicted(self):
        return False


class FakeReserved:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return "rid"

    def __exit__(self, *exc):
        return False


eval_run.missing_models = lambda models: ([], [])   # the stub serves every tag; no registry is asked
eval_run.meter.Reserved = FakeReserved
eval_run.OllamaBox = StubBox


def sample(run, *extra):
    StubBox.posts = []
    sys.argv = ["eval_run.py", "--run", run, "--set", "train_local", "--models", MODEL, "--meter", "x",
                "--parallel", "1"] + list(extra)
    eval_run.main()
    return [json.loads(l) for p in sorted(os.listdir(run)) if p.endswith(".jsonl") and "train_local" in p
            for l in open(os.path.join(run, p))]


# --------------------------------------------------------------------------- 1. eval_run --set train_local

LIMIT = 3
picked = {}
for c in SENT:   # what eval_run --limit 3 keeps: the first 3 sent jobs of each family, in bank order
    if len(picked.setdefault(c["family"], [])) < LIMIT:
        picked[c["family"]].append(c["id"])
JOBS = [c for c in SENT if c["id"] in {i for v in picked.values() for i in v}]
StubBox.order = {c["id"]: j for j, c in enumerate(JOBS)}
run = tempfile.mkdtemp()
rows = sample(run, "--passes", "3", "--limit", str(LIMIT))
man = json.load(open(os.path.join(run, "manifest.json")))
files = sorted(p for p in os.listdir(run) if p.endswith(".jsonl") and p != "capabilities.jsonl")
check("--passes 3 writes one file a pass, named for the model and the set",
      files == [f"{MODEL.replace(':', '_').replace('/', '_')}-train_local-pass{k}.jsonl" for k in (1, 2, 3)], files)
check(f"every pass samples every limited job once ({len(JOBS)} jobs), only jobs 3.3.1 sends",
      sorted((r["case"], r["seed"]) for r in rows) == sorted((c["id"], s) for c in JOBS for s in (7, 8, 9))
      and not {r["case"] for r in rows} & set(GONE))
check("pass k is sent at seed 7 + k - 1", all(json.loads(l)["seed"] == 6 + k for k in (1, 2, 3) for l in open(
    os.path.join(run, files[k - 1]))))
sweeps = {}
for cid, seed, num_ctx, _ in StubBox.posts:
    sweeps.setdefault(seed, []).append(num_ctx)
sweeps = {k: v[1:] for k, v in sweeps.items()}   # each pass opens with the warm call, which is never written
check("each pass sends one window at a time, smallest first",
      all(w == sorted(w) and len(set(w)) == 3 and len(w) == len(JOBS) for w in sweeps.values())
      and sorted(sweeps) == [7, 8, 9], sweeps)
keys = {"request", "reply", "pass", "why", "job_sha", "seed", "temperature", "num_ctx", "model", "offered"}
check("every sample keeps its full request, its reply as returned and its score",
      all(keys <= set(r) and r["request"]["messages"] and r["reply"].get("role") == "assistant" for r in rows))
check("without --temperature every request keeps the app's 0.5",
      all(r["temperature"] == 0.5 == r["request"]["options"]["temperature"] for r in rows))
check("the request recorded is the one sent: rebuilt from train_local at the pass's seed, byte for byte",
      all(r["request"] == dict(tl.request_local(next(c for c in JOBS if c["id"] == r["case"]), MODEL)["body"],
                               options=dict(tl.request_local(next(c for c in JOBS if c["id"] == r["case"]),
                                                             MODEL)["body"]["options"], seed=r["seed"]))
          for r in rows))
check("the manifest counts the jobs 3.3.1 sends, lists every one it answers alone, and names the set's own sha",
      man["sent"] == len(SENT) == len(CASES) - len(GONE) and man["not_sent"] == GONE and len(GONE) == 138
      and man["suite_sha"] == tl.case_sha(JOBS) and man["set"] == "train_local",
      (man["sent"], len(man["not_sent"]), man["suite_sha"][:12]))
check("the stub scores as held-out local: the ideal passes, an empty reply fails",
      all(r["pass"] == (r["seed"] != 9 and StubBox.order[r["case"]] != 1) for r in rows),
      [(r["case"], r["seed"], r["why"]) for r in rows if r["pass"] != (r["seed"] != 9 and StubBox.order[r["case"]] != 1)])
run_t = tempfile.mkdtemp()
rows_t = sample(run_t, "--limit", "1", "--temperature", "0.9")
check("--temperature 0.9 reaches every request and its row",
      rows_t and all(r["temperature"] == 0.9 == r["request"]["options"]["temperature"] for r in rows_t)
      and all(t == 0.9 for _, _, _, t in StubBox.posts))
eval_run.LOCAL = eval_run.TEMPERATURE = None
p = subprocess.run([sys.executable, os.path.join(RECIPE, "eval_run.py"), "--run", tempfile.mkdtemp(), "--set",
                    "holdout_v4", "--models", MODEL, "--temperature", "0.7"], capture_output=True, text=True)
check("--temperature on any other set is refused before any box",
      p.returncode != 0 and "--temperature is for --set train_local only" in p.stderr, p.stderr[-200:])

# --------------------------------------------------------------------------- 2 and 3. export_local

out = tempfile.mkdtemp()
s = export_local.export(run, out, per_case=2, vocab=VOCAB)
got = [json.loads(l) for l in open(os.path.join(out, "train_local.gated.jsonl"))]
want = {c["id"]: (0 if j == 1 else 1 if j % 2 else 2) for j, c in enumerate(JOBS)}
per = {}
for r in got:
    per[r["id"].split("#")[0]] = per.get(r["id"].split("#")[0], 0) + 1
check("an even job keeps two distinct passing replies, an odd job one (its second is the same words respaced), "
      "the job that never passes none", per == {k: v for k, v in want.items() if v}, (per, want))
check("the duplicates and the failing samples are counted, not written",
      s["dropped"]["duplicate"] == sum(1 for j in range(len(JOBS)) if j % 2 and j != 1) and s["rows"] == len(got))
zero = [i for f in s["families"].values() for i in f["zero_pass"]]
check("the job with no passing sample is listed as zero-pass, and the share counts it",
      zero == [JOBS[1]["id"]] and s["share_with_a_pass"] == round((len(JOBS) - 1) / len(JOBS), 3), (zero, s["share_with_a_pass"]))
rep = json.load(open(os.path.join(out, "train_local.gated.jsonl.gate.json")))
check("contamination --gate wrote the gated copy's .gate.json and dropped nothing",
      rep["rows"] == rep["kept"] == len(got) and rep["dropped"] == 0)
ok, msg = data_scan.check(os.path.join(out, "train_local.gated.jsonl"), VOCAB)
check("data_scan.check calls the gated copy clean", ok, msg)
case_by = {c["id"]: c for c in JOBS}
shape = all(set(r) == {"id", "family", "messages", "tools", "target"} and r["messages"][-1]["role"] == "user"
            and r["messages"] == tl.request_local(case_by[r["id"].split("#")[0]], MODEL)["body"]["messages"]
            for r in got)
check("every row is box_train's shape, the request the job sends, ending on the user's turn", shape)
rendered = [render_target(r["target"]) for r in got]
calls_ok = all((f"<function={c['name']}>" in t) for r, t in zip(got, rendered) for c in r["target"]["tool_calls"])
check("every row's target renders through box_train.render_target, tool calls as Qwen 3.5 writes them",
      all(t.endswith("<|im_end|>\n") for t in rendered) and calls_ok and any(r["target"]["tool_calls"] for r in got))
check("ids are unique", len({r["id"] for r in got}) == len(got))
s1 = export_local.export(run, tempfile.mkdtemp(), per_case=1, vocab=VOCAB)
check("--per-case 1 keeps one reply a job and counts the rest as over the cap",
      s1["rows"] == sum(1 for v in want.values() if v) and s1["dropped"]["over_cap"] == sum(1 for v in want.values()
                                                                                             if v == 2))
try:
    export_local.export(run, out, per_case=2, vocab=VOCAB)
    check("a second export into the same place is refused", False)
except SystemExit as e:
    check("a second export into the same place is refused", "already exists" in str(e))


def stub_sample(c, reply, seed=7, model=MODEL):
    """One eval_run sample row for case c with this reply, through eval_run.one_local itself."""
    class One:
        def post(self, path, body, timeout=600):
            return {"message": reply, "done_reason": "stop"}
    eval_run.LOCAL, eval_run.TEMPERATURE, saved = tl, None, suite.SEED
    suite.SEED = seed
    try:
        return eval_run.one(One(), model, c)
    finally:
        suite.SEED = saved
        eval_run.LOCAL = None


def export_rows(samples_, per_case=2):
    """(summary, gated rows, refusal) of an export of these samples. A refused export (the data scan refuses a file
    with no rows) leaves its summary in export_local.json and no gated row."""
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "m-train_local-pass1.jsonl"), "w") as f:
        for r in samples_:
            f.write(json.dumps(r) + "\n")
    o = tempfile.mkdtemp()
    try:
        s_ = export_local.export(d, o, per_case=per_case, vocab=VOCAB)
    except SystemExit as e:
        s_ = json.load(open(os.path.join(o, "export_local.json"))) if os.path.exists(
            os.path.join(o, "export_local.json")) else {}
        return s_, [], str(e)
    return s_, [json.loads(l) for l in open(os.path.join(o, "train_local.gated.jsonl"))], None


tool_job = next(c for c in SENT if c["family"] == "chat_tools" and c["tool"] in tl.request_local(c, MODEL)["offered"]
                and c["checks"])
plain = next(c for c in SENT if c["family"] == "chat_plain")
g = tl.ideal_reply(tool_job, tl.request_local(tool_job, MODEL))["message"]
other = json.loads(json.dumps(g))
other["tool_calls"][0]["function"]["arguments"]["note"] = "x"
s2, r2, e2 = export_rows([stub_sample(tool_job, g, 7), stub_sample(tool_job, other, 8)])
check("two passing tool calls with other arguments are two replies, not one duplicate",
      not e2 and len(r2) == 2 and s2["dropped"]["duplicate"] == 0, (e2, s2.get("dropped")))
pg = tl.ideal_reply(plain, tl.request_local(plain, MODEL))["message"]
three = [stub_sample(plain, dict(pg, content=pg["content"] + x), 7 + i) for i, x in enumerate(["", " Sure.", " Okay."])]
s3, r3, e3 = export_rows(three, per_case=2)
check("three distinct passing replies, --per-case 2: two rows",
      not e3 and len(r3) == 2 and s3["dropped"]["over_cap"] == 1, e3)
bad_job = stub_sample(plain, pg)
bad_job["job_sha"] = "0" * 64
bad_req = stub_sample(plain, pg, 8)
bad_req["request"]["messages"][1]["content"] += " "
s4, r4, e4 = export_rows([bad_job, bad_req, stub_sample(tool_job, g)])
check("a sample paired with another job or another request is never exported",
      not e4 and [r["id"].split("#")[0] for r in r4] == [tool_job["id"]] and s4["dropped"]["job_changed"] == 1
      and s4["dropped"]["request_changed"] == 1, (e4, s4.get("dropped")))
real_request = tl.request_local


def prefilled(c, model):
    r = real_request(c, model)
    if c["id"] == plain["id"] and "body" in r:
        r = dict(r, prefill="Sure: ", body=dict(r["body"], messages=r["body"]["messages"]
                                                 + [{"role": "assistant", "content": "Sure: "}]))
    return r


tl.request_local = prefilled
try:
    s5, r5, e5 = export_rows([stub_sample(plain, pg), stub_sample(tool_job, g)])
finally:
    tl.request_local = real_request
check("a prefilled turn is never a row (box_train would open a new assistant turn after it)",
      not e5 and [r["id"].split("#")[0] for r in r5] == [tool_job["id"]] and s5["dropped"]["prefilled"] == 1,
      (e5, s5.get("dropped")))
leak = next(r for r in (stub_sample(plain, dict(pg, content="Here you go: " + t)) for c in holdout_v4.cases()
                        for t in ct.v4_texts(c) if len(t.split()) >= 12 and not t.lstrip().startswith("{"))
            if r["pass"])   # the first held-out v4 text a plain reply can carry and still pass the scorer
key = stub_sample(plain, dict(pg, content="Here you go: " + "ghp_" + "A1b2C3d4" * 5), 8)
s6, r6, e6 = export_rows([leak, key, stub_sample(tool_job, g)])
check("a reply holding a held-out v4 run, and one holding a key shape, passed the scorer but are not written",
      leak["pass"] and key["pass"] and not e6 and [r["id"].split("#")[0] for r in r6] == [tool_job["id"]]
      and sum(s6["dropped"]["gate"].values()) == 1 and s6["dropped"]["data_scan"] == {"github token": 1},
      (leak["why"], key["why"], e6, s6.get("dropped")))
_, r7, e7 = export_rows([stub_sample(plain, pg), stub_sample(tool_job, g, 8, model="other:1")])
check("samples from two models are refused", not r7 and e7 and "2 models" in e7, e7)
d0 = tempfile.mkdtemp()
with open(os.path.join(d0, "m-train_local-pass1.jsonl"), "w") as f:
    f.write(json.dumps(stub_sample(plain, {"role": "assistant", "content": ""})) + "\n")
o0 = tempfile.mkdtemp()
try:
    export_local.export(d0, o0, vocab=VOCAB)
    check("an export the data scan refuses (here: no rows at all) is refused", False)
except SystemExit as e:
    check("an export the data scan refuses (here: no rows at all) is refused, and leaves no gated copy behind",
          "REFUSED" in str(e) and not os.path.exists(os.path.join(o0, "train_local.gated.jsonl"))
          and not os.path.exists(os.path.join(o0, "train_local.gated.jsonl.gate.json")), e)
try:
    export_local.export(run, tempfile.mkdtemp(), vocab="")
    check("an export with no identity vocabulary is refused", False)
except SystemExit as e:
    check("an export with no identity vocabulary is refused", "no identity vocabulary" in str(e))

print("SELF-TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
