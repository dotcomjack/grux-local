#!/usr/bin/env python3
"""eval_run.py spends GPU money only on models that load, and never past its cap. No box is made: a fake box and a
fake meter stand in.

  1. step 0: a model that does not load, or whose pulled digest is not the expected one, writes no rows;
  2. a model whose first cases all fail is stopped early instead of run through every case;
  3. the dollar cap binds inside a model, not only between models;
  4. a clean model writes one scored row per case and frees its disk only after its last pass;
  5. --passes scores each model into its own file, and the kill rule stops the run when the first pass of the named
     model is at or above the threshold.

  GRUX_SRC_V3=<Grux v3.3.1 Sources/Grux> python3 recipe/test_eval_run_measure.py
"""
import json
import os
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import daytona  # noqa: F401
except ImportError:   # no box is made here, so a stand-in for the SDK is enough
    sys.modules["daytona"] = types.SimpleNamespace(**{n: object for n in (
        "CreateSandboxFromImageParams", "Daytona", "GpuType", "Image", "ListSandboxesQuery", "Resources",
        "SessionExecuteRequest")})
import eval_run  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name)
    fails += 0 if ok else 1


class FakeBox:
    def __init__(self, loads=True, digest="sha256:abc", answer_ok=True):
        self.loads, self.digest, self.answer_ok = loads, digest, answer_ok
        self.drops, self.posts = 0, 0
        self.info = {"gpu_type": "H200", "id": "fake"}

    def pull(self, model, timeout=3600, deadline=None):
        self.pull_timeout = timeout if deadline is None else min(timeout, deadline - time.time())
        return "success", 0.1

    def get(self, path, timeout=30):
        if path == "/api/tags":
            return {"models": [{"name": "m:1", "digest": self.digest}]}
        if path == "/api/ps":
            return {"models": [{"name": "m:1", "runner": "mlx", "size_vram": 1}]}
        return {"version": "0.40.8"}

    def post(self, path, body, timeout=600):
        self.posts += 1
        if path == "/api/generate":
            if not self.loads:
                raise RuntimeError("this model requires MLX support")
            return {"done": True}
        return {}

    def drop(self, model):
        self.drops += 1

    def evicted(self):
        return False


CASES = [{"id": f"c{i}", "family": "f"} for i in range(12)]


def fake_one(ok=True, delay=0.0):
    def one(box, model, case):
        time.sleep(delay)
        if not ok:
            return {"case": case["id"], "family": "f", "status": "error", "error": "boom", "s": 0}
        return {"case": case["id"], "family": "f", "status": "scored", "pass": True, "s": 0}
    return one


def fake_one_warm_fails():
    calls = {"n": 0}

    def one(box, model, case):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"case": case["id"], "family": "f", "status": "error", "error": "out of memory", "s": 0}
        return {"case": case["id"], "family": "f", "status": "scored", "pass": True, "s": 0}
    return one


def rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


run = tempfile.mkdtemp()
eval_run.log = lambda msg, run_dir=None: None

eval_run.one = fake_one()
p = os.path.join(run, "a.jsonl")
check("a model that does not load is not scored", eval_run.run_model(FakeBox(loads=False), "m:1", CASES, p, run, 4) is False)
check("... and writes no rows", rows(p) == [])
probes = [json.loads(l) for l in open(os.path.join(run, "capabilities.jsonl"))]
check("... and its probe verdict is written down", probes[-1]["probe_verdict"] == "did not load")

p = os.path.join(run, "b.jsonl")
check("a pulled digest that is not the shipped one is not scored",
      eval_run.run_model(FakeBox(digest="sha256:zzz"), "m:1", CASES, p, run, 4, expect_digest="sha256:abc") is False)
check("... and writes no rows", rows(p) == [])

eval_run.one = fake_one(ok=False)
p = os.path.join(run, "c.jsonl")
box = FakeBox()
check("a model whose first cases all fail is stopped", eval_run.run_model(box, "m:1", CASES, p, run, 1) is False)
check("... before it runs every case", box.posts < len(CASES))

eval_run.one = fake_one(delay=0.1)
p = os.path.join(run, "d.jsonl")
check("the cap binds inside a model", eval_run.run_model(FakeBox(), "m:1", CASES, p, run, 1,
                                                         deadline=time.time() + 0.35) == "cap")
check("... after scoring started and before every case ran", 1 <= len(rows(p)) < len(CASES))

eval_run.one = fake_one()
p = os.path.join(run, "e.jsonl")
box = FakeBox()
check("a clean model finishes", eval_run.run_model(box, "m:1", CASES, p, run, 4) is True)
check("... one scored row per case", len(rows(p)) == len(CASES) and all(r["status"] == "scored" for r in rows(p)))
check("... and frees its disk once", box.drops == 1)
box = FakeBox()
eval_run.run_model(box, "m:1", CASES, os.path.join(run, "f.jsonl"), run, 4, drop=False)
check("a model with a pass still to come keeps its disk", box.drops == 0)

eval_run.one = fake_one_warm_fails()
p = os.path.join(run, "g.jsonl")
check("a model that loads small but fails its first real job is not scored",
      eval_run.run_model(FakeBox(), "m:1", CASES, p, run, 4) is False and rows(p) == [])

eval_run.one = fake_one()
box = FakeBox()
check("a cap already reached stops before the pull",
      eval_run.run_model(box, "m:1", CASES, os.path.join(run, "h.jsonl"), run, 4, deadline=time.time() - 1) == "cap"
      and not hasattr(box, "pull_timeout"))
box = FakeBox()
eval_run.run_model(box, "m:1", CASES, os.path.join(run, "i.jsonl"), run, 4, deadline=time.time() + 120)
check("the pull's timeout is bounded by the money left", box.pull_timeout <= 121)

sp = os.path.join(run, "skips.jsonl")
with open(sp, "w") as f:
    for i in range(8):
        f.write(json.dumps({"case": f"s{i}", "status": "scored", "pass": True}) + "\n")
    for i in range(2):
        f.write(json.dumps({"case": f"k{i}", "status": "skipped"}) + "\n")
check("a skipped case counts as a fail, never leaves the denominator", eval_run.pass_pct(sp) == (8, 10, 2))

eval_run.one = fake_one(ok=False)
box = FakeBox()
box.evicted = lambda: False
calls_ok = {"n": 0}


def mostly_fail(b, model, case):
    calls_ok["n"] += 1
    if calls_ok["n"] <= 6:
        return {"case": case["id"], "family": "f", "status": "scored", "pass": True, "s": 0}
    return {"case": case["id"], "family": "f", "status": "error", "error": "wedged", "s": 0}


eval_run.one = mostly_fail
many = [{"id": f"m{i}", "family": "f"} for i in range(40)]
check("a server that stops answering mid-pass stops the pass instead of writing skips",
      eval_run.run_model(box, "m:1", many, os.path.join(run, "j.jsonl"), run, 1) is False
      and sum(r["status"] == "skipped" for r in rows(os.path.join(run, "j.jsonl"))) <= eval_run.skip_max(40) + 1)
eval_run.one = fake_one()

import gpu_box  # noqa: E402
gb = gpu_box.OllamaBox.__new__(gpu_box.OllamaBox)
gb.log = lambda m: None
posted = []
gb.post = lambda path, body, timeout=600: (posted.append(timeout), (_ for _ in ()).throw(TimeoutError("slow")))[1]
_sleep = gpu_box.time.sleep
gpu_box.time.sleep = lambda s: None   # the time module is shared: restored right after these two checks
status, _ = gb.pull("m:1", timeout=3600, deadline=time.time() + 50)
check("a pull's retry never runs past the dollar deadline", len(posted) >= 1 and all(t <= 50 for t in posted))
posted.clear()
status, _ = gb.pull("m:1", timeout=3600, deadline=time.time() - 1)
check("no pull starts once the money is gone", posted == [] and status.startswith("not tried"))
gpu_box.time.sleep = _sleep

# main: passes and the kill rule, with a fake box and a fake meter
calls = []


def fake_run_model(box, model, cases, path, run_dir, parallel, deadline=None, drop=True, expect_digest=None, skip=()):
    calls.append((model, os.path.basename(path), drop))
    with open(path, "a") as f:
        for c in cases:
            f.write(json.dumps({"case": c["id"], "status": "scored", "pass": True}) + "\n")
    return True


class FakeReserved:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return "rid"

    def __exit__(self, *exc):
        return False


class FakeOllamaBox(FakeBox):
    def __init__(self, *a, **k):
        super().__init__()
        self.kw = k

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_missing = eval_run.missing_models
eval_run.missing_models = lambda models: ([], [])   # the fake model names below are served by no registry
eval_run.run_model = fake_run_model
eval_run.meter.Reserved = FakeReserved
eval_run.OllamaBox = FakeOllamaBox
run2 = tempfile.mkdtemp()
sys.argv = ["eval_run.py", "--run", run2, "--set", "holdout_v3", "--models", "a:1,b:1", "--passes", "2",
            "--limit", "1", "--meter", "x", "--mlx"]
eval_run.main()
check("--passes 2 scores each model twice, each pass in its own file",
      [c[1] for c in calls] == ["a_1-holdout_v3-pass1.jsonl", "a_1-holdout_v3-pass2.jsonl",
                                "b_1-holdout_v3-pass1.jsonl", "b_1-holdout_v3-pass2.jsonl"])
check("... and frees a model's disk only after its last pass", [c[2] for c in calls] == [False, True, False, True])

calls.clear()
run3 = tempfile.mkdtemp()
sys.argv = ["eval_run.py", "--run", run3, "--set", "holdout_v3", "--models", "a:1,b:1", "--passes", "2",
            "--limit", "1", "--meter", "x", "--kill-model", "a:1", "--kill-at", "85"]
eval_run.main()
check("the kill rule stops the run after the named model's first pass scores at or above the threshold",
      [c[0] for c in calls] == ["a:1"])
check("... and the finish file lists what was left unrun",
      len(json.load(open(os.path.join(run3, "finish.json")))["unfinished"]) == 3)

made = {"n": 0}


class CountingBox(FakeOllamaBox):
    def __init__(self, *a, **k):
        made["n"] += 1
        super().__init__(*a, **k)


eval_run.OllamaBox = CountingBox
run4 = tempfile.mkdtemp()
sys.argv = ["eval_run.py", "--run", run4, "--set", "holdout_v3", "--models", "a:1,b:1", "--limit", "1",
            "--meter", "x", "--cap", "1"]
eval_run.main()
fin = json.load(open(os.path.join(run4, "finish.json")))
check("with less than a useful box slice of money, no box is made", made["n"] == 0)
check("... and the finish file says the cap stopped it with every model unrun",
      fin["stopped_by"] == "cap" and len(fin["unfinished"]) == 2)

check("a library tag's manifest is read from Ollama's registry",
      eval_run.manifest_url("qwen3.5:122b-a10b") == "https://registry.ollama.ai/v2/library/qwen3.5/manifests/122b-a10b")
check("a host-qualified tag's manifest is read from its own host",
      eval_run.manifest_url("gruxai.com/grux/grux-local:9b") == "https://gruxai.com/v2/grux/grux-local/manifests/9b")
for model, url in [
        ("qwen3.5", "https://registry.ollama.ai/v2/library/qwen3.5/manifests/latest"),
        ("someuser/model:7b", "https://registry.ollama.ai/v2/someuser/model/manifests/7b"),
        ("localhost:5000/ns/m:tag", "https://localhost:5000/v2/ns/m/manifests/tag"),
        ("qwen3.5@sha256:abcd", "https://registry.ollama.ai/v2/library/qwen3.5/manifests/sha256:abcd"),
        ("gruxai.com/grux/grux-local:9b@sha256:ef01", "https://gruxai.com/v2/grux/grux-local/manifests/sha256:ef01")]:
    check(f"manifest url for {model}", eval_run.manifest_url(model) == url)
served = {"https://registry.ollama.ai/v2/library/qwen3.5/manifests/122b-a10b": 200,
          "https://registry.ollama.ai/v2/library/qwen3.5/manifests/35b": 503}
got = _missing(["qwen3.5:122b-a10b", "qwen3.5:122b-a10b-q4", "qwen3.5:35b"], fetch=lambda u: served.get(u, 404))
check("only a 404 counts as missing", got[0] == ["qwen3.5:122b-a10b-q4"])
check("a registry error is reported as unverified, not as missing", got[1] == ["qwen3.5:35b (answer 503)"])

eval_run.missing_models = lambda models: (["a:1"], [])
try:
    sys.argv = ["eval_run.py", "--run", tempfile.mkdtemp(), "--set", "holdout_v3", "--models", "a:1", "--meter", "x"]
    made["n"] = 0
    eval_run.main()
    check("a run naming a tag no registry serves is refused before any box", False)
except SystemExit as e:
    check("a run naming a tag no registry serves is refused before any box", "REFUSED" in str(e) and made["n"] == 0)

print("SELF-TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
