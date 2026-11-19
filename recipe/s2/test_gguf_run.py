#!/usr/bin/env python3
"""gguf_run and train_run's artifact path through stub boxes and stub Ollama clients. No model, no box, no network:
every answer here is scripted.

  1. the kill rule stops a tier on a bad tool-call probe or a bad limited pass (no full scoring), continues on a good
     one, refuses to compare without MLX rows, and a box that goes away is not a kill;
  2. a shipped tag whose digest differs is refused, at preflight (registry) and on the box (Ollama's listed digest);
  3. short of disk, a download is skipped with the reason and the run still finishes done (train_run and gguf_run);
     a failed download does not fail a run whose scoring finished; the Pro GGUF is never pulled by default;
  4. base-model packs are never packed or downloaded, while the trained model's is;
  5. GGUF, limited, serial and shipped row file names never collide with an MLX or base row, or match their globs;
  6. p95 is the nearest-rank 95th percentile, over S1's chat families only;
  7. a box too small for a tier is refused before anything is downloaded (before any setup when it holds only
     that tier); the box rate is the real one;
  8. run_group settles every reservation on every way out (done, a failed step, a SystemExit, Ctrl-C, the cap),
     buys no new box after a stop, the cap, a refusal or two evictions that wrote nothing, and a timer closes the box
     at the cap; the run exits non-zero when a group or a tier did not finish;
  9. the merge refuses an adapter that did not load whole; preflight refuses MLX rows that cannot be compared.
Prints PASS or FAIL per check; exit 1 on any FAIL.

  GRUX_SRC=<Grux v3.1.0 Sources/Grux> GRUX_SRC_V3=<Grux v3.3.1 Sources/Grux> python3 recipe/s2/test_gguf_run.py
"""
import argparse
import fnmatch
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
if importlib.util.find_spec("daytona") is None:   # the box SDK is not needed with stub boxes
    sys.modules["daytona"] = types.SimpleNamespace(**{n: object for n in (
        "CreateSandboxFromImageParams", "Daytona", "GpuType", "Image", "ListSandboxesQuery", "Resources",
        "SessionExecuteRequest")})
import gguf_run as g  # noqa: E402
import holdout_v3 as h  # noqa: E402
import train_run as t  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


t.time.sleep = lambda s: None
REAL_SETS = dict(t.SETS)

# --------------------------------------------------------------------------- 6. p95
check("p95 of 1..100 is 95 (nearest rank)", g.p95(list(range(1, 101))) == 95, str(g.p95(list(range(1, 101)))))
check("p95 of 1..20 is 19", g.p95(list(range(20, 0, -1))) == 19, str(g.p95(list(range(20, 0, -1)))))
check("p95 of one value is that value; of none, None", g.p95([7.5]) == 7.5 and g.p95([]) is None)
pd = tempfile.mkdtemp()
pp = os.path.join(pd, "s1.jsonl")
with open(pp, "w") as f:
    for i in range(1, 21):   # chat seconds 1..20; dictation rows at 1000 s must not count
        f.write(json.dumps({"case": f"c{i}", "family": ["chat_tools", "chat_plain", "chat_honesty"][i % 3], "s": i}) + "\n")
        f.write(json.dumps({"case": f"d{i}", "family": "dictation", "s": 1000}) + "\n")
check("chat p95 reads S1's chat families only", g.chat_p95([pp]) == {"p95_s": 19, "n": 20}, str(g.chat_p95([pp])))

# --------------------------------------------------------------------------- 5. row names
MLX_TAGS = ["grux-qwen35-2b", "grux-qwen35-4b", "grux-qwen35-9b", "grux-qwen35-a3b"]
BASE_TAGS = [x.replace("grux-", "base-") for x in MLX_TAGS]
d = "/r"
theirs = {t.eval_path(d, m, s, k) for m in MLX_TAGS + BASE_TAGS for s in REAL_SETS for k in (1, 2, 3)}
ours = set()
for m in MLX_TAGS:
    gm = g.gguf_model(m)
    ours |= {t.eval_path(d, lab, s, k) for lab in (gm, gm + "-limit", gm + "-serial") for s in REAL_SETS for k in (1, 2, 3)}
ours |= {t.eval_path(d, f"shipped-{x}", s, k) for x in g.SHIPPED for s in REAL_SETS for k in (1, 2, 3)}
check("no GGUF, limited, serial or shipped row file has an MLX or base row's name", not (ours & theirs),
      str(sorted(ours & theirs)[:3]))
names = [os.path.basename(p) for p in ours]
check("... and none matches the eval-grux-* or eval-base-* globs readers use",
      not [n for n in names if fnmatch.fnmatch(n, "eval-grux-*") or fnmatch.fnmatch(n, "eval-base-*")])
check("... and every one is a single file name (no / from a registry tag)", all("/" not in n for n in names))
check("a registry tag's file name is flat, as eval_run names it",
      os.path.basename(t.eval_path(d, "gruxai.com/grux/grux-local:2b", "s1", 1)) == "eval-gruxai.com_grux_grux-local_2b-s1.jsonl")
check("an MLX row keeps the name every reader uses", t.eval_path(d, "grux-qwen35-2b", "holdout_v3", 2)
      == "/r/eval-grux-qwen35-2b-holdout_v3-pass2.jsonl")

# --------------------------------------------------------------------------- 7. rate, host
check("the H200 box rate is $6.5268/h (GPU plus 16 vCPU and 192 GB)", round(g.rate("H200", 16, 192), 4) == 6.5268,
      str(g.rate("H200", 16, 192)))
check("without the box's numbers the rate is GPU plus BOX_EXTRA", round(g.rate("H200"), 4) == 6.5268)
host = g.parse_host("32\n200000000000\n64000000000\n300000000000\n500000000000\n")
check("host RAM is the cgroup limit when it is lower than MemTotal", host["ram_gb"] == 64.0 and host["nproc"] == 32, str(host))
check("a 64 GB box is refused for the Pro, naming RAM", "64.0 GB RAM, needs 80" in (g.host_short(host, "pro") or ""))
check("... and accepted for the 9B", g.host_short(host, "9b") is None, str(g.host_short(host, "9b")))

# --------------------------------------------------------------------------- Modelfile
CONFIG = {"model_format": "gguf", "model_family": "qwen35", "file_type": "Q4_K_M", "renderer": "qwen3.5",
          "parser": "qwen3.5"}
PARAMS = {"presence_penalty": 1.5, "temperature": 1, "top_k": 20, "top_p": 0.95}
mf = g.library_modelfile(CONFIG, PARAMS, "{{ .Prompt }}", "/w/x.gguf")
check("the Modelfile carries the library's renderer, parser, template and every parameter",
      all(x in mf.splitlines() for x in ("FROM /w/x.gguf", "RENDERER qwen3.5", "PARSER qwen3.5",
                                         'TEMPLATE """{{ .Prompt }}"""', "PARAMETER presence_penalty 1.5",
                                         "PARAMETER top_k 20")), mf)
try:
    g.library_modelfile({k: v for k, v in CONFIG.items() if k != "parser"}, PARAMS, "", "/x")
    refused = False
except ValueError:
    refused = True
check("a library config without a parser is refused", refused)
show = {"modelfile": mf, "parameters": "presence_penalty 1.5\ntemperature 1\ntop_k 20\ntop_p 0.95",
        "details": {"format": "gguf", "quantization_level": "Q4_K_M"}}
Show = type("Show", (), {"post": lambda self, p, b, timeout=None: dict(show)})
check("an import that matches the library passes", g.check_import(Show(), "m", mf, "Q4_K_M") is None,
      str(g.check_import(Show(), "m", mf, "Q4_K_M")))
show["modelfile"] = mf.replace("PARSER qwen3.5", "")
check("an import without the library's parser is refused", "PARSER qwen3.5" in (g.check_import(Show(), "m", mf, "Q4_K_M") or ""))
show["modelfile"] = mf
show["details"] = {"format": "gguf", "quantization_level": "Q8_0"}
check("an import of the wrong quant is refused", "quantization Q8_0" in (g.check_import(Show(), "m", mf, "Q4_K_M") or ""))

PS = type("PS", (), {"get": lambda self, p, timeout=None: {"models": [{"name": "gguf-m:latest", "size": 10, "size_vram": 6}]}})
check("the GPU check finds a local model under its :latest name, and sees a partial load",
      g.on_gpu_whole(PS(), "gguf-m") == (False, {"size": 10, "size_vram": 6}), str(g.on_gpu_whole(PS(), "gguf-m")))

# --------------------------------------------------------------------------- 1. kill rule
LIM = g.limited_v3(2)
PROBE = next(c for c in h.cases() if "call" in c["steps"][0])
V3 = [c for c in LIM]
V1 = t.holdout.cases()[:3]
V2 = t.holdout_v2.cases()[:3]
S1 = [c for c in t.suite.cases() if c["family"].startswith("chat_")][:3] + \
     [c for c in t.suite.cases() if not c["family"].startswith("chat_")][:2]
MODEL = "gguf-m"
KEY = {}
for c in {x["id"]: x for x in V3 + [PROBE]}.values():
    for k, r in enumerate(h.perfect_raws(c)):
        KEY[json.dumps(h.request_v3(c, MODEL, k)["body"]["messages"])] = r
EMPTY = {"message": {"role": "assistant", "content": ""}}


class Perfect:
    def __init__(self, only_probe=False):
        self.calls, self.only_probe = 0, only_probe

    def post(self, path, body, timeout=None):
        self.calls += 1
        if self.only_probe and body.get("messages") != h.request_v3(PROBE, MODEL, 0)["body"]["messages"]:
            return EMPTY
        return KEY.get(json.dumps(body.get("messages")), EMPTY)

    def get(self, path, timeout=None):
        return {"models": []}


class Mute(Perfect):
    def post(self, path, body, timeout=None):
        self.calls += 1
        return EMPTY


class Gone:
    def post(self, path, body, timeout=None):
        raise ConnectionError("box gone")

    def get(self, path, timeout=None):
        raise ConnectionError("box gone")


def tier_with_mlx(pass_rate=True):
    td = tempfile.mkdtemp()
    for k in (1, 2, 3):
        with open(t.eval_path(td, "grux-m", "holdout_v3", k), "w") as f:
            for c in h.cases():   # the whole set, as the MLX run wrote it; the kill rule picks the limited jobs
                f.write(json.dumps({"case": c["id"], "family": c["family"], "pass": pass_rate}) + "\n")
    return td


A = argparse.Namespace(kill_jobs=2, kill_points=15.0, parallel=2, passes=1)
t.SETS = {"holdout_v3": (lambda: V3, None, None),
          "holdout": (lambda: V1, t.holdout.request_holdout, t.holdout.score_holdout),
          "holdout_v2": (lambda: V2, t.holdout_v2.request_v2, t.holdout_v2.score_v2),
          "s1": (lambda: S1, t.suite.request, t.suite.score),
          "holdout_v4": (lambda: V3, None, None)}   # train_run scores v4 too; a GGUF tier is compared on GGUF_SETS only
good = tier_with_mlx()
st = g.evaluate(Perfect(), MODEL, good, "grux-m", A)
check("a good limited pass goes on to full scoring", st == "scored", str(g.status(good)))
check("... and evaluate leaves the final 'scored' to run_tier (the serial chat pass is not in yet)",
      g.status(good) == {}, str(g.status(good)))
check("... which writes every set's rows under the GGUF's own name",
      all(len(g.rows(t.eval_path(good, MODEL, n, 1))) == len(cs) for n, cs in (("holdout_v3", V3), ("holdout", V1),
                                                                                 ("holdout_v2", V2), ("s1", S1))))
check("... and no held-out v4 rows, in a file or in the manifest's scores (gguf_run.GGUF_SETS)",
      not os.path.exists(t.eval_path(good, MODEL, "holdout_v4", 1))
      and not any(k.startswith("holdout_v4") for k in json.load(open(os.path.join(good, "gguf-manifest.json")))["scores"]))
check("... the limited rows in their own file", len(g.rows(t.eval_path(good, MODEL + "-limit", "holdout_v3", 1))) == len(LIM))
man = json.load(open(os.path.join(good, "gguf-manifest.json")))
check("... and the manifest records the kill rule, the scores and a chat p95 with its parallel",
      man["kill_rule"]["kill"] is False and "s1-pass1" in man["scores"] and man["chat_p95"]["gguf"]["n"] == 3
      and man["chat_p95"]["parallel"] == 2, json.dumps(man)[:300])

bad = tier_with_mlx()
probe_only = Perfect(only_probe=True)
st = g.evaluate(probe_only, MODEL, bad, "grux-m", A)
check("a tool call on the probe but a bad limited pass kills the tier", st == "killed" and g.status(bad)["stage"] == "killed",
      str(g.status(bad)))
check("... naming both numbers and the threshold", "more than 15.0 points under" in g.status(bad)["why"], g.status(bad).get("why"))
check("... and scores nothing else", not any(os.path.exists(t.eval_path(bad, MODEL, n, 1)) for n in t.SETS))

mute = tier_with_mlx()
st = g.evaluate(Mute(), MODEL, mute, "grux-m", A)
check("no tool call on the probe kills the tier before the limited pass",
      st == "killed" and "tool-call probe" in g.status(mute)["why"]
      and not os.path.exists(t.eval_path(mute, MODEL + "-limit", "holdout_v3", 1)), str(g.status(mute)))

even = tier_with_mlx(pass_rate=False)   # the MLX build failed these jobs too: a GGUF that fails them is no worse
st = g.evaluate(probe_only, MODEL, even, "grux-m", A)
check("a limited pass as low as its MLX build is not killed", st == "scored", str(g.status(even)))

one = tier_with_mlx()
with open(t.eval_path(one, "grux-m", "holdout_v3", 2), "w") as f:   # pass 2 cut short: not read with --passes 1
    f.write(json.dumps({"case": h.cases()[0]["id"], "pass": True}) + "\n")
st = g.evaluate(Perfect(), MODEL, one, "grux-m", A)
check("with --passes 1 the kill rule reads only MLX pass 1, as preflight checks it", st == "scored", str(g.status(one)))
nomlx = tempfile.mkdtemp()
st = g.evaluate(Perfect(), MODEL, nomlx, "grux-m", A)
check("no MLX rows to compare refuses the comparison (killed, not scored unchecked)",
      st == "killed" and "no MLX row" in g.status(nomlx)["why"], str(g.status(nomlx)))

gone = tier_with_mlx()
try:
    g.evaluate(Gone(), MODEL, gone, "grux-m", A)
    raised = False
except t.BoxGone:
    raised = True
check("a box that is gone during the probe raises BoxGone and is not a kill", raised and g.status(gone) == {},
      str(g.status(gone)))



class FlakyProbe(Perfect):
    """The first request fails (a cold load, a proxy 502); the box answers throughout."""
    def post(self, path, body, timeout=None):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("502 from the preview proxy")
        return KEY.get(json.dumps(body.get("messages")), EMPTY)


ok_, detail = g.tool_probe(FlakyProbe(), MODEL)
check("the tool-call probe retries once, as every scored request does", ok_, detail)
NoName = type("NoName", (Perfect,), {"post": lambda self, p, b, timeout=None: {
    "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"arguments": {}}}]}}})
check("a tool call without a function name is not a tool call", not g.tool_probe(NoName(), MODEL)[0])


class Spilled(Perfect):
    """Perfect answers; /api/ps shows the model only partly in GPU memory, under its :latest name."""
    def get(self, path, timeout=None):
        if path == "/api/ps":
            return {"models": [{"name": MODEL + ":latest", "size": 30, "size_vram": 20}]}
        return {"models": []}


sp = tier_with_mlx()
g.evaluate(Spilled(), MODEL, sp, "grux-m", A)
man = json.load(open(os.path.join(sp, "gguf-manifest.json")))
check("the GPU is sampled after the limited v3 pass and after full v3 (the 65,536-token loads)",
      man.get("gpu_whole_v3_limit") is False and man.get("gpu_whole_v3_full") is False, str(man.get("ps_v3_limit")))
check("... and a partial load is a WARNING in the tier's log",
      "WARNING not wholly in GPU memory at v3_limit" in open(os.path.join(sp, "run.log")).read())


class ProbeThenDown(Perfect):
    """The probe gets its tool call; every later request errors while the box still answers."""
    def post(self, path, body, timeout=None):
        if body.get("messages") == h.request_v3(PROBE, MODEL, 0)["body"]["messages"]:
            return KEY[json.dumps(body["messages"])]
        raise ConnectionError("model error")


dmg = tier_with_mlx()
st = g.evaluate(ProbeThenDown(), MODEL, dmg, "grux-m", A)
check("a model that fails job after job on a live box is 'damaged', which a rerun retries",
      st == "damaged" and "damaged" not in g.FINAL, str(g.status(dmg)))

# --------------------------------------------------------------------------- 9. adapter proof
import gguf_box  # noqa: E402
KEYS = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
check("an adapter that loaded whole passes", gguf_box.adapter_problems(KEYS, KEYS, True, 0) == [])
check("an adapter weight that matched no module is refused",
      "matched no module" in " ".join(gguf_box.adapter_problems(KEYS, KEYS[:1], True, 0)))
check("an adapter whose lora_B is all zero is refused", "every lora_B is zero" in " ".join(
    gguf_box.adapter_problems(KEYS, KEYS, False, 0)))
check("an adapter that loaded other values is refused", "values other than" in " ".join(
    gguf_box.adapter_problems(KEYS, KEYS, True, 2)))
bc = tempfile.mkdtemp()
os.makedirs(os.path.join(bc, ".cache", "huggingface", "download"))
open(os.path.join(bc, ".cache", "huggingface", "download", "config.json.metadata"), "w").write("abc123\netag\n")
check("the base's Hugging Face commit is read from snapshot_download's metadata", gguf_box.base_commit(bc) == "abc123")

# --------------------------------------------------------------------------- 2. digests
MAN = {t_: json.dumps({"tier": t_}).encode() for t_ in g.SHIPPED}
PIN = json.dumps({"object": {"sha": g.LLAMA_CPP_COMMIT}}).encode()


def fetch(url, accept=None):
    if "api.github.com" in url:
        return PIN
    for t_, b in MAN.items():
        if url.endswith(f"/manifests/{t_}"):
            return b
    raise OSError("unexpected " + url)


meter = tempfile.NamedTemporaryFile(delete=False).name
charter = tempfile.NamedTemporaryFile(delete=False).name
os.environ.update(GRUX_SWING_CHARTER=charter, DAYTONA_API_KEY="stub")
PA = argparse.Namespace(tiers=[], train_root=tempfile.mkdtemp(), meter=meter, shipped=True, quant={},
                        gpu_of={"shipped": "H200"})
t.SETS = dict(REAL_SETS)
bad_, facts = g.preflight(PA, fetch=fetch)
check("preflight refuses every shipped tag whose registry digest is not the shipped one",
      sum("is not the shipped" in x for x in bad_) == len(g.SHIPPED), str(bad_))
saved = dict(g.SHIPPED)
g.SHIPPED.update({t_: hashlib.sha256(b).hexdigest()[:12] for t_, b in MAN.items()})
bad_, facts = g.preflight(PA, fetch=fetch)
check("... and passes when every digest matches (the digest is sha256 of the manifest bytes)", bad_ == [], str(bad_))
PIN = json.dumps({"object": {"sha": "0" * 40}}).encode()
bad_, facts = g.preflight(PA, fetch=fetch)
check("preflight refuses a llama.cpp tag that moved off the pinned commit", any("now points at" in x for x in bad_), str(bad_))
PIN = json.dumps({"object": {"sha": g.LLAMA_CPP_COMMIT}}).encode()
os.environ.pop("GRUX_SWING_CHARTER")
bad_, facts = g.preflight(PA, fetch=fetch)
check("preflight refuses without a charter", any("GRUX_SWING_CHARTER" in x for x in bad_), str(bad_))
os.environ["GRUX_SWING_CHARTER"] = charter
LIB = {"config": json.dumps(CONFIG).encode(), "params": json.dumps(PARAMS).encode(), "template": b"{{ .Prompt }}"}
LIB_MAN = json.dumps({"config": {"digest": "sha256:c"}, "layers": [
    {"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:p"},
    {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:t"}]}).encode()


def fetch_lib(url, accept=None):
    if "/library/qwen3.5/manifests/" in url:
        return LIB_MAN
    for d_, k in (("sha256:c", "config"), ("sha256:p", "params"), ("sha256:t", "template")):
        if url.endswith("/blobs/" + d_):
            return LIB[k]
    return fetch(url, accept)


tr2 = tempfile.mkdtemp()
os.makedirs(os.path.join(tr2, "2b"))
open(os.path.join(tr2, "2b", "adapter.tar"), "wb").write(b"x")
json.dump({"model": "Qwen/Qwen3.5-2B", "tag": "grux-m"}, open(os.path.join(tr2, "2b", "manifest.json"), "w"))
for name in REAL_SETS:
    for k in (1, 2):
        with open(t.eval_path(os.path.join(tr2, "2b"), "grux-m", name, k), "w") as f:
            if name == "holdout_v3":
                for c in (h.cases() if k == 1 else h.cases()[:10]):   # pass 2 is cut short
                    f.write(json.dumps({"case": c["id"], "pass": True}) + "\n")
PB = argparse.Namespace(tiers=["2b"], train_root=tr2, meter=meter, shipped=False, quant={"2b": "Q4_K_M"},
                        gpu_of={"2b": "H200"}, kill_jobs=2, passes=3)
bad_, facts = g.preflight(PB, fetch=fetch_lib)
check("preflight refuses MLX row files missing for a pass the GGUF will be compared on",
      any("MLX row files missing" in x and "pass3" in x for x in bad_), str(bad_))
check("... and an MLX v3 pass file without a row for every kill-rule job",
      any("holdout_v3-pass2.jsonl has no row for" in x for x in bad_), str(bad_))
check("... and builds the Modelfile from the library's config, params and template",
      "RENDERER qwen3.5" in facts["modelfile"].get("2b", ""), str(facts["modelfile"]))
PA.gpu_of = {"pro": "RTX-5090", "shipped": "H200"}
bad_, facts = g.preflight(PA, fetch=fetch)
check("preflight accepts RTX-5090 and H200 as Daytona GPU types", not any(x.startswith("GPU ") for x in bad_), str(bad_))
PA.gpu_of = {"pro": "RTX-5900"}
bad_, facts = g.preflight(PA, fetch=fetch)
check("... and refuses a GPU type the SDK does not have", any("RTX-5900" in x and "GpuType" in x for x in bad_), str(bad_))
PA.gpu_of = {"shipped": "H200"}
g.SHIPPED.clear()
g.SHIPPED.update(saved)


class Shipped:
    """A box Ollama that pulls and serves a registry tag, listing it under `digest`."""
    def __init__(self, digest):
        self.digest, self.scored = digest, 0

    def get(self, path, timeout=None):
        if path == "/api/tags":
            return {"models": [{"name": g.SHIPPED_TAG.format("2b"), "digest": self.digest}]}
        return {"models": []}

    def post(self, path, body, timeout=None):
        if path == "/api/pull":
            return {"status": "success"}
        if path == "/api/generate":
            return {"done": True}
        self.scored += 1
        return EMPTY

    def _req(self, *a, **k):
        return {}


t.SETS = {"holdout": (lambda: V1, t.holdout.request_holdout, t.holdout.score_holdout),
          "holdout_v2": (lambda: V2, t.holdout_v2.request_v2, t.holdout_v2.score_v2),
          "s1": (lambda: S1, t.suite.request, t.suite.score), "holdout_v3": (lambda: V3, None, None)}
g.SHIPPED.clear()
g.SHIPPED["2b"] = saved["2b"]
SA = argparse.Namespace(shipped_run=tempfile.mkdtemp(), parallel=2, passes=2)
wrong = Shipped("deadbeef" * 8)
g.run_shipped(wrong, SA, SA.shipped_run)
st = g.status(SA.shipped_run).get("shipped-2b", {})
check("on the box, a shipped tag listed under another digest is refused and nothing is scored",
      st.get("stage") == "refused" and wrong.scored == 0 and not [n for n in os.listdir(SA.shipped_run) if n.startswith("eval-")],
      str(st))
right = Shipped(saved["2b"] + "0" * 52)
SA.shipped_run = tempfile.mkdtemp()
g.run_shipped(right, SA, SA.shipped_run)
st = g.status(SA.shipped_run).get("shipped-2b", {})
check("... and the right digest is scored on v1, v2 and S1 only, every pass, under shipped-<tier>",
      st.get("stage") == "scored" and sorted(n for n in os.listdir(SA.shipped_run) if n.startswith("eval-")) == sorted(
          os.path.basename(t.eval_path(SA.shipped_run, "shipped-2b", s, k)) for s in g.SHIPPED_SETS for k in (1, 2)),
      str(sorted(os.listdir(SA.shipped_run))))
g.SHIPPED.clear()
g.SHIPPED.update(saved)
t.SETS = dict(REAL_SETS)


# --------------------------------------------------------------------------- 3, 4. disk, base packs
class Box:
    """A training box that records every command; `get` writes `payload` (or raises `get_error`)."""
    def __init__(self, payload=b"x" * 100, get_error=None, sizes=None):
        self.cmds, self.gets, self.puts = [], [], []
        self.payload, self.get_error, self.sizes = payload, get_error, sizes or {}
        self.info = {"id": "stubbox0", "gpu_type": "GpuType.H200", "cpu": 16, "memory": 192, "disk": 512}

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def sh(self, cmd, timeout=None):
        self.cmds.append(cmd)
        if cmd.startswith("ollama list"):
            return 0, self.sizes.get("list", "2.6 GB")
        if cmd.startswith("stat -c %s"):
            return 0, str(self.sizes.get("stat", len(self.payload)))
        if cmd.startswith("python /workspace/pack_ollama.py"):
            return 0, "packed 3 blobs"
        return 0, ""

    def run_long(self, name, cmd, max_s=0, poll_s=0):
        self.cmds.append(cmd)
        return 0, "ok\nEXIT=0"

    def put(self, local, remote):
        self.puts.append(remote)

    def get(self, remote, local, timeout=None):
        self.gets.append(remote)
        open(local, "wb").write(self.payload[: len(self.payload) // 2] if self.get_error else self.payload)
        if self.get_error:   # the write failed partway: half a file is on disk, as a full disk leaves it
            raise RuntimeError(self.get_error)
        return len(self.payload)

    def ollama(self):
        return Perfect()

    def gone(self):
        return False


def run_one_box(box, floor):
    rd = tempfile.mkdtemp()
    a = argparse.Namespace(gpu="H200", max_minutes=60, owner="grux", data=os.devnull, model="Qwen/Qwen3.5-2B",
                           until="all", skip_base=False, pull=True, parallel=2, passes=1)
    saved_ = (t.TrainBox, t.train, t.score_model, t.DISK_FLOOR)
    t.TrainBox = lambda *x, **k: box
    t.train = lambda *x, **k: True
    t.score_model = lambda *x, **k: None
    t.DISK_FLOOR = floor
    try:
        status_, cost = t.one_box(a, rd, "grux-qwen35-2b", "base-qwen35-2b", reservation="rid")
    finally:
        t.TrainBox, t.train, t.score_model, t.DISK_FLOOR = saved_
    return status_, rd


box = Box()
st, rd = run_one_box(box, floor=0)
check("with room, the trained model's pack is downloaded", st == "done" and box.gets == ["/workspace/grux-qwen35-2b.tar"]
      and os.path.exists(os.path.join(rd, "ollama-grux-qwen35-2b.tar")), f"{st} {box.gets}")
check("the base model's pack is never packed or downloaded",
      not any("base-qwen35-2b" in c for c in box.cmds if "pack_ollama" in c) and not any("base-" in x for x in box.gets),
      str([c for c in box.cmds if "pack" in c]))
check("... though the base is still built and scored on the box", any("create base-qwen35-2b" in c for c in box.cmds))
box = Box()
st, rd = run_one_box(box, floor=1e18)
log = open(os.path.join(rd, "run.log")).read()
check("short of disk, the pack is skipped before packing and the run is done", st == "done" and box.gets == []
      and not any("pack_ollama" in c for c in box.cmds) and "SKIPPED before packing" in log, f"{st} {box.gets}")
check("... naming what it needed and what was free", "plus a 1000000000 GB floor" in log, log[-300:])
box = Box(get_error="download /workspace/grux-qwen35-2b.tar: Write failed: [Errno 28] No space left on device")
st, rd = run_one_box(box, floor=0)
check("a download that fails after scoring leaves the run done, with no partial file",
      st == "done" and not [n for n in os.listdir(rd) if n.startswith("ollama-")], f"{st} {os.listdir(rd)}")
check("... and the failure is in the log", "FAILED, the scores stand" in open(os.path.join(rd, "run.log")).read())
check("... and pull.json records it", json.load(open(os.path.join(rd, "pull.json")))["pull"].startswith("failed:"))
box = Box()
rd_ = tempfile.mkdtemp()
os.makedirs(os.path.join(rd_, "pull.json"))   # writing pull.json raises IsADirectoryError, an OSError
saved_mk = tempfile.mkdtemp
tempfile.mkdtemp = lambda: rd_
st, rd = run_one_box(box, floor=0)
tempfile.mkdtemp = saved_mk
check("a pull.json that cannot be written leaves a scored run done", st == "done", st)
box = Box(sizes={"list": "0.1 GB", "stat": 10 ** 18})
st, rd = run_one_box(box, floor=0)
check("a pack that turns out bigger than the free disk is skipped after packing, not downloaded",
      st == "done" and box.gets == [] and any("pack_ollama" in c for c in box.cmds)
      and json.load(open(os.path.join(rd, "pull.json")))["pull"].startswith("skipped:"), f"{st} {box.gets}")

GA = argparse.Namespace(pull_pro=False)
payload = b"gguf bytes"
sha = hashlib.sha256(payload).hexdigest()
box = Box(payload=payload)
td = tempfile.mkdtemp()
check("the Pro GGUF is not pulled by default", g.fetch_gguf(box, "pro", td, "/w/g.gguf", sha, GA).startswith("not pulled")
      and box.gets == [])
check("a download leaves at least 20 GB free here (the 2026-11-18 rule)", t.DISK_FLOOR == 20e9, t.DISK_FLOOR)
_floor = t.DISK_FLOOR
t.DISK_FLOOR = 1e18
r = g.fetch_gguf(box, "2b", td, "/w/g.gguf", sha, GA)
t.DISK_FLOOR = _floor
check("short of disk, a GGUF is skipped with the reason", r.startswith("skipped: needs") and box.gets == [], r)
t.DISK_FLOOR = 0
r = g.fetch_gguf(box, "2b", td, "/w/g.gguf", sha, GA)
check("with room, a GGUF is pulled and its sha256 checked", r.startswith("pulled") and
      open(os.path.join(td, "g.gguf"), "rb").read() == payload, r)
r = g.fetch_gguf(Box(payload=payload, get_error="Write failed: [Errno 28]"), "4b", td, "/w/f.gguf", sha, GA)
check("a GGUF download that fails partway leaves no .part", r.startswith("failed:") and not os.path.exists(
    os.path.join(td, "f.gguf.part")) and not os.path.exists(os.path.join(td, "f.gguf")), r)
r = g.fetch_gguf(Box(payload=b"other"), "4b", td, "/w/h.gguf", sha, GA)
t.DISK_FLOOR = _floor
check("a GGUF whose sha256 differs is not kept", r.startswith("failed: sha256") and not os.path.exists(os.path.join(td, "h.gguf")), r)


# --------------------------------------------------------------------------- 7. host refusal before any download
class SmallBox(Box):
    def sh(self, cmd, timeout=None):
        self.cmds.append(cmd)
        if cmd == g.HOST_CMD:
            return 0, "16\n64000000000\nmax\n200000000000\n240000000000"
        return 0, ""


root = tempfile.mkdtemp()
tr = tempfile.mkdtemp()
os.makedirs(os.path.join(tr, "pro"))
json.dump({"tag": "grux-qwen35-a3b"}, open(os.path.join(tr, "pro", "manifest.json"), "w"))
RA = argparse.Namespace(train_root=tr, quant={"pro": "Q4_K_M"})
sb = SmallBox()
st = g.run_tier(sb, Perfect(), "pro", RA, {"library": {}, "modelfile": {"pro": mf}}, {}, root)
check("a box without 80 GB RAM and 250 GB disk refuses the Pro before any upload or download",
      st == "refused" and sb.puts == [] and sb.gets == [] and not any("snapshot_download" in c for c in sb.cmds),
      f"{st} {sb.puts} {sb.cmds}")
check("... recording both numbers for the lead", "64.0 GB RAM, needs 80" in g.status(os.path.join(tr, "pro"))["why"]
      and "200.0 GB free disk, needs 250" in g.status(os.path.join(tr, "pro"))["why"], str(g.status(os.path.join(tr, "pro"))))



class ConvertFails(SmallBox):
    """A big enough box whose converter exits 1: the tier stops, the box (and the next tier) goes on."""
    def sh(self, cmd, timeout=None):
        if cmd == g.HOST_CMD:
            self.cmds.append(cmd)
            return 0, "32\n400000000000\nmax\n500000000000\n512000000000"
        return Box.sh(self, cmd, timeout)

    def run_long(self, name, cmd, max_s=0, poll_s=0):
        self.cmds.append(cmd)
        if name.startswith("convert"):
            return 1, "KeyError: model.layers.0.mtp\nEXIT=1"
        return 0, 'TRAIN DONE\nMERGE FACTS {"device": "cuda"}\nEXIT=0'


os.makedirs(os.path.join(tr, "2b"))
json.dump({"tag": "grux-qwen35-2b"}, open(os.path.join(tr, "2b", "manifest.json"), "w"))
RA.quant["2b"] = "Q4_K_M"
cb = ConvertFails()
st = g.run_tier(cb, Perfect(), "2b", RA, {"library": {}, "modelfile": {"2b": mf}}, {}, root)
check("a converter that fails stops that tier with its error, not the box", st == "stopped"
      and "convert-2b failed" in g.status(os.path.join(tr, "2b"))["why"] and not any("create" in c for c in cb.cmds),
      f"{st} {g.status(os.path.join(tr, '2b'))}")


# --------------------------------------------------------------------------- 8. run_group: meter, cap, stop
METER_LOG = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
fake_meter = tempfile.NamedTemporaryFile("w", delete=False, suffix=".py")
fake_meter.write("import os, sys\nlog = os.environ['FAKE_METER_LOG']\nopen(log, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
                 "if sys.argv[1] == 'reserve':\n    if os.environ.get('FAKE_METER_REFUSE'):\n        sys.exit(3)\n"
                 "    print('rid-%d' % sum(1 for l in open(log) if l.startswith('reserve')))\n")
fake_meter.close()
os.environ["FAKE_METER_LOG"] = METER_LOG
BIG = "32\n400000000000\nmax\n500000000000\n512000000000"
SMALL = "16\n64000000000\nmax\n200000000000\n240000000000"


class StubTB:
    made, host, gone_, stop_on_enter, stop_in_boot = [], BIG, False, False, False

    def __init__(self, gpus, label, max_minutes, log=None, owner=None, reservation=None):
        self.info = {"id": f"box{len(StubTB.made)}-0000", "gpu_type": "GpuType.H200", "cpu": 16, "memory": 192,
                     "disk": 512}
        self.released = self.exited = False
        self.runs, self.reservation = [], reservation
        self.lease = types.SimpleNamespace(release=self._release)
        StubTB.made.append(self)

    def _release(self):
        self.released = True

    def __enter__(self):
        if StubTB.stop_on_enter:   # Ctrl-C while the box was being created
            g.STOP.set()
        return self

    def __exit__(self, *e):
        self.exited = True
        return False

    def _alive(self):
        if self.released:
            raise ConnectionError("box deleted")

    def sh(self, cmd, timeout=None):
        self._alive()
        return (0, StubTB.host) if cmd == g.HOST_CMD else (0, "")

    def run_long(self, name, cmd, max_s=0, poll_s=0):
        self._alive()
        self.runs.append(name)
        if StubTB.stop_in_boot and name == "boot":   # Ctrl-C during BOOT: only the watcher can close this box
            g.STOP.set()
            for _ in range(500):
                self._alive()
                __import__("threading").Event().wait(0.01)
        return 0, "ok\nEXIT=0"

    def put(self, *x):
        self._alive()

    def ollama(self):
        return Perfect()

    def gone(self):
        return self.released or StubTB.gone_


def meter_calls():
    lines = open(METER_LOG).read().splitlines()
    return sum(l.startswith("reserve") for l in lines), sum(l.startswith("settle") for l in lines)


def group(run_tier, tiers=("2b",), cap=30.0, host=BIG, gone=False, shipped=False, final=()):
    open(METER_LOG, "w").close()
    StubTB.made, StubTB.host, StubTB.gone_ = [], host, gone
    a = argparse.Namespace(run=tempfile.mkdtemp(), train_root=tempfile.mkdtemp(), shipped_run=tempfile.mkdtemp(),
                           meter=fake_meter.name, meter_lane="test", max_minutes=60, owner="test", gpu="H200",
                           pro_gpu="RTX-5090", cap=cap, pro_cap=cap)
    for tier in tiers:
        os.makedirs(os.path.join(a.train_root, tier))
        json.dump({"tag": "grux-m"}, open(os.path.join(a.train_root, tier, "manifest.json"), "w"))
        if tier in final:
            g.set_status(os.path.join(a.train_root, tier), "scored", "every set scored")
    g.attach_spend(a)
    saved_ = (g.TrainBox, g.run_tier, g.run_shipped)
    g.TrainBox, g.run_tier, g.run_shipped = StubTB, run_tier, (lambda *x: None)
    try:
        return g.run_group(a, "RTX-5090" if tiers == ("pro",) else "H200", list(tiers), shipped, {}, a.run), a
    finally:
        g.TrainBox, g.run_tier, g.run_shipped = saved_


st, a_ = group(lambda *x: "scored")
check("a group that finishes is done, with one reservation settled and the box closed",
      st == "done" and meter_calls() == (1, 1) and all(b.exited for b in StubTB.made), f"{st} {meter_calls()}")


def boom(*x):
    raise RuntimeError("planted step failure")


st, a_ = group(boom)
check("a failed step ends the group without a second box, its reservation settled",
      st.startswith("failed: RuntimeError") and meter_calls() == (1, 1), f"{st} {meter_calls()}")


def exits(*x):
    raise SystemExit("planted: holdout_v3.wire_tools on a wrong tree")


try:
    group(exits)
    escaped = False
except SystemExit:
    escaped = True
check("a SystemExit inside a box still settles its reservation and closes the box",
      escaped and meter_calls() == (1, 1) and StubTB.made[0].exited, str(meter_calls()))


def interrupted(box, *x):
    box.released = True   # the lease's signal handler deleted the box
    g.STOP.set()
    raise ConnectionError("box deleted")


st, a_ = group(interrupted)
g.STOP.clear()
check("Ctrl-C mid-box: the group stops, buys no new box, and settles", st == "stopped" and meter_calls() == (1, 1),
      f"{st} {meter_calls()}")
g.STOP.set()
st, a_ = group(lambda *x: "scored")
g.STOP.clear()
check("... and a group not yet started reserves nothing", st == "stopped" and meter_calls() == (0, 0))


def slow(box, *x):
    for _ in range(200):
        box._alive()
        t.time.time()
        __import__("threading").Event().wait(0.01)
    return "scored"


g.MIN_BOX_HOURS = 0   # a cap that leaves about 2 seconds of box after teardown
st, a_ = group(slow, cap=0.33)
g.MIN_BOX_HOURS = 0.25
check("the cap timer closes the box mid-tier; the group stops at 'cap' without another box, and settles",
      st == "cap" and StubTB.made[0].released and meter_calls() == (1, 1), f"{st} {meter_calls()}")
st, a_ = group(lambda *x: "scored", cap=1.0)
check("a cap with less than one useful box slice left (teardown plus 15 minutes) buys no box",
      st == "cap" and meter_calls() == (0, 0), f"{st} {meter_calls()}")
StubTB.stop_on_enter = True
st, a_ = group(lambda *x: "scored")
StubTB.stop_on_enter = False
g.STOP.clear()
check("Ctrl-C while the box is being created: no setup runs, the box is closed, and the group settles",
      st == "stopped" and StubTB.made[0].runs == [] and StubTB.made[0].exited and meter_calls() == (1, 1),
      f"{st} {StubTB.made[0].runs} {meter_calls()}")
StubTB.stop_in_boot = True
st, a_ = group(lambda *x: "scored")
StubTB.stop_in_boot = False
g.STOP.clear()
check("Ctrl-C during BOOT: the STOP watcher closes the box, the group stops and settles, no second box",
      st == "stopped" and StubTB.made[0].released and meter_calls() == (1, 1), f"{st} {meter_calls()}")
st, a_ = group(lambda *x: "scored", final=("2b",))
check("a group whose every tier is final buys no box", st == "done" and meter_calls() == (0, 0) and StubTB.made == [],
      f"{st} {meter_calls()}")
import inspect  # noqa: E402
import signal  # noqa: E402
order = []
signal.signal(signal.SIGUSR1, lambda n, f: order.append(("prior", g.STOP.is_set())))
g.install_stop((signal.SIGUSR1,))
os.kill(os.getpid(), signal.SIGUSR1)
g.STOP.clear()
check("on a signal STOP is set first, then the earlier handler (lease's box delete) runs",
      order == [("prior", True)], str(order))
src = inspect.getsource(g.main)
check("main installs lease's handlers, then the STOP handler, before any group starts",
      -1 < src.find("lease.install()") < src.find("install_stop()") < src.find("pool.submit("))
st, a_ = group(lambda *x: "scored", tiers=("pro",), host=SMALL)
check("a 5090 too small for the Pro is refused before any setup, settled, and not bought again",
      st == "refused" and StubTB.made[0].runs == [] and meter_calls() == (1, 1)
      and g.status(os.path.join(a_.train_root, "pro"))["stage"] == "refused", f"{st} {StubTB.made[0].runs}")


def lost(box, *x):
    raise ConnectionError("box evicted")


st, a_ = group(lost, gone=True)
check("two evictions in a row that wrote nothing stop the group at two boxes, both settled",
      st == "evicted twice without progress" and meter_calls() == (2, 2), f"{st} {meter_calls()}")
os.environ["FAKE_METER_REFUSE"] = "1"
st, a_ = group(lambda *x: "scored")
os.environ.pop("FAKE_METER_REFUSE")
check("a meter refusal creates no box", st == "meter refused" and StubTB.made == [])
check("the run is not done when a group or a tier did not finish",
      len(g.verdict({"H200": "done", "RTX-5090": "refused"}, {"2b": {"stage": "scored"},
                                                          "pro": {"stage": "refused", "why": "host"}})) == 2
      and g.verdict({"H200": "done"}, {"2b": {"stage": "killed"}}) == []
      and g.verdict({"H200": "done"}, {"4b": {}}) != []
      and g.verdict({"H200": "done"}, {"2b": {"stage": "stopped", "why": "import"}}) != []
      and g.verdict({"H200": "done"}, {}, {"shipped-2b": {"stage": "refused", "why": "digest"}}) != [])


# --------------------------------------------------------------------------- scored is final only when the tier is
tr3 = tempfile.mkdtemp()
os.makedirs(os.path.join(tr3, "pro"))
json.dump({"tag": "grux-m"}, open(os.path.join(tr3, "pro", "manifest.json"), "w"))
TA = argparse.Namespace(train_root=tr3, quant={"pro": "Q4_K_M"}, serial_chat=["pro"], pull_pro=False, passes=1)
saved_fns = (g.build_tier, g.check_import, g.probe, g.evaluate, g.serial_chat)
g.build_tier = lambda *x: {"gguf": "/w/g.gguf", "gguf_sha256": "0" * 64}
g.check_import = lambda *x: None
g.probe = lambda *x, **k: ({"runner": "llamacpp"}, None)
g.evaluate = lambda *x: "scored"


def lost_in_serial(*x):
    raise t.BoxGone("Ollama stopped answering during the serial chat pass")


g.serial_chat = lost_in_serial
bigbox = ConvertFails()
try:
    g.run_tier(bigbox, Perfect(), "pro", TA, {"library": {}, "modelfile": {"pro": mf}}, {}, root)
    raised = False
except t.BoxGone:
    raised = True
st_ = g.status(os.path.join(tr3, "pro"))
check("a box lost during the Pro's serial chat pass leaves the tier unfinished, not final", raised
      and st_.get("stage") not in g.FINAL and g.has_work(TA, ["pro"], False), str(st_))
check("... so the run does not exit 0", g.verdict({"RTX-5090": "done"}, {"pro": st_}) != [])
g.serial_chat = lambda *x: None
st = g.run_tier(ConvertFails(), Perfect(), "pro", TA, {"library": {}, "modelfile": {"pro": mf}}, {}, root)
check("... and once the serial chat pass is in, the tier is scored and final",
      st == "scored" and g.status(os.path.join(tr3, "pro"))["stage"] == "scored", str(g.status(os.path.join(tr3, "pro"))))
g.build_tier, g.check_import, g.probe, g.evaluate, g.serial_chat = saved_fns

print("SELF-TEST", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
sys.exit(1 if FAILS else 0)
