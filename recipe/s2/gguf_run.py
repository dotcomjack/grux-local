#!/usr/bin/env python3
"""GGUF builds of trained Grux Local tiers, scored against their nvfp4 (MLX runtime) builds on the same sets.

Per tier, on a leased spot box (several tiers share one box, one after another): upload the trained adapter, download
the public base from Hugging Face, merge (s2/gguf_box.py, box_train's own merge and finalize), convert the merged
model with llama.cpp's convert_hf_to_gguf.py at LLAMA_CPP_TAG, quantize (Q4_K_M by default, --quant per tier), and
import it into the box's Ollama with the Modelfile of Ollama's library qwen3.5 GGUF of the same size, read from the
registry (renderer, parser, template and parameters, never typed here). Then, before any full scoring:

  kill rule: one Grux turn must come back as a tool call, and a limited held-out v3 pass (--kill-jobs per family)
  must score within --kill-points of the MLX build on the same jobs. A tier that fails stops there with the reason in
  gguf-status.json; a template or parser mismatch costs minutes, not a full scoring run.

A tier that passes is scored with train_run.score_model on held-out v3, v1, v2 and S1, --passes times, think false
(as Grux sends it). Rows go in the tier's run dir as eval-gguf-<tag>-<set>[-pass<k>].jsonl, never an MLX row's name.
The box that holds the small tiers also scores the shipped registry tags on v1, v2 and S1 (digest checked), rows in
--shipped-run. Every box is reserved and settled through the spend meter (GRUX_SWING_CHARTER names the charter).

  DAYTONA_API_KEY=<your key> GRUX_SWING_CHARTER=<charter> python s2/gguf_run.py --run results/gguf-<stamp> \\
    --train-root results/train-v3-20261116 --meter <meter cli> [--preflight-only]
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import holdout_v3  # noqa: E402
import suite  # noqa: E402
import train_run  # noqa: E402  (sets suite.THINK = False: Grux sends think false on every call)
from eval_run import BOX_EXTRA, PRICE, log, manifest_url, probe  # noqa: E402
from train_box import BOOT, OLLAMA_VERSION, TrainBox  # noqa: E402

# Ollama 0.40.8 builds its llama.cpp runner from this exact tag (LLAMA_CPP_VERSION in its source at v0.40.8), and the
# tag's converter registers Qwen3_5ForConditionalGeneration (qwen35) and Qwen3_5MoeForConditionalGeneration
# (qwen35moe): the GGUF is written by the same llama.cpp that loads it.
LLAMA_CPP_TAG = "b11738"
LLAMA_CPP_COMMIT = "f174553e1621c6a22a64e7bfcbb58d3a5c49e6fb"
TIERS = {"2b": ("Qwen/Qwen3.5-2B", "qwen3.5:2b"), "4b": ("Qwen/Qwen3.5-4B", "qwen3.5:4b"),
         "9b": ("Qwen/Qwen3.5-9B", "qwen3.5:9b"), "pro": ("Qwen/Qwen3.5-35B-A3B", "qwen3.5:35b-a3b")}
# Host RAM and disk (GB) a tier's build needs, checked on the box before its base is downloaded. A 35B merges on the
# host when the card cannot hold its 70 GB of bf16 (an RTX 5090): about 80 GB of RAM, and base, merged and bf16 GGUF
# side by side at the peak.
HOST_NEEDS = {"2b": (16, 60), "4b": (24, 80), "9b": (40, 120), "pro": (80, 250)}
SHIPPED = {"2b": "601cd23c66ef", "4b": "1de6472396f2", "9b": "357e4bd45ff0"}
SHIPPED_TAG = "gruxai.com/grux/grux-local:{}"
SHIPPED_SETS = ("holdout", "holdout_v2", "s1")   # their v3 is scored on its own (eval_run --set holdout_v3)
FINAL = ("scored", "killed", "stopped")   # a tier in one of these is not built again by a rerun
CPU_HOUR, GB_HOUR = 0.0504, 0.0162   # what a GPU box bills beside the GPU (eval_run.BOX_EXTRA at 16 vCPU, 192 GB)
GGUF_SETUP = (
    "apt-get install -y -qq build-essential >/dev/null 2>&1; "
    # Never llama.cpp's requirements file: it pins torch from the CPU index, and a replaced torch stops transformers
    # importing (train_box BOOT, measured 2026-10-22). The converter reads gguf-py from its own tree.
    "pip install -q --break-system-packages cmake pyyaml tqdm requests >/dev/null 2>&1 && "
    f"curl -fsSL -o /tmp/llama.tar.gz https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.tar.gz && "
    "rm -rf /workspace/llama.cpp && mkdir -p /workspace/llama.cpp && "
    "tar xzf /tmp/llama.tar.gz -C /workspace/llama.cpp --strip-components=1 && rm -f /tmp/llama.tar.gz && "
    "cd /workspace/llama.cpp && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF "
    "-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF >/workspace/llama-build.log 2>&1 && "
    "cmake --build build --target llama-quantize -j $(nproc) >>/workspace/llama-build.log 2>&1 && "
    "test -x build/bin/llama-quantize && python convert_hf_to_gguf.py --help >/dev/null && "
    "python -c 'import torch; assert torch.__version__.startswith(\"2.11.\"), torch.__version__; "
    "print(\"llama.cpp ready, torch\", torch.__version__)'")
LOCK = threading.Lock()


# --------------------------------------------------------------------------- pure parts (tested without a box)

def p95(xs):
    """Nearest-rank 95th percentile: the smallest value at or above 95% of the list."""
    s = sorted(xs)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)] if s else None


def rows(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def chat_p95(paths):
    """p95 of per-request seconds over S1's chat families (chat_tools, chat_plain, chat_honesty) in these row files.
    A skipped row counts at the seconds it took: leaving it out would hide the slowest requests."""
    xs = [r["s"] for p in paths for r in rows(p) if str(r.get("family", "")).startswith("chat_") and "s" in r]
    return {"p95_s": p95(xs), "n": len(xs)}


def rate(gpu, cpu=None, memory=None):
    """Dollars an hour: the GPU's rate plus the box's vCPU and RAM at what they bill (H200, 16 vCPU, 192 GB: $6.5268,
    the rate Daytona charged in the 2026-11-15 smoke run). Without the box's own numbers, eval_run.BOX_EXTRA."""
    extra = cpu * CPU_HOUR + memory * GB_HOUR if cpu and memory else BOX_EXTRA
    return PRICE.get(gpu, max(PRICE.values())) + extra


def gpu_name(info):
    g = str(info.get("gpu_type")).split(".")[-1].replace("_", "-").strip("'>")
    return next((k for k in PRICE if k.replace("-", "") == g.replace("-", "").upper()), g)


def gguf_model(mlx_tag):
    """The Ollama name of a tier's GGUF build, and so the label of its rows: eval-gguf-<tag>-... never matches an MLX
    row (eval-grux-...), a base row (eval-base-...) or sweep_report's eval-grux-* glob."""
    return "gguf-" + mlx_tag


def library_modelfile(config, params, template, gguf_path):
    """The Modelfile of Ollama's library GGUF of the same size, pointed at our file. Refuses (ValueError) a library
    config without a renderer or a parser: Grux's tool calls are read by the qwen3.5 parser, and without one every
    call would arrive as plain text."""
    if config.get("model_format") != "gguf":
        raise ValueError(f"library model is {config.get('model_format')}, not gguf")
    for k in ("renderer", "parser"):
        if not config.get(k):
            raise ValueError(f"library config has no {k}")
    lines = [f"FROM {gguf_path}"]
    if template:
        lines.append(f'TEMPLATE """{template}"""')
    lines += [f"RENDERER {config['renderer']}", f"PARSER {config['parser']}"]
    for k, v in params.items():
        for x in (v if isinstance(v, list) else [v]):
            lines.append(f"PARAMETER {k} {json.dumps(x) if isinstance(x, str) else x}")
    return "\n".join(lines) + "\n"


def kill_verdict(gguf_rows, mlx_paths, max_points):
    """(kill, why, gguf %, mlx %). The MLX side is the same jobs in every MLX pass file there is (their mean); a job
    with no MLX row refuses the comparison instead of guessing (kill, so the tier never scores unchecked)."""
    ids = [r["case"] for r in gguf_rows]
    if not ids:
        return True, "the limited pass wrote no rows", None, None
    by_pass = [{r["case"]: bool(r.get("pass")) for r in rows(p)} for p in mlx_paths]
    by_pass = [m for m in by_pass if m]
    missing = [i for i in ids if not by_pass or any(i not in m for m in by_pass)]
    if missing:
        return True, f"no MLX row to compare for {len(missing)} jobs (first {missing[0]})", None, None
    g = 100.0 * sum(bool(r.get("pass")) for r in gguf_rows) / len(ids)
    m = 100.0 * sum(mp[i] for mp in by_pass for i in ids) / (len(ids) * len(by_pass))
    if m - g > max_points:
        return True, f"GGUF {g:.1f}% against MLX {m:.1f}% on the same {len(ids)} jobs, more than {max_points} points under", g, m
    return False, f"GGUF {g:.1f}% against MLX {m:.1f}% on the same {len(ids)} jobs", g, m


def limited_v3(per_family):
    """The first `per_family` held-out v3 jobs of each family, in bank order (deterministic)."""
    seen = {}
    return [c for c in holdout_v3.cases() if seen.setdefault(c["family"], []).append(1) or len(seen[c["family"]]) <= per_family]


def parse_host(out):
    """nproc, MemTotal bytes, cgroup memory.max, free and total bytes of /workspace (one per line) -> GB."""
    v = out.split()
    ram = int(v[1])
    if v[2].isdigit():
        ram = min(ram, int(v[2]))
    return {"nproc": int(v[0]), "ram_gb": round(ram / 1e9, 1), "disk_free_gb": round(int(v[3]) / 1e9, 1),
            "disk_gb": round(int(v[4]) / 1e9, 1)}


HOST_CMD = ("nproc; awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo; cat /sys/fs/cgroup/memory.max 2>/dev/null || "
            "echo max; df -B1 --output=avail /workspace | tail -1; df -B1 --output=size /workspace | tail -1")


def host_short(host, tier):
    need_ram, need_disk = HOST_NEEDS[tier]
    bad = []
    if host["ram_gb"] < need_ram:
        bad.append(f"{host['ram_gb']} GB RAM, needs {need_ram}")
    if host["disk_free_gb"] < need_disk:
        bad.append(f"{host['disk_free_gb']} GB free disk, needs {need_disk}")
    return "; ".join(bad) or None


def status(tier_dir):
    p = os.path.join(tier_dir, "gguf-status.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def set_status(tier_dir, stage, why, **extra):
    json.dump({"stage": stage, "why": why, **extra}, open(os.path.join(tier_dir, "gguf-status.json"), "w"), indent=1)


def update_manifest(tier_dir, **kv):
    p = os.path.join(tier_dir, "gguf-manifest.json")
    m = json.load(open(p)) if os.path.exists(p) else {}
    m.update(kv)
    json.dump(m, open(p, "w"), indent=1)
    return m


# --------------------------------------------------------------------------- preflight ($0, before any box)

def default_fetch(url, accept="application/vnd.docker.distribution.manifest.v2+json"):
    r = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "grux-local-eval/1"})
    with urllib.request.urlopen(r, timeout=30) as resp:
        return resp.read()


def blob_url(model, digest):
    return manifest_url(model).rsplit("/manifests/", 1)[0] + "/blobs/" + digest


def library_parts(lib_tag, fetch):
    """(config, params, template) of a library tag, read from the registry: the manifest, then its small blobs."""
    man = json.loads(fetch(manifest_url(lib_tag)))
    config = json.loads(fetch(blob_url(lib_tag, man["config"]["digest"])))
    layer = {l["mediaType"].rsplit(".", 1)[-1]: l["digest"] for l in man["layers"]}
    params = json.loads(fetch(blob_url(lib_tag, layer["params"]))) if "params" in layer else {}
    template = fetch(blob_url(lib_tag, layer["template"])).decode() if "template" in layer else ""
    return config, params, template


def registry_digest(tag, fetch):
    """The digest Ollama lists for a tag: sha256 of its manifest bytes (checked 2026-11-17 against the three
    shipped tags and the smoke run's qwen3.5:35b-a3b)."""
    return hashlib.sha256(fetch(manifest_url(tag))).hexdigest()


def preflight(a, fetch=default_fetch):
    """(problems, facts). Every check here is free; a box is created only when problems is empty."""
    bad, facts = [], {"modelfile": {}, "library": {}}
    for tier in a.tiers:
        d = os.path.join(a.train_root, tier)
        if not os.path.exists(os.path.join(d, "adapter.tar")):
            bad.append(f"{tier}: no adapter.tar in {d}")
            continue
        try:
            man = json.load(open(os.path.join(d, "manifest.json")))
        except (OSError, ValueError) as e:
            bad.append(f"{tier}: manifest.json unreadable: {e}")
            continue
        if man.get("model") != TIERS[tier][0]:
            bad.append(f"{tier}: trained from {man.get('model')}, expected {TIERS[tier][0]}")
        mlx = [train_run.eval_path(d, man.get("tag", ""), "holdout_v3", k) for k in range(1, 4)]
        if not any(os.path.exists(p) for p in mlx):
            bad.append(f"{tier}: no MLX held-out v3 rows ({os.path.basename(mlx[0])}) for the kill rule")
        lib = TIERS[tier][1]
        try:
            config, params, template = library_parts(lib, fetch)
            facts["modelfile"][tier] = library_modelfile(config, params, template, "{gguf}")
            facts["library"][tier] = {"tag": lib, "file_type": config.get("file_type"), "family": config.get("model_family")}
            if config.get("file_type") != a.quant[tier]:
                facts.setdefault("notes", []).append(
                    f"{tier}: library {lib} is {config.get('file_type')}; this run quantizes to {a.quant[tier]}")
        except Exception as e:
            bad.append(f"{tier}: library Modelfile for {lib}: {type(e).__name__}: {str(e)[:160]}")
    bad += [f"sets: {x}" for x in train_run.preflight_sets()]
    if not a.meter or not os.path.exists(a.meter):
        bad.append(f"no spend meter CLI at {a.meter!r} (--meter)")
    charter = os.path.expanduser(os.environ.get("GRUX_SWING_CHARTER", ""))
    if not charter or not os.path.exists(charter):
        bad.append("GRUX_SWING_CHARTER is not set to a charter file: the meter refuses every GPU box without one")
    if not os.environ.get("DAYTONA_API_KEY"):
        bad.append("DAYTONA_API_KEY is not set")
    if a.shipped:
        for t, want in SHIPPED.items():
            tag = SHIPPED_TAG.format(t)
            try:
                got = registry_digest(tag, fetch)
                if not got.startswith(want):
                    bad.append(f"{tag}: registry digest {got[:12]} is not the shipped {want}")
            except Exception as e:
                bad.append(f"{tag}: digest unverified: {type(e).__name__}: {str(e)[:120]}")
    try:
        ref = json.loads(fetch(f"https://api.github.com/repos/ggml-org/llama.cpp/git/ref/tags/{LLAMA_CPP_TAG}",
                               accept="application/vnd.github+json"))
        if ref.get("object", {}).get("sha") != LLAMA_CPP_COMMIT:
            bad.append(f"llama.cpp {LLAMA_CPP_TAG} now points at {ref.get('object', {}).get('sha')}, pinned "
                       f"{LLAMA_CPP_COMMIT}")
    except Exception as e:
        bad.append(f"llama.cpp pin unverified: {type(e).__name__}: {str(e)[:120]}")
    for g in sorted(set(a.gpu_of.values())):
        if g not in PRICE:
            bad.append(f"GPU {g}: no listed rate in eval_run.PRICE")
        try:
            from daytona import GpuType
            GpuType(g)
        except ValueError:
            bad.append(f"GPU {g}: not a Daytona GpuType")
        except Exception as e:   # no SDK here (a test stub): the box create is the check
            facts.setdefault("notes", []).append(f"GPU {g}: GpuType unverified here ({type(e).__name__})")
    return bad, facts


# --------------------------------------------------------------------------- on a box

def tool_probe(client, model):
    """(ok, detail): one Grux turn (the first held-out v3 job whose first step is a call) must come back as a tool
    call. A renderer or parser mismatch shows here as plain text where a call should be."""
    case = next(c for c in holdout_v3.cases() if "call" in c["steps"][0])
    req = holdout_v3.request_v3(case, model, 0)
    try:
        raw = client.post(req["path"], req["body"], timeout=1800)
    except Exception as e:
        try:   # a box that is gone is not a verdict on the model: the next box probes again
            client.get("/api/tags", timeout=30)
        except Exception as e2:
            raise train_run.BoxGone(f"Ollama stopped answering during the tool-call probe: {type(e2).__name__}") from None
        return False, f"{case['id']}: {type(e).__name__}: {str(e)[:160]}"
    m = raw.get("message") or {}
    names = [(x.get("function") or {}).get("name") for x in m.get("tool_calls") or []]
    if not names:
        return False, f"{case['id']}: no tool call; reply began {str(m.get('content') or '')[:120]!r}"
    return True, f"{case['id']}: called {names}"


def evaluate(client, model, tier_dir, mlx_tag, a):
    """The kill rule, then (if it passes) the full scoring. Returns the stage written to gguf-status.json. BoxGone
    propagates (a new box reruns the tier); SetDamaged in the limited pass stops the tier (the model, not the box)."""
    ok, detail = tool_probe(client, model)
    log(f"{model}: tool-call probe {'ok' if ok else 'FAILED'}: {detail}", tier_dir)
    update_manifest(tier_dir, tool_probe={"ok": ok, "detail": detail})
    if not ok:
        set_status(tier_dir, "killed", "tool-call probe: " + detail)
        return "killed"
    lim = limited_v3(a.kill_jobs)
    label = model + "-limit"
    try:
        train_run.score_model(client, model, tier_dir, a.parallel, 1,
                              sets={"holdout_v3": (lambda: lim, None, None)}, label=label)
    except train_run.SetDamaged as e:
        set_status(tier_dir, "stopped", f"limited pass: {e}")
        return "stopped"
    kill, why, g, m = kill_verdict(rows(train_run.eval_path(tier_dir, label, "holdout_v3", 1)),
                                   [train_run.eval_path(tier_dir, mlx_tag, "holdout_v3", k) for k in range(1, 4)],
                                   a.kill_points)
    log(f"{model}: kill rule {'KILL' if kill else 'pass'}: {why}", tier_dir)
    update_manifest(tier_dir, kill_rule={"kill": kill, "why": why, "gguf_pct": g, "mlx_pct": m,
                                         "jobs": len(lim), "max_points": a.kill_points})
    if kill:
        set_status(tier_dir, "killed", why)
        return "killed"
    try:
        train_run.score_model(client, model, tier_dir, a.parallel, a.passes)
    except train_run.SetDamaged as e:
        set_status(tier_dir, "stopped", f"full scoring: {e}")
        return "stopped"
    summary = {}
    for name in train_run.SETS:
        for k in range(1, a.passes + 1):
            r = rows(train_run.eval_path(tier_dir, model, name, k))
            mr = rows(train_run.eval_path(tier_dir, mlx_tag, name, k))
            summary[f"{name}-pass{k}"] = {"gguf": [sum(bool(x.get("pass")) for x in r), len(r)],
                                          "mlx": [sum(bool(x.get("pass")) for x in mr), len(mr)]}
    s1 = [train_run.eval_path(tier_dir, model, "s1", k) for k in range(1, a.passes + 1)]
    s1_mlx = [train_run.eval_path(tier_dir, mlx_tag, "s1", k) for k in range(1, a.passes + 1)]
    p = {"gguf": chat_p95(s1), "mlx": chat_p95(s1_mlx), "parallel": a.parallel}
    log(f"{model}: chat p95 {p['gguf']['p95_s']} s over {p['gguf']['n']} requests at parallel {a.parallel} "
        f"(MLX build {p['mlx']['p95_s']} s)", tier_dir)
    update_manifest(tier_dir, scores=summary, chat_p95=p)
    set_status(tier_dir, "scored", "every set scored", passes=a.passes)
    return "scored"


def serial_chat(client, model, tier_dir):
    """S1's chat requests one at a time (no queueing behind other requests), after one untimed warm request: the
    single-user latency a charter's p95 means."""
    chat = [c for c in suite.cases() if c["family"].startswith("chat_")]
    req = suite.request(chat[0], model)
    client.post(req["path"], req["body"], timeout=1800)
    train_run.score_model(client, model, tier_dir, 1, 1, sets={"s1": (lambda: chat, suite.request, suite.score)},
                          label=model + "-serial")
    p = chat_p95([train_run.eval_path(tier_dir, model + "-serial", "s1", 1)])
    log(f"{model}: serial chat p95 {p['p95_s']} s over {p['n']} requests", tier_dir)
    update_manifest(tier_dir, chat_p95_serial=dict(p, parallel=1))


def on_gpu_whole(client, model):
    """(whole, ps entry): True when the loaded model sits in GPU memory entirely (size_vram == size), None when it
    is not loaded right now."""
    ps = next((x for x in client.get("/api/ps").get("models", []) if model in (x.get("name"), x.get("model"))), {})
    if not ps:
        return None, {}
    return ps.get("size_vram", 0) >= ps.get("size", 1), {k: ps.get(k) for k in ("size", "size_vram")}


def fetch_gguf(box, tier, tier_dir, remote, sha, a):
    """Bring the GGUF here only when the disk allows (size plus train_run.DISK_FLOOR); never the Pro's by default."""
    if tier == "pro" and not a.pull_pro:
        return "not pulled: the Pro GGUF stays on the box unless --pull-pro"
    dest = os.path.join(tier_dir, os.path.basename(remote))
    try:
        code, size = box.sh(f"stat -c %s {remote}", timeout=60)
        short = train_run.disk_short(tier_dir, int(size)) if code == 0 and size.strip().isdigit() else "size unreadable"
        if short:
            return "skipped: " + short
        box.get(remote, dest + ".part")
        h = hashlib.sha256()
        with open(dest + ".part", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            os.remove(dest + ".part")
            return f"failed: sha256 {h.hexdigest()[:12]} here, {sha[:12]} on the box"
        os.replace(dest + ".part", dest)
        return "pulled " + dest
    except Exception as e:
        if os.path.exists(dest + ".part"):
            os.remove(dest + ".part")
        return f"failed: {type(e).__name__}: {str(e)[:160]}"


def step(box, name, cmd, tier_dir, max_s, marker=None):
    code, out = box.run_long(name, cmd, max_s=max_s, poll_s=15)
    tail = out.strip().splitlines()[-3:] if out.strip() else []
    log(f"{name}: exit {code} {tail}", tier_dir)
    if code != 0 or (marker and marker not in out):
        raise RuntimeError(f"{name} failed (exit {code}): {' | '.join(tail)[-300:]}")
    return out


def build_tier(box, tier, tier_dir, mlx_tag, modelfile, quant):
    """Merge, convert, quantize, import. Returns the manifest facts; raises on any failed step."""
    model, hf_id = gguf_model(mlx_tag), TIERS[tier][0]
    box.sh("rm -rf /workspace/base /workspace/out /workspace/gguf && mkdir -p /workspace/out /workspace/gguf", timeout=300)
    box.put(os.path.join(tier_dir, "adapter.tar"), "/workspace/adapter.tar")
    box.sh("cd /workspace && tar xf adapter.tar -C out && rm -f adapter.tar && ls out/adapter", timeout=300)
    step(box, f"base-{tier}", "python -c \"from huggingface_hub import snapshot_download as s; "
         f"s('{hf_id}', local_dir='/workspace/base')\"", tier_dir, 3600)
    out = step(box, f"merge-{tier}", "python /workspace/gguf_box.py --base /workspace/base --adapter "
               "/workspace/out/adapter --out /workspace/out", tier_dir, 5400, marker="TRAIN DONE")
    merged_on = next((l for l in out.splitlines() if l.startswith("MERGE on")), "")
    box.sh("rm -rf /workspace/base", timeout=600)
    bf16 = f"/workspace/gguf/{model}-bf16.gguf"
    step(box, f"convert-{tier}", f"python /workspace/llama.cpp/convert_hf_to_gguf.py /workspace/out/merged "
         f"--outtype bf16 --no-mtp --outfile {bf16}", tier_dir, 5400)
    box.sh("rm -rf /workspace/out", timeout=600)
    gguf = f"/workspace/gguf/{model}-{quant}.gguf"
    step(box, f"quantize-{tier}", f"/workspace/llama.cpp/build/bin/llama-quantize {bf16} {gguf} {quant} $(nproc) "
         f"&& rm -f {bf16}", tier_dir, 5400)
    code, sha = box.sh(f"sha256sum {gguf} | cut -d' ' -f1", timeout=1800)
    code, size = box.sh(f"stat -c %s {gguf}", timeout=60)
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as f:
        f.write(modelfile.replace("{gguf}", gguf))
    box.put(f.name, "/workspace/gguf/Modelfile")
    os.remove(f.name)
    step(box, f"create-{tier}", f"ollama create {model} -f /workspace/gguf/Modelfile", tier_dir, 3600)
    return {"gguf": gguf, "gguf_sha256": sha.strip(), "gguf_bytes": int(size.strip() or 0), "quant": quant,
            "merged_on": merged_on, "modelfile": modelfile}


def check_import(client, model, modelfile, quant):
    """None when the imported model carries the library's renderer, parser and parameters and is a GGUF of the
    quant asked for; else what differs (the tier stops before any request is scored)."""
    show = client.post("/api/show", {"model": model}, timeout=120)
    text, det = show.get("modelfile") or "", show.get("details") or {}
    bad = [l for l in modelfile.splitlines() if l.startswith(("RENDERER", "PARSER")) and l not in text]
    params = show.get("parameters") or ""
    for l in modelfile.splitlines():
        if l.startswith("PARAMETER "):
            k, v = l.split(" ", 2)[1:]
            if not any(p.split(None, 1) == [k, v] for p in params.splitlines()):
                bad.append(l)
    if det.get("format") != "gguf":
        bad.append(f"format {det.get('format')}")
    if str(det.get("quantization_level", "")).upper() != quant.upper():
        bad.append(f"quantization {det.get('quantization_level')}")
    return ("does not match the library: " + "; ".join(bad)) if bad else None


def run_tier(box, client, tier, a, facts, host_info, root):
    tier_dir = os.path.join(a.train_root, tier)
    st = status(tier_dir)
    if st.get("stage") in FINAL:
        log(f"{tier}: already {st['stage']} ({st.get('why', '')[:120]}); skipped", root)
        return st["stage"]
    mlx_tag = json.load(open(os.path.join(tier_dir, "manifest.json")))["tag"]
    model, quant = gguf_model(mlx_tag), a.quant[tier]
    code, out = box.sh(HOST_CMD, timeout=60)
    host = parse_host(out)
    short = host_short(host, tier)
    update_manifest(tier_dir, box=host_info, host_measured=host, llama_cpp={"tag": LLAMA_CPP_TAG, "commit": LLAMA_CPP_COMMIT},
                    ollama_version=OLLAMA_VERSION, library=facts["library"].get(tier), think=suite.THINK)
    if short:
        log(f"{tier}: REFUSED before downloading anything: this box has {short}", root)
        set_status(tier_dir, "refused", f"host: {short}", host=host)
        return "refused"
    built = build_tier(box, tier, tier_dir, mlx_tag, facts["modelfile"][tier], quant)
    update_manifest(tier_dir, **built)
    why = check_import(client, model, facts["modelfile"][tier], quant)
    info, why_not = probe(client, model)
    whole, ps = on_gpu_whole(client, model)
    update_manifest(tier_dir, import_check=why or "ok", probe=info, gpu_whole=whole, ps=ps)
    log(f"{tier}: {model} imported: {why or 'matches the library'}; probe {info} -> {why_not or 'ok'}; "
        f"on GPU whole: {whole} {ps}", tier_dir)
    if why or why_not:
        set_status(tier_dir, "stopped", why or why_not)
        return "stopped"
    stage = evaluate(client, model, tier_dir, mlx_tag, a)
    if stage == "scored" and tier in a.serial_chat:
        serial_chat(client, model, tier_dir)
    whole, ps = on_gpu_whole(client, model)   # after the 65,536-token sets: still whole, or spilled to the host
    update_manifest(tier_dir, gpu_whole_after=whole, ps_after=ps)
    if whole is False:
        log(f"{tier}: WARNING the model was not wholly in GPU memory ({ps}): its seconds are not a GPU number", tier_dir)
    pulled = fetch_gguf(box, tier, tier_dir, built["gguf"], built["gguf_sha256"], a)
    open(os.path.join(tier_dir, f"Modelfile.{model}"), "w").write(
        facts["modelfile"][tier].replace("{gguf}", "./" + os.path.basename(built["gguf"])))
    update_manifest(tier_dir, gguf_download=pulled)
    log(f"{tier}: GGUF {pulled}", tier_dir)
    box.sh(f"ollama rm {model}; rm -rf /workspace/gguf", timeout=600)
    return stage


def run_shipped(client, a, root):
    """The shipped tags on v1, v2 and S1, digest checked on the box too (refused on a mismatch, nothing scored)."""
    d = a.shipped_run
    sets = {k: train_run.SETS[k] for k in SHIPPED_SETS}
    for t, want in SHIPPED.items():
        tag, label = SHIPPED_TAG.format(t), f"shipped-{t}"
        if status(d).get(label, {}).get("stage") in FINAL:
            continue
        try:
            pulled = client.post("/api/pull", {"model": tag, "stream": False}, timeout=3600).get("status")
        except Exception as e:
            pulled = f"error {type(e).__name__}: {str(e)[:160]}"
        info, why_not = probe(client, tag, expect_digest=want)
        log(f"{tag}: pull {pulled}; probe {info} -> {why_not or 'ok'}", d)
        if why_not:
            mark_shipped(d, label, "refused", why_not, probe=info)
            continue
        train_run.score_model(client, tag, d, a.parallel, a.passes, sets=sets, label=label)
        p = chat_p95([train_run.eval_path(d, label, "s1", k) for k in range(1, a.passes + 1)])
        log(f"{tag}: chat p95 {p['p95_s']} s over {p['n']} requests at parallel {a.parallel}", d)
        mark_shipped(d, label, "scored", "v1, v2 and S1 scored", probe=info, chat_p95=dict(p, parallel=a.parallel))
        try:
            client._req("DELETE", "/api/delete", {"model": tag}, timeout=120)
        except Exception:
            pass


def mark_shipped(d, label, stage, why, **extra):
    p = os.path.join(d, "gguf-status.json")
    with LOCK:
        all_ = json.load(open(p)) if os.path.exists(p) else {}
        all_[label] = {"stage": stage, "why": why, **extra}
        json.dump(all_, open(p, "w"), indent=1)


def one_box(a, gpu, tiers, shipped, facts, rid, root):
    """One leased box for a group of tiers (and the shipped tags when `shipped`). Returns (status, dollars)."""
    with TrainBox([gpu], "gguf-" + os.path.basename(root), max_minutes=a.max_minutes,
                  log=lambda m: log(m, root), owner=a.owner, reservation=rid) as box:
        g = gpu_name(box.info)
        price = rate(g, box.info.get("cpu"), box.info.get("memory"))
        t0 = time.time()
        cost = lambda: (time.time() - t0) / 3600 * price  # noqa: E731
        log(f"box {box.info['id'][:8]} {g}, {box.info.get('cpu')} vCPU, {box.info.get('memory')} GB RAM, "
            f"{box.info.get('disk')} GB disk at ${price:.4f}/h", root)
        try:
            box.sh("mkdir -p /workspace", timeout=60)
            for name, cmd in (("boot", BOOT), ("llama", GGUF_SETUP)):
                code, out = box.run_long(name, cmd, max_s=2400, poll_s=15)
                log(f"{name} exit {code}: {out.strip().splitlines()[-2:] if out.strip() else ''}", root)
                if code != 0:
                    return f"{name} failed", cost()
            box.put(os.path.join(HERE, "gguf_box.py"), "/workspace/gguf_box.py")
            box.put(os.path.join(HERE, "box_train.py"), "/workspace/box_train.py")
            code, host_out = box.sh(HOST_CMD, timeout=60)
            host_info = dict(box.info, host=parse_host(host_out), gpu=g, rate_per_h=round(price, 4))
            host_info.pop("started_at", None)
            log(f"box measured: {host_info['host']}; {box.info.get('nvidia_smi')}", root)
            client = box.ollama()
            for tier in tiers:
                if a.spent_of(gpu) + cost() >= a.cap_of(gpu):
                    log(f"STOP: {gpu} cap ${a.cap_of(gpu)} reached before {tier}", root)
                    return "cap", cost()
                log(f"{tier}: {run_tier(box, client, tier, a, facts, host_info, root)}", root)
            if shipped:
                run_shipped(client, a, root)
            return "done", cost()
        except Exception as e:
            why = f"{type(e).__name__}: {str(e)[:300]}"
            log(f"box lost or step failed: {why}; gone={box.gone()}", root)
            return ("evicted" if box.gone() else "failed: " + why), cost()


def run_group(a, gpu, tiers, shipped, facts, root):
    status_ = "not started"
    for attempt in range(4):
        if a.spent_of(gpu) >= a.cap_of(gpu):
            log(f"STOP: {gpu} cap ${a.cap_of(gpu)} reached (${a.spent_of(gpu):.2f})", root)
            break
        hours = max(0.25, (a.cap_of(gpu) - a.spent_of(gpu)) / rate(gpu))
        cmd = [sys.executable, a.meter, "reserve", "--lane", a.meter_lane, "--hours", f"{hours:.3f}",
               "--gpu", gpu.replace("-", ""), "--purpose", "measure"]
        charter = os.path.expanduser(os.environ.get("GRUX_SWING_CHARTER", ""))
        r = __import__("subprocess").run(cmd + ["--charter", charter], capture_output=True, text=True)
        rid = r.stdout.strip()
        if r.returncode != 0 or not rid:
            log(f"STOP: the spend meter refused the {gpu} box (exit {r.returncode}): {r.stderr.strip()[:200]}", root)
            return "meter refused"
        t_box = time.time()
        try:
            status_, cost = one_box(a, gpu, tiers, shipped, facts, rid, root)
        except Exception as e:   # a box reclaimed while it starts raises from create
            log(f"{gpu} box {attempt + 1} never came up: {type(e).__name__}: {str(e)[:160]}", root)
            status_, cost = "evicted", 0.0
            time.sleep(30)
        __import__("subprocess").run([sys.executable, a.meter, "settle", rid, "--hours",
                                      f"{(time.time() - t_box) / 3600:.4f}"], capture_output=True, text=True)
        a.add_spent(gpu, cost)
        log(f"{gpu} box {attempt + 1}: {status_}, spent about ${a.spent_of(gpu):.2f}", root)
        if status_ != "evicted":
            break
    return status_


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="this run's log, manifest, spend and status")
    ap.add_argument("--train-root", required=True, help="the trained tiers' run dirs: <root>/<tier>/adapter.tar")
    ap.add_argument("--tiers", default="2b,4b,9b,pro")
    ap.add_argument("--shipped-run", default="", help="row dir for the shipped tags (default <run>/shipped)")
    ap.add_argument("--no-shipped", dest="shipped", action="store_false")
    ap.add_argument("--gpu", default="H200", help="the box for every tier but the Pro, and the shipped tags")
    ap.add_argument("--pro-gpu", default="RTX-5090", help="the Pro's box (the charter's chat p95 is on an RTX 5090)")
    ap.add_argument("--quant", default="Q4_K_M", help="one type for every tier, or per tier: Q4_K_M,2b=Q8_0")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=4, help="as train_run scored the MLX builds (BOOT serves 4)")
    ap.add_argument("--kill-jobs", type=int, default=8, help="held-out v3 jobs per family in the limited pass")
    ap.add_argument("--kill-points", type=float, default=15.0,
                    help="stop a tier whose limited pass is more than this many points under its MLX build")
    ap.add_argument("--serial-chat", default="pro", help="tiers that also get a one-at-a-time chat latency pass")
    ap.add_argument("--pull-pro", action="store_true", help="also download the Pro GGUF (about 21 GB)")
    ap.add_argument("--cap", type=float, default=35.0, help="dollars for the --gpu box group")
    ap.add_argument("--pro-cap", type=float, default=25.0, help="dollars for the --pro-gpu box group")
    ap.add_argument("--max-minutes", type=int, default=480)
    ap.add_argument("--meter", default="", help="spend meter CLI (reserve/settle) called around every box")
    ap.add_argument("--meter-lane", default="grux-local")
    ap.add_argument("--owner", default="grux")
    ap.add_argument("--preflight-only", action="store_true", help="run the $0 checks and stop")
    a = ap.parse_args(argv)
    a.run, a.train_root = os.path.abspath(a.run), os.path.abspath(a.train_root)
    a.shipped_run = os.path.abspath(a.shipped_run or os.path.join(a.run, "shipped"))
    a.tiers = [t for t in a.tiers.split(",") if t]
    unknown = [t for t in a.tiers if t not in TIERS]
    if unknown:
        ap.error(f"unknown tiers {unknown}; known {list(TIERS)}")
    default, per = "Q4_K_M", {}
    for part in a.quant.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            per[k] = v
        else:
            default = part
    a.quant = {t: per.get(t, default) for t in TIERS}
    a.serial_chat = [t for t in a.serial_chat.split(",") if t]
    a.gpu_of = {t: (a.pro_gpu if t == "pro" else a.gpu) for t in a.tiers}
    if a.shipped:
        a.gpu_of.setdefault("shipped", a.gpu)
    return a


def attach_spend(a):
    spent_path = os.path.join(a.run, "spent.json")
    spent = json.load(open(spent_path)) if os.path.exists(spent_path) else {}

    def add(gpu, x):
        with LOCK:
            spent[gpu] = round(spent.get(gpu, 0.0) + x, 4)
            json.dump(spent, open(spent_path, "w"))
    a.spent_of = lambda g: spent.get(g, 0.0)
    a.add_spent = add
    a.cap_of = lambda g: a.pro_cap if g == a.pro_gpu and g != a.gpu else a.cap


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.run, exist_ok=True)
    os.makedirs(a.shipped_run, exist_ok=True)
    bad, facts = preflight(a)
    json.dump({**{k: v for k, v in vars(a).items() if not callable(v)}, "llama_cpp_tag": LLAMA_CPP_TAG,
               "llama_cpp_commit": LLAMA_CPP_COMMIT, "ollama_version": OLLAMA_VERSION, "think": suite.THINK,
               "preflight": bad, "facts": facts}, open(os.path.join(a.run, "manifest.json"), "w"), indent=1)
    for n in facts.get("notes", []):
        log("NOTE " + n, a.run)
    if bad:
        log("REFUSED preflight: " + " | ".join(bad), a.run)
        json.dump({"status": "preflight refused: " + " | ".join(bad)}, open(os.path.join(a.run, "status.json"), "w"))
        sys.exit(4)
    log(f"preflight ok: {len(a.tiers)} tiers, shipped {a.shipped}, llama.cpp {LLAMA_CPP_TAG}", a.run)
    if a.preflight_only:
        return
    attach_spend(a)
    groups = {}
    for t in a.tiers:
        groups.setdefault(a.gpu_of[t], []).append(t)
    if a.shipped:
        groups.setdefault(a.gpu, [])
    with cf.ThreadPoolExecutor(len(groups)) as pool:   # one box per GPU type, side by side
        futs = {g: pool.submit(run_group, a, g, ts, a.shipped and g == a.gpu, facts, a.run) for g, ts in groups.items()}
        result = {g: f.result() for g, f in futs.items()}
    tiers = {t: status(os.path.join(a.train_root, t)) for t in a.tiers}
    json.dump({"status": result, "tiers": tiers, "spent": {g: a.spent_of(g) for g in groups}},
              open(os.path.join(a.run, "status.json"), "w"), indent=1)
    for t in a.tiers:   # a GGUF only passes through .part on its way here
        for n in os.listdir(os.path.join(a.train_root, t)):
            if n.startswith("gguf-") and n.endswith(".part"):
                os.remove(os.path.join(a.train_root, t, n))
    log(f"FINISHED: {result}; spent {[(g, round(a.spent_of(g), 2)) for g in groups]}", a.run)


if __name__ == "__main__":
    main()
