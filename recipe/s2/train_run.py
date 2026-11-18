#!/usr/bin/env python3
"""S2 train and score run (see recipe/README.md; caps: training $40 per model, eval $10).

One leased spot box at a time: train a LoRA (every checkpoint pulled out the moment it lands), merge,
import the base and the trained model into the box's Ollama as nvfp4 (the format a Mac serves),
score both on the S1 suite and the frozen held-out sets v1, v2, v3 and v4 (--passes times) with thinking off, then pull the
trained model's Ollama pack out for the Mac check when the disk here allows (pull.json says what happened; the base
is public and never pulled). A spot box goes without warning (measured 2026-10-22), so a new box
resumes from the last checkpoint pulled out, skips training once the adapter is out, and skips every
case already scored. Nothing of ours but the training file goes in; nothing leaves but files.

  DAYTONA_API_KEY=<your key> .venv/bin/python s2/train_run.py \\
    --run s2/results/train-4b-<stamp> --model Qwen/Qwen3.5-4B --data s2/results/data-<stamp>/train.jsonl
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import holdout  # noqa: E402
import holdout_v2  # noqa: E402
import holdout_v3  # noqa: E402
import holdout_v4  # noqa: E402
import suite  # noqa: E402
from eval_run import BOX_EXTRA, EARLY_ERRORS, PRICE, log, skip_max  # noqa: E402
from train_box import BOOT, TrainBox  # noqa: E402

suite.THINK = False   # Grux sends think false on every Ollama call (loop row D-THINK, PROVEN)
MODELFILE = ("FROM {src}\nRENDERER qwen3.5\nPARSER qwen3.5\nPARAMETER temperature 1\nPARAMETER top_k 20\n"
             "PARAMETER top_p 0.95\nPARAMETER presence_penalty 1.5\nPARAMETER min_p 0\n")
SETS = {"holdout_v3": (holdout_v3.cases, None, None),   # first: the gate's set; a job runs step by step (run_case)
        "holdout": (holdout.cases, holdout.request_holdout, holdout.score_holdout),
        "holdout_v2": (holdout_v2.cases, holdout_v2.request_v2, holdout_v2.score_v2),
        "s1": (suite.cases, suite.request, suite.score),
        "holdout_v4": (holdout_v4.cases, None, None)}   # last: held-out v3's runner and scorer; a cap cuts it first


ALLOC = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"   # less cached-but-free memory for kernels outside PyTorch


class BoxGone(RuntimeError):
    """The box's Ollama stopped answering mid-set: nothing more is written, the next box resumes the set."""


class SetDamaged(RuntimeError):
    """The box answers but the model fails job after job (eval_run's skip_max, or its first EARLY_ERRORS): the set
    stops, its skips are not written, and a rerun scores those jobs again."""


def preflight_sets(names=None):
    """Build every scoring set's cases and the first case's request on this machine, before any box exists.
    Returns the problems (empty when every set can be scored). Measured 2026-11-17: v1 needs Grux 3.1.0's source
    (GRUX_SRC) and the launcher had none, so a 2B scoring box ran 17 minutes of setup before v1's first request
    failed; this check fails in seconds for $0. `names` checks only those sets (gguf_run scores no held-out v4)."""
    bad = []
    for name, (cases_fn, request, _score) in SETS.items():
        if names is not None and name not in names:
            continue
        try:
            cases = cases_fn()
            if not cases:
                bad.append(f"{name}: no cases")
                continue
            for case in {c["family"]: c for c in cases}.values():   # one per family: each reads its own files
                if request is None:
                    holdout_v3.request_v3(case, "preflight")
                else:
                    request(case, "preflight")
        except (Exception, SystemExit) as e:   # holdout_v3.wire_tools exits on a source tree that is not 3.3.1
            bad.append(f"{name}: {type(e).__name__}: {str(e)[:160]}")
    return bad


def eval_path(run_dir, model, name, k):
    """Pass 1 keeps the name every reader already uses; pass k > 1 adds -pass<k>. A registry tag's `/` and `:` become
    `_`, as eval_run names its files (a local tag has neither, so its name is unchanged)."""
    model = model.replace(":", "_").replace("/", "_")
    return os.path.join(run_dir, f"eval-{model}-{name}" + (f"-pass{k}" if k > 1 else "") + ".jsonl")


def scored(path):
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path):
        try:
            out.add(json.loads(line)["case"])
        except (ValueError, KeyError):
            pass
    return out


def score_model(client, model, run_dir, parallel, passes=1, sets=None, label=None):
    """Every case of every set for one model, `passes` times (the gate's reruns, one seed as eval_run keeps);
    appends as it goes, skips what is already scored. A case that errors twice is written as skipped (a fail).
    `sets` scores a subset (default SETS, every set); `label` names the row files instead of the model (s2/gguf_run.py
    keeps a limited pass and a registry tag's rows apart from every other file in the run)."""
    stop = threading.Event()

    def one(case, request, score):
        if stop.is_set():   # the set is ending: queued jobs return at once instead of billing their retries
            return None
        t0 = time.time()
        for attempt in (1, 2):
            try:
                if request is None:   # held-out v3: the job's steps in order, stopping at the first failed one
                    ok, why, got, steps = holdout_v3.run_case(
                        case, model, lambda p, b: client.post(p, b, timeout=900))
                else:
                    req = request(case, model)
                    raw = client.post(req["path"], req["body"], timeout=900)
                    ok, why, got = score(case, raw)
                    steps = None
                break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(5)
        else:   # a skipped job counts as a fail, as eval_run's do, so a pass file always covers every case ...
            try:   # ... unless the box itself is gone (a spot reclaim fails every call at once): write nothing,
                client.get("/api/tags", timeout=30)   # and the next box reruns the set from here
            except Exception as e2:
                raise BoxGone(f"Ollama stopped answering during {case['id']}: {type(e2).__name__}") from None
            return {"case": case["id"], "family": case["family"], "status": "skipped", "pass": False,
                    "why": "error twice: " + err, "got": None, "s": round(time.time() - t0, 2), "think": suite.THINK}
        row = {"case": case["id"], "family": case["family"], "status": "scored", "pass": ok, "why": why,
               "got": got, "s": round(time.time() - t0, 2), "think": suite.THINK}
        if steps is not None:
            row["steps"] = steps
        return row

    # set by set, its passes together: one model load per context size (v3 runs at 65,536, the rest at 16,384)
    sets = SETS if sets is None else sets
    for (name, (cases_fn, request, score)), k in ((kv, k) for kv in sets.items() for k in range(1, passes + 1)):
        cases = cases_fn()
        path = eval_path(run_dir, label or model, name, k)
        todo = [c for c in cases if c["id"] not in scored(path)]
        if not todo:
            continue
        log(f"{model} {name} pass {k}: {len(todo)} of {len(cases)} cases to score", run_dir)
        skips, seen = [], 0   # skipped rows are written only when the set ends whole (a damaged set reruns them)
        pool = cf.ThreadPoolExecutor(parallel)
        try:
            with open(path, "a") as f:
                for r in pool.map(lambda c: one(c, request, score), todo):
                    if r is None:
                        continue
                    seen += 1
                    if r["status"] == "skipped":
                        skips.append(r)
                        if len(skips) > skip_max(len(cases)) or (len(todo) > EARLY_ERRORS and len(skips) == seen == EARLY_ERRORS):
                            raise SetDamaged(f"{model} {name} pass {k}: {len(skips)} of {seen} jobs errored twice "
                                             f"with the box up; last: {r['why'][:120]}")
                        continue
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                for r in skips:
                    f.write(json.dumps(r) + "\n")
        except BaseException:
            stop.set()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        rows = []
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        skipped = sum(r.get("status") == "skipped" for r in rows)
        log(f"{model} {name} pass {k}: {sum(r['pass'] for r in rows)}/{len(rows)} pass"
            + (f", {skipped} SKIPPED (counted as fails)" if skipped else ""), run_dir)


def train(box, a, run_dir):
    """Train (fresh, or resumed from the last checkpoint out), pulling checkpoints and the adapter."""
    ckpt_tar, adapter_tar = os.path.join(run_dir, "ckpt.tar"), os.path.join(run_dir, "adapter.tar")
    if os.path.exists(adapter_tar):
        box.put(adapter_tar, "/workspace/adapter.tar")
        code, out = box.sh("cd /workspace && mkdir -p out && tar xf adapter.tar -C out && ls out/adapter", timeout=300)
        log(f"adapter already out: merge only ({out[-80:]})", run_dir)
        cmd = f"{ALLOC} python box_train.py --model /workspace/base --data train.jsonl --out out --merge-only out/adapter"
    else:
        resume = ""
        if os.path.exists(ckpt_tar):
            box.put(ckpt_tar, "/workspace/ckpt.tar")
            box.sh("cd /workspace && mkdir -p resume && tar xf ckpt.tar -C resume", timeout=300)
            resume = " --resume resume/ckpt"
            log(f"resuming from {json.load(open(ckpt_tar + '.json'))}", run_dir)
        cmd = (f"{ALLOC} python box_train.py --model /workspace/base --data train.jsonl --out out --epochs {a.epochs} "
               f"--rank {a.rank} --accum {a.accum} --lr {a.lr} --save-every {a.save_every} --max-len {a.max_len} "
               f"--micro {a.micro} --micro-tokens {a.micro_tokens}{resume}")
    code, smi = box.sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; "
                       "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader", timeout=60)
    log(f"GPU before training: {' | '.join(smi.strip().splitlines()) or 'no compute apps'}", run_dir)
    box.sb.process.create_session("train")
    from daytona import SessionExecuteRequest
    box.sb.process.execute_session_command("train", SessionExecuteRequest(
        command=f"cd /workspace && {cmd} > train.log 2>&1; echo EXIT=$? >> train.log", run_async=True))
    pulled = json.load(open(ckpt_tar + ".json"))["step"] if os.path.exists(ckpt_tar + ".json") else 0
    while True:
        time.sleep(60)
        code, tail = box.sh("tail -c 3000 /workspace/train.log", timeout=60)
        lines = tail.splitlines()
        steps = [l for l in lines if l.startswith("step ")]
        if steps:
            log(f"train: {steps[-1]}", run_dir)
        ck = [int(m.group(1)) for m in re.finditer(r"^CKPT (\d+)/", tail, re.M)]
        if ck and max(ck) > pulled:
            c2, _ = box.sh("cd /workspace/out && cat ckpt/step.json && tar cf /workspace/ckpt.tar ckpt", timeout=600)
            if c2 == 0:
                box.get("/workspace/ckpt.tar", ckpt_tar + ".part")
                os.replace(ckpt_tar + ".part", ckpt_tar)
                pulled = max(ck)
                json.dump({"step": pulled}, open(ckpt_tar + ".json", "w"))
                log(f"checkpoint {pulled} out ({os.path.getsize(ckpt_tar) / 1e6:.0f} MB)", run_dir)
        if any(l.startswith("ADAPTER DONE") for l in lines) and not os.path.exists(adapter_tar):
            box.sh("cd /workspace/out && tar cf /workspace/adapter.tar adapter", timeout=600)
            box.get("/workspace/adapter.tar", adapter_tar + ".part")
            os.replace(adapter_tar + ".part", adapter_tar)
            log(f"adapter out ({os.path.getsize(adapter_tar) / 1e6:.0f} MB)", run_dir)
            for p in (ckpt_tar, ckpt_tar + ".json"):   # the adapter supersedes the checkpoint (bounded space)
                if os.path.exists(p):
                    os.remove(p)
        done = [l for l in lines if l.startswith("EXIT=")]
        if done:
            ok = done[-1] == "EXIT=0" and "TRAIN DONE" in tail
            log(f"train finished: {done[-1]}; {'merged' if ok else tail[-800:]}", run_dir)
            if not ok:   # the tail above is the last 800 characters; the whole log is the evidence
                dest = os.path.join(run_dir, f"train-fail-{time.strftime('%Y%m%d-%H%M%S')}.log")
                try:
                    box.get("/workspace/train.log", dest)
                    log(f"full train log out: {dest}", run_dir)
                except Exception as e:
                    log(f"full train log not pulled: {type(e).__name__}: {str(e)[:120]}", run_dir)
            return ok


DISK_FLOOR = 10e9   # a download leaves at least this much free on this machine


def disk_short(dest_dir, nbytes, floor=None):
    """None when `nbytes` fit in dest_dir with `floor` (default DISK_FLOOR) left over, else why not. Measured
    2026-11-17: the Pro run scored every set, then lost its result writing a 22 GB base pack into 23 GB of free space
    (Errno 28)."""
    floor = DISK_FLOOR if floor is None else floor
    free = __import__("shutil").disk_usage(dest_dir).free
    if nbytes + floor > free:
        return f"needs {nbytes / 1e9:.2f} GB plus a {floor / 1e9:.0f} GB floor, {free / 1e9:.2f} GB free"
    return None


def model_bytes(box, name):
    """The model's size as the box's Ollama lists it (0 when it is not listed)."""
    code, out = box.sh(f"ollama list | awk '$1 == \"{name}:latest\" || $1 == \"{name}\" {{print $3, $4}}'", timeout=60)
    try:
        n, unit = out.split()[:2]
        return int(float(n) * {"KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}[unit.upper()])
    except (ValueError, KeyError):
        return 0


def pull_model(box, name, run_dir):
    """The Ollama model as one tar rooted at ~/.ollama/models (s2/pack_ollama.py: Ollama 0.40.8 keeps a
    manifests-v2 symlink to a manifest blob, not the old manifests/ tree), so `tar xf` there on a Mac
    makes `ollama list` show it. Returns what happened in words. Free disk here is checked before the pack (a pack
    of a model that cannot land is box time for nothing) and again with the pack's real size before the download;
    short of space it skips with the reason. Never raises: the scores are already out, a pack is an extra."""
    dest = os.path.join(run_dir, f"ollama-{name}.tar")
    if os.path.exists(dest):
        return "already here"
    try:
        short = disk_short(run_dir, model_bytes(box, name))
        if short:
            log(f"pack {name} SKIPPED before packing: {short}", run_dir)
            return "skipped: " + short
        box.put(os.path.join(HERE, "pack_ollama.py"), "/workspace/pack_ollama.py")
        code, out = box.sh(f"python /workspace/pack_ollama.py {name} /workspace/{name}.tar", timeout=900)
        log(f"pack {name}: exit {code} {out[-80:]}", run_dir)
        if code != 0:
            return f"pack failed: {out[-200:]}"
        code, size = box.sh(f"stat -c %s /workspace/{name}.tar", timeout=60)
        short = disk_short(run_dir, int(size)) if code == 0 and size.strip().isdigit() else "pack size unreadable"
        if short:
            log(f"pull {name} SKIPPED: {short}", run_dir)
            return "skipped: " + short
        t0 = time.time()
        box.get(f"/workspace/{name}.tar", dest + ".part")
        os.replace(dest + ".part", dest)
        log(f"pulled {name}: {os.path.getsize(dest) / 1e9:.2f} GB in {time.time() - t0:.0f} s", run_dir)
        return "pulled"
    except Exception as e:
        if os.path.exists(dest + ".part"):
            os.remove(dest + ".part")
        why = f"{type(e).__name__}: {str(e)[:200]}"
        log(f"pull {name} FAILED, the scores stand: {why}", run_dir)
        return "failed: " + why


def one_box(a, run_dir, tag, base_tag, reservation=None):
    with TrainBox(a.gpu.split(","), "s2-train-" + os.path.basename(run_dir), max_minutes=a.max_minutes,
                  log=lambda m: log(m, run_dir), owner=a.owner, reservation=reservation) as box:
        gpu = box.info["gpu_type"].split(".")[-1].replace("_", "-")
        price = PRICE.get(gpu, max(PRICE.values())) + BOX_EXTRA
        log(f"box {box.info['id'][:8]} {gpu} at ${price:.2f}/h", run_dir)
        t0 = time.time()
        try:
            box.sh("mkdir -p /workspace", timeout=60)
            code, out = box.run_long("boot", BOOT, max_s=1800, poll_s=15)
            log(f"boot exit {code}: {out.strip().splitlines()[-2:] if out.strip() else ''}", run_dir)
            if code != 0:
                return "boot failed", (time.time() - t0) / 3600 * price
            box.put(a.data, "/workspace/train.jsonl")
            box.put(os.path.join(HERE, "box_train.py"), "/workspace/box_train.py")
            code, out = box.run_long("base", "python -c \"from huggingface_hub import snapshot_download as s; "
                                     f"s('{a.model}', local_dir='/workspace/base')\"", max_s=1800, poll_s=15)
            log(f"base weights exit {code}", run_dir)
            if not train(box, a, run_dir):
                return "train failed", (time.time() - t0) / 3600 * price
            if a.until == "train":
                return "done", (time.time() - t0) / 3600 * price
            builds = ((tag, "/workspace/out/merged"),) if a.skip_base else (
                (base_tag, "/workspace/base"), (tag, "/workspace/out/merged"))
            for name, src in builds:
                box.sh(f"printf '{MODELFILE.format(src=src)}' > /workspace/Modelfile.{name}", timeout=60)
                code, out = box.run_long(f"create-{name}", f"ollama create {name} -f /workspace/Modelfile.{name} "
                                         "--quantize nvfp4", max_s=1800, poll_s=10)
                log(f"import {name}: exit {code} {out.strip().splitlines()[-2:] if out.strip() else ''}", run_dir)
                if code != 0:
                    return f"import {name} failed", (time.time() - t0) / 3600 * price
            client = box.ollama()
            for name, _ in builds:
                w0 = time.time()
                client.post("/api/chat", {"model": name, "stream": False, "think": False,
                                          "messages": [{"role": "user", "content": "hi"}],
                                          "options": {"num_predict": 4}}, timeout=1800)
                log(f"{name}: warm in {time.time() - w0:.0f} s (first nvfp4 call compiles kernels)", run_dir)
                score_model(client, name, run_dir, a.parallel, a.passes)
            if a.pull:   # only the trained model: a base pack is the public base, never worth the disk (Pro, 2026-11-17)
                pulled = pull_model(box, tag, run_dir)
                try:   # a record of the pull must never fail a scored run either
                    json.dump({"model": tag, "pull": pulled}, open(os.path.join(run_dir, "pull.json"), "w"))
                except OSError as e:
                    log(f"pull.json not written ({e}); pull: {pulled}", run_dir)
            return "done", (time.time() - t0) / 3600 * price
        except Exception as e:
            why = f"{type(e).__name__}: {str(e)[:300]}"
            log(f"box lost or step failed: {why}; gone={box.gone()}", run_dir)
            return ("evicted" if box.gone() else "failed: " + why), (time.time() - t0) / 3600 * price


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True, help="Hugging Face id, e.g. Qwen/Qwen3.5-4B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--gpu", default="H200,H100",
                    help="first type with capacity wins. Measured 2026-10-22 evening: 34 of 34 sweep boxes came up "
                         "H100 with H100 listed first and were reclaimed every few minutes, while H200 teacher boxes "
                         "ran 1.5 h untouched. Hopper only: the fast kernels are checked against reference there")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-every", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=8192, help="chat turns carry about 5,000 tokens of prompt")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--cap", type=float, default=50.0, help="training $40 plus scoring $10")
    ap.add_argument("--max-minutes", type=int, default=300)
    ap.add_argument("--until", choices=["train", "all"], default="all", help="train: stop once the adapter is out")
    ap.add_argument("--passes", type=int, default=3, help="score every set this many times (the charter's reruns)")
    ap.add_argument("--tag", default="", help="Ollama name for the trained model (sweeps need one each)")
    ap.add_argument("--skip-base", action="store_true", help="another run of the sweep scores this base")
    ap.add_argument("--no-pull", dest="pull", action="store_false",
                    help="keep only the adapter; a winner is rebuilt later (bounded space)")
    ap.add_argument("--micro", type=int, default=8, help="examples per forward pass")
    ap.add_argument("--micro-tokens", type=int, default=48000, help="token ceiling per forward pass")
    ap.add_argument("--meter", default="", help="path to a spend meter CLI (reserve/settle) to call around every box")
    ap.add_argument("--meter-lane", default="grux-local")
    ap.add_argument("--charter", default="", help="swing charter file the meter checks before a GPU box")
    ap.add_argument("--owner", default="grux", help="grux.factory owner label on every box")
    ap.add_argument("--data-scan-vocab", default="", help="identity vocabulary files for s2/data_scan.py (colon list)")
    a = ap.parse_args()
    run_dir = os.path.abspath(a.run)
    os.makedirs(run_dir, exist_ok=True)
    a.data = os.path.abspath(a.data)
    size = a.model.split("-")[-1].lower()
    tag, base_tag = a.tag or f"grux-qwen35-{size}", f"base-qwen35-{size}"
    json.dump({**vars(a), "tag": tag, "base_tag": base_tag, "holdout_sha": "1623e0e7", "think": suite.THINK,
               "train_rows": sum(1 for _ in open(a.data))}, open(os.path.join(run_dir, "manifest.json"), "w"), indent=1)
    if a.until == "all":   # scoring runs on this machine against the box: prove every set builds before paying
        bad = preflight_sets()
        if bad:
            log("REFUSED scoring preflight: " + " | ".join(bad), run_dir)
            json.dump({"status": "scoring preflight refused: " + " | ".join(bad)},
                      open(os.path.join(run_dir, "status.json"), "w"))
            sys.exit(4)
    if not a.meter:
        sys.exit("no spend meter (--meter): every box needs its own reservation; nothing created")
    import data_scan   # no secret shape or identity term goes into a box (s2/data_scan.py), checked before any create
    ok, msg = data_scan.check(a.data, a.data_scan_vocab or os.environ.get("GRUX_DATA_SCAN_VOCAB", ""),
                              require_vocab=True)   # name your identity terms, or nothing is uploaded
    log(msg, run_dir)
    if not ok:
        json.dump({"status": "data scan refused: " + msg}, open(os.path.join(run_dir, "status.json"), "w"))
        sys.exit(3)
    spent = float(json.load(open(os.path.join(run_dir, "spent.json")))["spent"]) if os.path.exists(
        os.path.join(run_dir, "spent.json")) else 0.0
    status = "not started"
    for attempt in range(8):                     # checkpoints keep progress and --cap bounds spend
        if spent >= a.cap:
            log(f"STOP: cap ${a.cap} reached (${spent:.2f})", run_dir)
            break
        rid, t_box = None, time.time()
        if a.meter:   # a shared spend meter, reserved before every box create; exit 3 refuses the box
            gpu = max(a.gpu.split(","), key=lambda g: PRICE.get(g, 0))
            hours = max(0.25, (a.cap - spent) / (PRICE.get(gpu, max(PRICE.values())) + BOX_EXTRA))
            cmd = [sys.executable, a.meter, "reserve", "--lane", a.meter_lane, "--hours", f"{hours:.3f}",
                   "--gpu", gpu.replace("-", ""), "--purpose", "train"] + (["--charter", a.charter] if a.charter else [])
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                log(f"STOP: the spend meter refused the box (exit {r.returncode}): {r.stderr.strip()[:200]}", run_dir)
                status = "meter refused"
                break
            rid = r.stdout.strip()
            if not rid:
                log("STOP: the spend meter returned no reservation id", run_dir)
                status = "meter refused"
                break
        try:
            status, cost = one_box(a, run_dir, tag, base_tag, rid)
        except Exception as e:   # a box reclaimed while it starts raises from create (measured 2026-10-22)
            log(f"box {attempt + 1} never came up: {type(e).__name__}: {str(e)[:160]}", run_dir)
            status, cost = "evicted", 0.0
            time.sleep(30)
        if rid:
            subprocess.run([sys.executable, a.meter, "settle", rid, "--hours", f"{(time.time() - t_box) / 3600:.4f}"],
                           capture_output=True, text=True)
        spent += cost
        json.dump({"spent": round(spent, 2)}, open(os.path.join(run_dir, "spent.json"), "w"))
        log(f"box {attempt + 1}: {status}, spent about ${spent:.2f}", run_dir)
        if status != "evicted":
            break
    json.dump({"status": status}, open(os.path.join(run_dir, "status.json"), "w"))
    for name in sorted(os.listdir(run_dir)):     # a model tar only passes through here on its way to the Mac
        if name.startswith("ollama-") and name.endswith(".part"):
            os.remove(os.path.join(run_dir, name))
    log(f"FINISHED: spent about ${spent:.2f}", run_dir)


if __name__ == "__main__":
    main()
