#!/usr/bin/env python3
"""S2 train and score run (see recipe/README.md; caps: training $40 per model, eval $10).

One leased spot box at a time: train a LoRA (every checkpoint pulled out the moment it lands), merge,
import the base and the trained model into the box's Ollama as nvfp4 (the format a Mac serves),
score both on the S1 suite and the frozen held-out set with thinking off, then pull both Ollama
models out for the Mac check. A spot box goes without warning (measured 2026-10-22), so a new box
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
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import holdout  # noqa: E402
import suite  # noqa: E402
from eval_run import BOX_EXTRA, PRICE, log  # noqa: E402
from train_box import BOOT, TrainBox  # noqa: E402

suite.THINK = False   # Grux sends think false on every Ollama call (loop row D-THINK, PROVEN)
MODELFILE = ("FROM {src}\nRENDERER qwen3.5\nPARSER qwen3.5\nPARAMETER temperature 1\nPARAMETER top_k 20\n"
             "PARAMETER top_p 0.95\nPARAMETER presence_penalty 1.5\nPARAMETER min_p 0\n")
SETS = {"s1": (suite.cases, suite.request, suite.score),
        "holdout": (holdout.cases, holdout.request_holdout, holdout.score_holdout)}


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


def score_model(client, model, run_dir, parallel):
    """Every case of every set for one model; appends as it goes, skips what is already scored."""
    def one(case, request, score):
        req = request(case, model)
        t0 = time.time()
        for attempt in (1, 2):
            try:
                raw = client.post(req["path"], req["body"], timeout=900)
                break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(5)
        else:
            return {"case": case["id"], "family": case["family"], "status": "error", "error": err}
        ok, why, got = score(case, raw)
        return {"case": case["id"], "family": case["family"], "status": "scored", "pass": ok, "why": why,
                "got": got, "s": round(time.time() - t0, 2)}

    for name, (cases_fn, request, score) in SETS.items():
        cases = cases_fn()
        path = os.path.join(run_dir, f"eval-{model}-{name}.jsonl")
        todo = [c for c in cases if c["id"] not in scored(path)]
        if not todo:
            continue
        log(f"{model} {name}: {len(todo)} of {len(cases)} cases to score", run_dir)
        with open(path, "a") as f, cf.ThreadPoolExecutor(parallel) as pool:
            for r in pool.map(lambda c: one(c, request, score), todo):
                if r["status"] == "scored":
                    f.write(json.dumps(r) + "\n")
                    f.flush()
        rows = [json.loads(l) for l in open(path)]
        log(f"{model} {name}: {sum(r['pass'] for r in rows)}/{len(rows)} pass", run_dir)


def train(box, a, run_dir):
    """Train (fresh, or resumed from the last checkpoint out), pulling checkpoints and the adapter."""
    ckpt_tar, adapter_tar = os.path.join(run_dir, "ckpt.tar"), os.path.join(run_dir, "adapter.tar")
    if os.path.exists(adapter_tar):
        box.put(adapter_tar, "/workspace/adapter.tar")
        code, out = box.sh("cd /workspace && mkdir -p out && tar xf adapter.tar -C out && ls out/adapter", timeout=300)
        log(f"adapter already out: merge only ({out[-80:]})", run_dir)
        cmd = f"python box_train.py --model /workspace/base --data train.jsonl --out out --merge-only out/adapter"
    else:
        resume = ""
        if os.path.exists(ckpt_tar):
            box.put(ckpt_tar, "/workspace/ckpt.tar")
            box.sh("cd /workspace && mkdir -p resume && tar xf ckpt.tar -C resume", timeout=300)
            resume = " --resume resume/ckpt"
            log(f"resuming from {json.load(open(ckpt_tar + '.json'))}", run_dir)
        cmd = (f"python box_train.py --model /workspace/base --data train.jsonl --out out --epochs {a.epochs} "
               f"--rank {a.rank} --accum {a.accum} --lr {a.lr} --save-every {a.save_every} --max-len {a.max_len} "
               f"--micro {a.micro} --micro-tokens {a.micro_tokens}{resume}")
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
            return ok


def pull_model(box, name, run_dir):
    """The Ollama model as one tar rooted at ~/.ollama/models (s2/pack_ollama.py: Ollama 0.40.8 keeps a
    manifests-v2 symlink to a manifest blob, not the old manifests/ tree), so `tar xf` there on a Mac
    makes `ollama list` show it."""
    dest = os.path.join(run_dir, f"ollama-{name}.tar")
    if os.path.exists(dest):
        return
    box.put(os.path.join(HERE, "pack_ollama.py"), "/workspace/pack_ollama.py")
    code, out = box.sh(f"python /workspace/pack_ollama.py {name} /workspace/{name}.tar", timeout=900)
    log(f"pack {name}: exit {code} {out[-80:]}", run_dir)
    if code != 0:
        raise RuntimeError(f"pack {name} failed: {out[-200:]}")
    t0 = time.time()
    box.get(f"/workspace/{name}.tar", dest + ".part")
    os.replace(dest + ".part", dest)
    log(f"pulled {name}: {os.path.getsize(dest) / 1e9:.2f} GB in {time.time() - t0:.0f} s", run_dir)


def one_box(a, run_dir, tag, base_tag):
    with TrainBox(a.gpu.split(","), "s2-train-" + os.path.basename(run_dir), max_minutes=a.max_minutes,
                  log=lambda m: log(m, run_dir)) as box:
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
                score_model(client, name, run_dir, a.parallel)
            for name, _ in (builds if a.pull else ()):
                pull_model(box, name, run_dir)
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
    ap.add_argument("--tag", default="", help="Ollama name for the trained model (sweeps need one each)")
    ap.add_argument("--skip-base", action="store_true", help="another run of the sweep scores this base")
    ap.add_argument("--no-pull", dest="pull", action="store_false",
                    help="keep only the adapter; a winner is rebuilt later (bounded space)")
    ap.add_argument("--micro", type=int, default=8, help="examples per forward pass")
    ap.add_argument("--micro-tokens", type=int, default=48000, help="token ceiling per forward pass")
    a = ap.parse_args()
    run_dir = os.path.abspath(a.run)
    os.makedirs(run_dir, exist_ok=True)
    a.data = os.path.abspath(a.data)
    size = a.model.split("-")[-1].lower()
    tag, base_tag = a.tag or f"grux-qwen35-{size}", f"base-qwen35-{size}"
    json.dump({**vars(a), "tag": tag, "base_tag": base_tag, "holdout_sha": "1623e0e7",
               "train_rows": sum(1 for _ in open(a.data))}, open(os.path.join(run_dir, "manifest.json"), "w"), indent=1)
    spent = float(json.load(open(os.path.join(run_dir, "spent.json")))["spent"]) if os.path.exists(
        os.path.join(run_dir, "spent.json")) else 0.0
    status = "not started"
    for attempt in range(8):                     # checkpoints keep progress and --cap bounds spend
        if spent >= a.cap:
            log(f"STOP: cap ${a.cap} reached (${spent:.2f})", run_dir)
            break
        try:
            status, cost = one_box(a, run_dir, tag, base_tag)
        except Exception as e:   # a box reclaimed while it starts raises from create (measured 2026-10-22)
            log(f"box {attempt + 1} never came up: {type(e).__name__}: {str(e)[:160]}", run_dir)
            status, cost = "evicted", 0.0
            time.sleep(30)
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
