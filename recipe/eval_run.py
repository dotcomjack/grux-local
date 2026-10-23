#!/usr/bin/env python3
"""S1 local model eval runner (see recipe/README.md).

One ephemeral GPU box serves Ollama; this process (outside the box) sends every synthetic case
to every Cookbook model and appends each scored result to results/<run>/<model>.jsonl the moment
it returns, so a spot eviction loses at most the requests in flight. Resumes a run directory:
cases already scored are skipped. Stops at the card's dollar cap.

  DAYTONA_API_KEY=<your key> \
    python eval_run.py --run results/s1-<stamp> [--models a,b] [--gpu H200,H100] [--cap 40]
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import catalog  # noqa: E402
import suite  # noqa: E402
from gpu_box import OllamaBox  # noqa: E402

ONLY_THINKING = False
# Listed GPU rate per hour, plus what every GPU box also bills: 16 vCPU at $0.0504 and 192 GB RAM at
# $0.0162 an hour, $3.92 (measured 2026-10-22: the wallet moved $23.04 where GPU rates alone said $12).
BOX_EXTRA = 16 * 0.0504 + 192 * 0.0162
PRICE = {"H100": 2.27, "H200": 2.61, "B200": 3.59, "B300": 4.08, "RTX-PRO-6000": 1.74,
         "RTX-4090": 0.57, "RTX-5090": 0.74, "MI355X": 3.44}


def log(msg, run_dir=None):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if run_dir:
        with open(os.path.join(run_dir, "run.log"), "a") as f:
            f.write(line + "\n")


def done_ids(path):
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("status") in ("scored", "skipped"):
            out.add(r["case"])
    return out


def one(box, model, case):
    req = suite.request(case, model)
    t0 = time.time()
    try:
        raw = box.post(req["path"], req["body"], timeout=600)
        err = None
    except Exception as e:  # recorded, retried once by the caller
        raw, err = None, f"{type(e).__name__}: {str(e)[:200]}"
    secs = round(time.time() - t0, 2)
    if err:
        return {"case": case["id"], "family": case["family"], "status": "error", "error": err, "s": secs}
    ok, why, got = suite.score(case, raw)
    usage = raw.get("usage") or {}
    return {"case": case["id"], "family": case["family"], "status": "scored", "pass": ok, "why": why,
            "got": got, "s": secs, "out_tokens": usage.get("completion_tokens", raw.get("eval_count"))}


def run_model(box, model, cases, path, run_dir, parallel):
    have = done_ids(path)
    todo = [c for c in cases if c["id"] not in have]
    if not todo:
        log(f"{model}: already complete ({len(have)} cases)", run_dir)
        return True
    status, pull_s = box.pull(model)
    log(f"{model}: pull {status} in {pull_s}s; {len(todo)} cases to run", run_dir)
    if status != "success":
        return False
    try:
        show = box.post("/api/show", {"model": model}, timeout=60)
        caps = {"capabilities": show.get("capabilities"), "details": show.get("details"),
                "params": (show.get("parameters") or "")[:400]}
    except Exception as e:
        caps = {"error": str(e)[:200]}
    with open(os.path.join(run_dir, "capabilities.jsonl"), "a") as f:
        f.write(json.dumps({"model": model, **caps}) + "\n")
    log(f"{model}: capabilities {caps.get('capabilities')}", run_dir)
    if ONLY_THINKING and "thinking" not in (caps.get("capabilities") or []):
        log(f"{model}: does not think; nothing to measure with think false", run_dir)
        box.drop(model)
        return True
    warm = one(box, model, cases[0])  # first call carries the model load; never timed
    log(f"{model}: warm {warm['s']}s ({warm['status']})", run_dir)
    retry = []
    with open(path, "a") as out, cf.ThreadPoolExecutor(parallel) as pool:
        for r in pool.map(lambda c: one(box, model, c), todo):
            if r["status"] == "error":
                retry.append(r["case"])
                continue
            out.write(json.dumps(r) + "\n")
            out.flush()
        for cid in retry:
            if box.evicted():
                return False
            r = one(box, model, next(c for c in cases if c["id"] == cid))
            if r["status"] == "error":
                r["status"] = "skipped"
            out.write(json.dumps(r) + "\n")
            out.flush()
    box.drop(model)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--models", default="")
    ap.add_argument("--gpu", default="H200,H100")
    ap.add_argument("--cap", type=float, default=40.0)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--ttl", type=int, default=600)
    ap.add_argument("--think", choices=["unset", "false"], default="unset",
                    help="unset sends no think field, as Grux does today")
    ap.add_argument("--limit", type=int, default=0, help="first N cases per family (smoke runs)")
    ap.add_argument("--only-thinking", action="store_true", help="skip models without the thinking capability")
    ap.add_argument("--candidates", action="store_true", help="run candidates.json instead of the Cookbook catalog")
    a = ap.parse_args()
    global ONLY_THINKING
    ONLY_THINKING = a.only_thinking
    if a.think == "false":
        suite.THINK = False
    run_dir = os.path.abspath(a.run)
    os.makedirs(run_dir, exist_ok=True)
    cases = suite.cases()
    if a.limit:
        seen = {}
        cases = [c for c in cases if seen.setdefault(c["family"], []).append(1) or len(seen[c["family"]]) <= a.limit]
    pool = catalog.candidates() if a.candidates else catalog.catalog()
    models = [m["id"] for m in pool]
    if a.models:
        models = [m for m in models if m in a.models.split(",")]
    models.sort(key=lambda mid: next(m["memGB"] for m in pool if m["id"] == mid))
    json.dump({"cases": len(cases), "families": suite.family_counts(cases), "models": models,
               "gpu": a.gpu, "cap": a.cap, "think": a.think, "suite_sha": suite.fingerprint(cases)},
              open(os.path.join(run_dir, "manifest.json"), "w"), indent=1)
    spent = 0.0
    attempts = 0
    pending = list(models)
    while pending and attempts < 3:
        attempts += 1
        with OllamaBox(a.gpu.split(","), "lme-" + os.path.basename(run_dir), ttl_minutes=a.ttl,
                       log=lambda m: log(m, run_dir)) as box:
            gpu = str(box.info.get("gpu_type")).split(".")[-1].replace("_", "-").strip("'>")
            gpu = next((g for g in PRICE if g.replace("-", "") == gpu.replace("-", "").upper()), gpu)
            price = PRICE.get(gpu, max(PRICE.values())) + BOX_EXTRA
            log(f"box {box.info.get('id')} {gpu} at ${price}/h, spent so far ${spent:.2f}, "
                f"ollama {box.get('/api/version').get('version')}", run_dir)
            t0 = time.time()
            while pending:
                cost_now = spent + (time.time() - t0) / 3600 * price
                if cost_now >= a.cap:
                    log(f"STOP: cap ${a.cap} reached (${cost_now:.2f})", run_dir)
                    pending = []
                    break
                model = pending[0]
                path = os.path.join(run_dir, model.replace(":", "_").replace("/", "_") + ".jsonl")
                if run_model(box, model, cases, path, run_dir, a.parallel):
                    pending.pop(0)
                    log(f"{model}: done, cost so far ${spent + (time.time() - t0) / 3600 * price:.2f}", run_dir)
                elif box.evicted():
                    log(f"box evicted during {model}; a new box resumes it", run_dir)
                    break
                else:
                    log(f"{model}: could not run; recorded and skipped", run_dir)
                    pending.pop(0)
            spent += (time.time() - t0) / 3600 * price
        log(f"box closed: left {box.info.get('boxes_left')}, spent ${spent:.2f}", run_dir)
    json.dump({"spent_estimate": round(spent, 2), "unfinished": pending},
              open(os.path.join(run_dir, "finish.json"), "w"), indent=1)
    log(f"FINISHED: spent about ${spent:.2f}, unfinished {pending}", run_dir)


if __name__ == "__main__":
    main()
