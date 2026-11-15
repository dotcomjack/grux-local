#!/usr/bin/env python3
"""S1 local model eval runner (see recipe/README.md).

One ephemeral GPU box serves Ollama; this process (outside the box) sends every synthetic case
to every Cookbook model and appends each scored result to results/<run>/<model>.jsonl the moment
it returns, so a spot eviction loses at most the requests in flight. Resumes a run directory:
cases already scored are skipped. Stops at the card's dollar cap.

  DAYTONA_API_KEY=<your key> \
    python eval_run.py --run results/s1-<stamp> [--models a,b] [--gpu H200,H100] [--cap 40]

--set holdout_v3 --models a,b scores held-out v3 (s2/holdout_v3.py) instead, each job step by step.
"""
import argparse
import urllib.error
import concurrent.futures as cf
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suite  # noqa: E402
from gpu_box import MLX_IMAGE, MLX_SETUP, OLLAMA_VERSION, OllamaBox  # noqa: E402
import meter  # noqa: E402

ONLY_THINKING = False
TEARDOWN_USD = lambda price: price * 0.05   # about 3 minutes of delete and in-flight jobs kept out of the cap
MIN_BOX_HOURS = 0.25   # create, setup and delete alone take minutes: no box is made for less than this
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


V3 = None   # holdout_v3 when --set holdout_v3


def one(box, model, case):
    if V3 is not None:
        return one_v3(box, model, case)
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


def one_v3(box, model, case):
    """A held-out v3 job, step by step through the box (holdout_v3.run_case); an error is retried by the caller."""
    t0 = time.time()
    try:
        ok, why, got, steps = V3.run_case(case, model, lambda p, b: box.post(p, b, timeout=600))
    except Exception as e:
        return {"case": case["id"], "family": case["family"], "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:200]}", "s": round(time.time() - t0, 2)}
    return {"case": case["id"], "family": case["family"], "status": "scored", "pass": ok, "why": why, "got": got,
            "s": round(time.time() - t0, 2), "steps": steps, "prompt_tokens": [x["prompt_tokens"] for x in steps],
            "out_tokens": sum(x["out_tokens"] or 0 for x in steps)}


def probe(box, model, expect_digest=None, timeout=900):
    """Step 0 for every model, before any case: does it load and answer, on which runner, with which digest.
    Returns (info, reason); reason is None when the model may be scored."""
    try:
        tags = box.get("/api/tags").get("models", [])
        digest = next((m.get("digest") for m in tags if model in (m.get("name"), m.get("model"))), None)
        r = box.post("/api/generate", {"model": model, "prompt": "ok", "stream": False,
                                       "options": {"num_predict": 4, "num_ctx": 4096}}, timeout=max(1, timeout))
        ps = box.get("/api/ps").get("models", [])
        m = next((x for x in ps if model in (x.get("name"), x.get("model"))), {})
        info = {"digest": digest, "runner": m.get("runner"), "format": (m.get("details") or {}).get("format"),
                "quant": (m.get("details") or {}).get("quantization_level"), "size_vram": m.get("size_vram"),
                "answered": bool(r.get("done"))}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}, "did not load"
    if expect_digest and not (digest or "").startswith(expect_digest):
        return info, f"digest {digest} is not the expected {expect_digest}"
    if not info["answered"]:
        return info, "loaded but gave no answer"
    return info, None


def manifest_url(model):
    """Where a registry serves this tag's manifest: host/ns/name:tag, or a library name:tag on registry.ollama.ai."""
    name, _, tag = model.partition(":")
    parts = name.split("/")
    if len(parts) >= 3 and "." in parts[0]:
        return f"https://{parts[0]}/v2/{'/'.join(parts[1:])}/manifests/{tag or 'latest'}"
    path = name if "/" in name else "library/" + name
    return f"https://registry.ollama.ai/v2/{path}/manifests/{tag or 'latest'}"


def missing_models(models, fetch=None):
    """Tags no registry serves. Checked before any box exists: 2026-11-14 a tag listed on ollama.com
    (qwen3.5:122b-a10b-q4) had no manifest, and the box only found out after scoring two other models."""
    import urllib.request

    def default_fetch(url):
        r = urllib.request.Request(url, method="GET", headers={   # the default Python agent is refused at the edge
            "Accept": "application/vnd.docker.distribution.manifest.v2+json", "User-Agent": "grux-local-eval/1"})
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status
        except urllib.error.HTTPError as e:
            return e.code

    fetch = fetch or default_fetch
    return [m for m in models if fetch(manifest_url(m)) != 200]


EARLY_ERRORS = 4   # a model whose first 4 cases all error is stopped, not run through all of them


def skip_max(n):
    """Skipped cases a pass may carry and still count: 5 or 2%, whichever is larger. Past it the pass is damaged."""
    return max(5, int(0.02 * n))


def remaining(deadline, cap_s):
    return cap_s if deadline is None else max(0.0, min(cap_s, deadline - time.time()))


def run_model(box, model, cases, path, run_dir, parallel, deadline=None, drop=True, expect_digest=None):
    """True when the model is done, False when it could not run, "cap" when the dollar cap was reached mid-model."""
    have = done_ids(path)
    todo = [c for c in cases if c["id"] not in have]
    if not todo:
        log(f"{model}: already complete ({len(have)} cases)", run_dir)
        return True
    if deadline is not None and time.time() >= deadline:
        return "cap"
    status, pull_s = box.pull(model, timeout=3600, deadline=deadline)
    log(f"{model}: pull {status} in {pull_s}s; {len(todo)} cases to run", run_dir)
    if deadline is not None and time.time() >= deadline:
        return "cap"
    if status != "success":
        return False
    info, why_not = probe(box, model, expect_digest, timeout=remaining(deadline, 900))
    try:
        show = box.post("/api/show", {"model": model}, timeout=60)
        caps = {"capabilities": show.get("capabilities"), "details": show.get("details"),
                "params": (show.get("parameters") or "")[:400]}
    except Exception as e:
        caps = {"error": str(e)[:200]}
    with open(os.path.join(run_dir, "capabilities.jsonl"), "a") as f:
        f.write(json.dumps({"model": model, **caps, "probe": info, "probe_verdict": why_not or "ok"}) + "\n")
    log(f"{model}: probe {info} -> {why_not or 'ok'}; capabilities {caps.get('capabilities')}", run_dir)
    if why_not:
        return False
    if ONLY_THINKING and "thinking" not in (caps.get("capabilities") or []):
        log(f"{model}: does not think; nothing to measure with think false", run_dir)
        box.drop(model)
        return True
    if deadline is not None and time.time() >= deadline:
        return "cap"
    warm = one(box, model, cases[0])  # first call carries the model load at the real window; never timed
    log(f"{model}: warm {warm['s']}s ({warm['status']})", run_dir)
    if warm["status"] == "error":   # loads at 4,096 but not at the real window: no pool of slow failures
        log(f"{model}: the first real job failed ({warm.get('error')}); not scored", run_dir)
        return False
    retry, seen, result = [], 0, True
    pool = cf.ThreadPoolExecutor(parallel)
    futs = [pool.submit(one, box, model, c) for c in todo]
    try:
        with open(path, "a") as out:
            for fut in cf.as_completed(futs):
                r = fut.result()
                seen += 1
                if r["status"] == "error":
                    retry.append(r["case"])
                    if seen == EARLY_ERRORS and len(retry) == EARLY_ERRORS:
                        log(f"{model}: first {EARLY_ERRORS} cases all failed ({r.get('error')}); stopped", run_dir)
                        result = False
                        break
                else:
                    out.write(json.dumps(r) + "\n")
                    out.flush()
                if deadline is not None and time.time() >= deadline:
                    log(f"{model}: STOP: dollar cap reached mid-model after {seen} of {len(todo)} cases", run_dir)
                    result = "cap"
                    break
            if result is True:
                skipped = 0
                for cid in retry:
                    if box.evicted():
                        return False
                    if deadline is not None and time.time() >= deadline:
                        log(f"{model}: STOP: dollar cap reached during retries", run_dir)
                        return "cap"
                    r = one(box, model, next(c for c in cases if c["id"] == cid))
                    if r["status"] == "error":
                        if box.evicted():   # the box went away mid-call: a new box reruns it, never a skip
                            return False
                        r["status"] = "skipped"
                        skipped += 1
                    out.write(json.dumps(r) + "\n")
                    out.flush()
                    if skipped > skip_max(len(cases)):
                        log(f"{model}: {skipped} cases failed twice; the server is not answering, pass stopped", run_dir)
                        return False
    finally:
        for f in futs:
            f.cancel()
        pool.shutdown(wait=True)
    if result is True and drop:
        box.drop(model)
    return result


def pass_pct(path):
    """(passed, counted, skipped). A skipped case counts as a fail: leaving it out of the denominator would
    inflate the score of a model that times out on the hardest jobs."""
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    scored = [r for r in rows if r.get("status") == "scored"]
    skipped = sum(r.get("status") == "skipped" for r in rows)
    return sum(bool(r.get("pass")) for r in scored), len(scored) + skipped, skipped


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
    ap.add_argument("--meter", default="", help="spend meter CLI: each box is reserved and settled on its own")
    ap.add_argument("--meter-lane", default="grux")
    ap.add_argument("--set", choices=["s1", "holdout_v3"], default="s1",
                    help="holdout_v3: held-out v3 (s2/holdout_v3.py), each job step by step; needs --models")
    ap.add_argument("--mlx", action="store_true",
                    help="serve through Ollama's MLX runtime (the Mac's nvfp4 format) on the pinned version")
    ap.add_argument("--passes", type=int, default=1, help="score each model this many times (the gate's reruns)")
    ap.add_argument("--kill-model", default="", help="stop the whole run if this model's first pass scores ...")
    ap.add_argument("--kill-at", type=float, default=0.0, help="... at or above this percent (the charter's kill rule)")
    ap.add_argument("--expect-digest", action="append", default=[],
                    help="model=digest-prefix: refuse to score a model whose pulled digest differs")
    a = ap.parse_args()
    global ONLY_THINKING, V3
    ONLY_THINKING = a.only_thinking
    if a.think == "false":
        suite.THINK = False
    run_dir = os.path.abspath(a.run)
    os.makedirs(run_dir, exist_ok=True)
    if a.set == "holdout_v3":
        sys.path.insert(0, os.path.join(HERE, "s2"))
        import holdout_v3
        V3 = holdout_v3
    cases = V3.cases() if V3 else suite.cases()
    if a.limit:
        seen = {}
        cases = [c for c in cases if seen.setdefault(c["family"], []).append(1) or len(seen[c["family"]]) <= a.limit]
    if V3:
        if not a.models:
            raise SystemExit("--set holdout_v3 needs --models (the catalog is the S1 Cookbook's)")
        models = a.models.split(",")
    else:
        import catalog  # S1 only (the Grux tree); train_run imports this module for BOX_EXTRA, PRICE, log
        pool = catalog.candidates() if a.candidates else catalog.catalog()
        models = [m["id"] for m in pool]
        if a.models:
            models = [m for m in models if m in a.models.split(",")]
        models.sort(key=lambda mid: next(m["memGB"] for m in pool if m["id"] == mid))
    expect = dict(x.split("=", 1) for x in a.expect_digest)
    if V3:
        gone = missing_models(models)
        if gone:
            raise SystemExit(f"REFUSED no registry serves {gone}; no box is created for a tag that cannot pull")
    json.dump({"cases": len(cases), "families": suite.family_counts(cases), "models": models, "set": a.set,
               "gpu": a.gpu, "cap": a.cap, "think": a.think, "mlx": a.mlx, "passes": a.passes,
               "kill": [a.kill_model, a.kill_at], "expect_digest": expect,
               "suite_sha": V3.case_sha(cases) if V3 else suite.fingerprint(cases)},
              open(os.path.join(run_dir, "manifest.json"), "w"), indent=1)
    spent = 0.0
    attempts = 0
    pending = [(m, k) for m in models for k in range(1, a.passes + 1)]
    not_run, stopped_by = [], None   # what a cap or the kill rule left unrun, and which one
    rate_gpu = max(a.gpu.split(","), key=lambda g: PRICE.get(g, 0))
    while pending and attempts < 3:
        attempts += 1
        hours = max(0.25, (a.cap - spent) / (PRICE.get(rate_gpu, max(PRICE.values())) + BOX_EXTRA))
        t_enter = time.time()   # the box bills from its create, so the cap clock starts here, not after setup
        price = PRICE.get(rate_gpu, max(PRICE.values())) + BOX_EXTRA
        if spent + TEARDOWN_USD(price) + price * MIN_BOX_HOURS >= a.cap:   # a new box would be billed for nothing
            log(f"STOP: ${a.cap - spent:.2f} left, under one useful box slice; no new box", run_dir)
            not_run, stopped_by, pending = list(pending), "cap", []
            break
        with meter.Reserved(a.meter, a.meter_lane, hours, rate_gpu, "measure") as rid, \
                OllamaBox(a.gpu.split(","), "lme-" + os.path.basename(run_dir), ttl_minutes=a.ttl,
                          log=lambda m: log(m, run_dir), reservation=rid, num_parallel=a.parallel,
                          **({"image": MLX_IMAGE, "setup": MLX_SETUP, "expect_version": OLLAMA_VERSION}
                             if a.mlx else {})) as box:
            gpu = str(box.info.get("gpu_type")).split(".")[-1].replace("_", "-").strip("'>")
            gpu = next((g for g in PRICE if g.replace("-", "") == gpu.replace("-", "").upper()), gpu)
            price = PRICE.get(gpu, max(PRICE.values())) + BOX_EXTRA
            log(f"box {box.info.get('id')} {gpu} at ${price}/h, spent so far ${spent:.2f}, "
                f"ollama {box.get('/api/version').get('version')}", run_dir)
            t0 = t_enter
            deadline = t0 + max(0.0, a.cap - spent - TEARDOWN_USD(price)) / price * 3600
            while pending:
                cost_now = spent + (time.time() - t0) / 3600 * price
                if cost_now >= a.cap:
                    log(f"STOP: cap ${a.cap} reached (${cost_now:.2f})", run_dir)
                    not_run, stopped_by, pending = list(pending), "cap", []
                    break
                model, k = pending[0]
                path = os.path.join(run_dir, model.replace(":", "_").replace("/", "_")
                                    + ("-holdout_v3" if V3 else "") + (f"-pass{k}" if a.passes > 1 else "") + ".jsonl")
                last = not any(m == model for m, _ in pending[1:])
                got = run_model(box, model, cases, path, run_dir, a.parallel, deadline=deadline, drop=last,
                                expect_digest=expect.get(model))
                if got == "cap":
                    log(f"STOP: cap ${a.cap} reached during {model} pass {k}", run_dir)
                    not_run, stopped_by, pending = list(pending), "cap", []
                    break
                if got:
                    pending.pop(0)
                    passed, scored, skipped = pass_pct(path)
                    log(f"{model} pass {k}: done, {passed} of {scored} passed ({skipped} skipped, counted as fails), "
                        f"cost so far ${spent + (time.time() - t0) / 3600 * price:.2f}", run_dir)
                    if (model == a.kill_model and k == 1 and scored and skipped <= skip_max(len(cases))
                            and 100.0 * passed / scored >= a.kill_at > 0):
                        log(f"KILL: {model} scored {100.0 * passed / scored:.1f}%, at or above {a.kill_at}%: "
                            "no headroom, the run stops (charter kill rule)", run_dir)
                        not_run, stopped_by, pending = list(pending), "kill rule", []
                        break
                elif box.evicted():
                    log(f"box evicted during {model}; a new box resumes it", run_dir)
                    break
                else:
                    log(f"{model}: could not run; recorded and skipped (all its passes)", run_dir)
                    pending = [(m, j) for m, j in pending if m != model]
        spent += (time.time() - t_enter) / 3600 * price   # after the with: the delete is billed too
        log(f"box closed: left {box.info.get('boxes_left')}, spent ${spent:.2f}", run_dir)
    json.dump({"spent_estimate": round(spent, 2), "unfinished": pending + not_run, "stopped_by": stopped_by},
              open(os.path.join(run_dir, "finish.json"), "w"), indent=1)
    log(f"FINISHED: spent about ${spent:.2f}, unfinished {pending}", run_dir)


if __name__ == "__main__":
    main()
