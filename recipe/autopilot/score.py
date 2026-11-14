#!/usr/bin/env python3
"""Score base models stock on Grux's two held-out sets, one spot GPU box per model, all in parallel.

Each model is scored as a Mac runs it: the default Ollama library build of that tag, pulled by the box's own
Ollama (the same manifest digest any Mac pulls; the result records it), thinking off, num_ctx 16,384, one
request at a time, every request built by Grux's own prompts and scored by Grux's own parsers (eval_local.py's
exact dispatch, held-out v1 464 and held-out v2 466). The harness runs here, outside the box; nothing of ours
goes into the box, which only pulls a public tag and answers HTTP.

Progress saving (spot boxes go without warning): every finished task is one JSONL line in this machine's
outbox before the next is sent; a run starts by reading what is done, so a replacement box (at most 3 per
model) or a rerun of the same command resumes there. A task that fails on its own is retried once, then
recorded as skipped.

Spend: before EVERY box create, the shared weekly meter is reserved (exit 3 refuses: no box); after, it is
settled with the real hours. A swing estimated over $150 refuses before any reservation or box.

  DAYTONA_API_KEY=<your key> <venv>/bin/python recipe/autopilot/score.py \\
      --models mimo-v2.6:9b,qwen3.5:9b [--gpu H200,H100] [--est-hours 1.2]
Writes results/autopilot/<date>-<model>.json per model.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SETS = ("holdout", "holdout2")
SET_SIZE = {"holdout": 464, "holdout2": 466}
SET_HASH = {"holdout": "7c3f2a9ff3316ecc", "holdout2": "da6577c7178a0963"}
MAX_BOXES = 3      # boxes that came up, per invocation (an eviction is replaced, at most 3 times)
MAX_ATTEMPTS = 6   # creates per invocation, counting the ones spot capacity refused
NUM_CTX = 16384


def check_prompts():
    """The scorer must send exactly the requests the published numbers were measured with, or nothing runs."""
    common.ensure_grux_src()
    r = subprocess.run([sys.executable, os.path.join(common.RECIPE, "check_prompts.py")], capture_output=True,
                       text=True, env=os.environ)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    if r.returncode != 0 or len(lines) != 2 or not all(l.endswith("MATCH") for l in lines):
        raise SystemExit("check_prompts.py did not print two MATCH lines; refusing to score:\n" + r.stdout + r.stderr)
    return lines


def set_fns(name):
    import suite
    suite.THINK = False
    suite.NUM_CTX = NUM_CTX
    if name == "holdout":
        import holdout
        return holdout.cases, holdout.request_holdout, holdout.score_holdout
    import holdout_v2
    return holdout_v2.cases, holdout_v2.request_v2, holdout_v2.score_v2


def registry_digest(model):
    """The digest any client without a key is served for this tag (equal to the box's /api/tags digest on the three
    models scored 2026-11-10); None when the registry cannot be read."""
    import watch
    try:
        name, size = model.split(":")
        return watch.registry_build(name, size)["digest"]
    except Exception:  # noqa: BLE001
        return None


def done_ids(path):
    out = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "scored":         # a skipped task is sent again on the next box
                out.add(r["case"])
    return out


def outbox_rows(path):
    """The last row per case (a skipped row is superseded when a later box scores it). Two SCORED rows for one
    case would mean a resume bug, so that refuses."""
    rows, scored = {}, set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("status") == "scored":
                if r["case"] in scored:
                    raise SystemExit(f"{path}: case {r['case']} scored twice (a resume bug)")
                scored.add(r["case"])
            elif r["case"] in scored:
                continue
            rows[r["case"]] = r
    return list(rows.values())


def outbox_complete(run_dir):
    return all(sum(r.get("status") == "scored" for r in outbox_rows(os.path.join(run_dir, f"{s}.jsonl"))) == SET_SIZE[s]
               for s in SETS)


def one_case(box, name, c, request, score, model):
    """eval_local.py's row, sent through the box. Raises on transport errors (the caller decides retry vs evicted)."""
    req = request(c, model)
    t = time.time()
    raw = box.post(req["path"], req["body"], timeout=600)
    ok, why, got = score(c, raw, req["offered"]) if name == "holdout2" and "offered" in req else score(c, raw)
    row = {"case": c["id"], "family": c["family"], "status": "scored", "pass": ok, "why": why, "got": got,
           "s": round(time.time() - t, 2), "out_tokens": raw.get("eval_count")}
    if "offered" in req:
        row["offered"] = req["offered"]
        row["prompt_tokens"] = raw.get("prompt_eval_count")
    return row


def score_on_box(box, model, run_dir, logp, deadline):
    """Every remaining task of both sets on this box. Returns 'done' (every task sent once), 'lost' (the box is
    gone) or 'deadline' (the box has used the hours reserved for it)."""
    for name in SETS:
        cases_fn, request, score = set_fns(name)
        cases = cases_fn()
        path = os.path.join(run_dir, f"{name}.jsonl")
        have = done_ids(path)
        todo = [c for c in cases if c["id"] not in have]
        common.log(f"{model} {name}: {len(have)} of {len(cases)} already out, {len(todo)} to send", logp)
        with open(path, "a") as out:
            for i, c in enumerate(todo):
                if time.time() > deadline:
                    common.log(f"{model}: past the reserved hours at {name} {c['id']}; stopping this box", logp)
                    return "deadline"
                row = None
                for attempt in (1, 2):
                    try:
                        row = one_case(box, name, c, request, score, model)
                        break
                    except Exception as e:  # noqa: BLE001 (a dead box surfaces as any transport error)
                        err = f"{type(e).__name__}: {str(e)[:200]}"
                        if box.evicted():
                            common.log(f"{model}: box gone at {name} {c['id']} ({err})", logp)
                            return "lost"
                        time.sleep(5)
                if row is None:
                    row = {"case": c["id"], "family": c["family"], "status": "skipped", "error": err}
                out.write(json.dumps(row) + "\n")
                out.flush()
                if (i + 1) % 50 == 0:
                    common.log(f"{model} {name}: {len(have) + i + 1} of {len(cases)} out", logp)
    return "done"


def summarize(rows):
    fam = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        fam[r["family"]][0] += 1
        fam[r["family"]][1] += bool(r.get("pass"))
    chat = sorted(r["s"] for r in rows if "s" in r and (r["family"].startswith("chat_") or r["family"] == "injection"))
    inj = fam.get("injection", [0, 0])
    return {"total": common.cell(sum(v[1] for v in fam.values()), len(rows)),
            "hidden_instructions": common.cell(inj[1], inj[0]),
            "families": {f: common.cell(v[1], v[0]) for f, v in sorted(fam.items())},
            "unscored": sum(r.get("status") != "scored" for r in rows),
            "chat_p95_s_gpu": round(chat[int(0.95 * (len(chat) - 1))], 2) if chat else None}


def run_one(model, gpus, run_dir, est_hours, ttl):
    """One model, up to MAX_BOXES spot boxes in turn, resuming from the outbox each time."""
    from gpu_box import OllamaBox
    os.makedirs(run_dir, exist_ok=True)
    logp = os.path.join(run_dir, "run.log")
    meta_p = os.path.join(run_dir, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {"model": model, "boxes": []}
    meta["check_prompts"] = check_prompts()
    complete = False
    first = len(meta["boxes"])     # per invocation: at most MAX_BOXES boxes that came up, MAX_ATTEMPTS creates
    while not complete:
        mine = meta["boxes"][first:]
        if sum(bool(b.get("id")) for b in mine) >= MAX_BOXES or len(mine) >= MAX_ATTEMPTS:
            common.log(f"{model}: stopping after {len(mine)} creates ({sum(bool(b.get('id')) for b in mine)} came up); "
                       "a rerun resumes from the outbox", logp)
            break
        if mine and not mine[-1].get("id"):
            time.sleep(30)                              # a refused create is spot capacity; wait before the next
        left = sum(SET_SIZE[s] - len(done_ids(os.path.join(run_dir, f"{s}.jsonl"))) for s in SETS)
        hours = round(0.25 + est_hours * left / sum(SET_SIZE.values()), 3)
        rate_gpu = max(gpus, key=lambda g: common.GPU_RATE[g])
        rid = common.reserve(hours, rate_gpu)          # Refused propagates: no box is created
        common.log(f"{model}: reserved {hours} h at the {rate_gpu} rate (meter id {rid}); {left} tasks left", logp)
        t0 = time.time()
        rec = {"meter_id": rid, "reserved_hours": hours, "tasks_left_at_start": left}
        meta["boxes"].append(rec)
        try:
            with OllamaBox(gpus, "gl-" + common.stem(model), ttl_minutes=ttl, owner=common.OWNER, num_parallel=1, image=common.OLLAMA_IMAGE,
                           log=lambda m: common.log(m, logp), reservation=rid) as box:
                rec.update({k: str(v) for k, v in box.info.items() if k in ("id", "gpu_type", "spot", "nvidia_smi")})
                version = box.get("/api/version").get("version")
                if version != common.OLLAMA_PIN:   # a score on another runtime is another table
                    rec["outcome"] = f"runtime {version} is not the pinned {common.OLLAMA_PIN}: not scored"
                    common.log(f"{model}: {rec['outcome']}", logp)
                    break
                meta["ollama_version"] = version
                status, pull_s = box.pull(model)
                common.log(f"{model}: pull {status} in {pull_s}s", logp)
                if status != "success":
                    rec["outcome"] = f"pull failed: {status}"
                    if not box.evicted():
                        break                           # a pull the registry refuses is a result, not a retry
                    continue
                tags = box.get("/api/tags").get("models", [])
                m = next((x for x in tags if x.get("name") == model or x.get("model") == model), {})
                meta["digest"] = "sha256:" + m["digest"] if m.get("digest") else registry_digest(model)
                meta["digest_source"] = "the box's /api/tags" if m.get("digest") else "the registry (the box listed none)"
                meta["details"] = m.get("details")
                meta["size_bytes"] = m.get("size")
                w0 = time.time()
                box.post("/api/chat", {"model": model, "stream": False, "think": False,
                                       "messages": [{"role": "user", "content": "ok"}],
                                       "options": {"num_ctx": NUM_CTX, "num_predict": 4}}, timeout=900)
                common.log(f"{model}: warm in {time.time() - w0:.1f}s; digest {(meta['digest'] or 'none listed')[:19]}", logp)
                how = score_on_box(box, model, run_dir, logp, t0 + hours * 3600)   # never past what was reserved
                complete = outbox_complete(run_dir)
                rec["outcome"] = "complete" if complete else {"done": "tasks skipped", "lost": "box lost"}.get(how, how)
        except common.Refused:
            raise
        except Exception as e:  # noqa: BLE001 (a box reclaimed while it starts raises from create: an eviction)
            rec["outcome"] = f"box lost or never up: {type(e).__name__}: {str(e)[:200]}"
            common.log(f"{model}: {rec['outcome']}", logp)
        finally:
            real = round((time.time() - t0) / 3600, 4)
            rec["hours"] = real
            rec["settled_usd"] = common.settle(rid, real)
            common.write_json(meta_p, meta)
            common.log(f"{model}: box {len(meta['boxes'])} {rec.get('outcome')}, {real} h, settled ${rec['settled_usd']}", logp)
    rows = {s: outbox_rows(os.path.join(run_dir, f"{s}.jsonl")) for s in SETS}
    complete = outbox_complete(run_dir)
    result = {"model": model, "date": os.path.basename(run_dir)[:10], "source": "Ollama library build, pulled by the box",
              "digest": meta.get("digest"), "digest_source": meta.get("digest_source"), "details": meta.get("details"), "size_bytes": meta.get("size_bytes"),
              "ollama_version": meta.get("ollama_version"), "think": False, "num_ctx": NUM_CTX, "num_parallel": 1,
              "complete": complete,
              "check_prompts": meta["check_prompts"],
              "sets": {s: {"set_hash": SET_HASH[s], "rows": len(rows[s]), **summarize(rows[s])} for s in SETS if rows[s]},
              "boxes": [{k: b.get(k) for k in ("id", "gpu_type", "spot", "hours", "settled_usd", "outcome",
                                               "tasks_left_at_start")} for b in meta["boxes"]],
              "box_hours": round(sum(b.get("hours", 0) for b in meta["boxes"]), 3),
              "outbox": run_dir.replace(os.path.expanduser("~"), "~", 1)}
    out = os.path.join(common.RESULTS, f"{result['date']}-{common.stem(model)}.json")
    common.write_json(out, result)
    common.log(f"{model}: wrote {out}; complete {result['complete']}", logp)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma list of Ollama library tags")
    ap.add_argument("--one", default="", help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--gpu", default="H200,H100", help="first type with spot capacity wins")
    ap.add_argument("--est-hours", type=float, default=1.2, help="box hours per model for the swing estimate")
    ap.add_argument("--ttl", type=int, default=240, help="grux.max-minutes label for the sweeper")
    ap.add_argument("--date", default="", help="run date to resume (default today)")
    a = ap.parse_args()
    gpus = [g.strip() for g in a.gpu.split(",") if g.strip()]
    if a.one:
        try:
            r = run_one(a.one, gpus, a.run_dir, a.est_hours, a.ttl)
        except common.Refused as e:
            print(str(e), flush=True)
            sys.exit(3)
        sys.exit(0 if r["complete"] else 1)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if not models:
        ap.error("give --models")
    if not common.CHARTER:
        for m in models:
            iid = common.backlog("measure", f"score {m} on held-out v1 and v2 on the pinned Ollama runtime "
                                            f"{common.OLLAMA_PIN}")
            print(f"BACKLOGGED {m} as {iid}: no swing charter (GRUX_SWING_CHARTER), so no box of its own", flush=True)
        sys.exit(common.BACKLOGGED)
    rate = max(common.box_rate(g) for g in gpus)
    est = round(len(models) * a.est_hours * rate, 2)
    try:
        common.swing_guard(est, f"scoring {len(models)} model(s) at {a.est_hours} h each on {'/'.join(gpus)}")
    except common.Refused as e:
        print(str(e), flush=True)
        sys.exit(4)
    check_prompts()
    date = a.date or common.today()
    print(f"swing estimate ${est} for {models}; one box per model, in parallel", flush=True)
    procs = {}
    for m in models:
        run_dir = os.path.join(common.STATE, "runs", f"{date}-{common.stem(m)}")
        os.makedirs(run_dir, exist_ok=True)
        procs[m] = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--one", m, "--run-dir", run_dir,
                                     "--gpu", a.gpu, "--est-hours", str(a.est_hours), "--ttl", str(a.ttl)],
                                    stdout=open(os.path.join(run_dir, "stdout.log"), "a"), stderr=subprocess.STDOUT)
        print(f"started {m}: pid {procs[m].pid}, outbox {run_dir}", flush=True)
        time.sleep(10)                                  # stagger creates; spot capacity is contended
    codes = {m: p.wait() for m, p in procs.items()}
    print(json.dumps({"exit": codes}), flush=True)
    sys.exit(0 if all(c == 0 for c in codes.values()) else (3 if 3 in codes.values() else 1))


if __name__ == "__main__":
    main()
