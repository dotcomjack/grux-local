#!/usr/bin/env python3
"""S1 on a real Mac: the same suite, sent one request at a time to this Mac's own Ollama, as a
person's Grux would. Records pass, latency and the Ollama runner's resident memory, so a pick is
proven on the hardware it is for, not only on a GPU.

  python3 eval_local.py --model qwen3.5:4b --run results/mac-<stamp> [--think false] [--per-family N]
                        [--set holdout|holdout2|holdout_v3]

--set holdout scores S2's frozen held-out set (s2/holdout.py) instead of the S1 suite; --set holdout2 scores
held-out version 2 (s2/holdout_v2.py), whose chat-shaped cases are sent the way a Grux 3.1.0 local turn sends
them (local_turn.py) and scored against the tools that turn offered. --set holdout_v3 scores held-out version 3
(s2/holdout_v3.py): the full Grux turn with every tool at its pinned tag (GRUX_SRC_V3), each job sent step by
step with the scripted results of the steps before it; num_ctx defaults to that set's own window.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suite  # noqa: E402

BASE = "http://127.0.0.1:11434"

# The run stops itself before it can take the Mac down (test_eval_local_guard.py). Measured 2026-11-14 on the
# Mini, 16 GB with 4 GB of disk free: held-out v3 at 65,536 ran it out of swap, macOS killed system services, the
# login session reset and every LaunchAgent stopped until a password was typed.
MEM_FREE_PCT_MIN = 10      # memory_pressure's own free percentage
PRESSURE_CRITICAL = 4      # kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
DISK_FREE_GB_MIN = 8       # room for macOS to grow swap into; the Mini had 4.2
GUARD_EVERY_S = 5


def parse_swap_free_mb(text):
    m = re.search(r"free = ([\d.]+)([MG])", text or "")
    if not m:
        return None
    return round(float(m.group(1)) * (1024 if m.group(2) == "G" else 1), 2)


def parse_free_pct(text):
    m = re.search(r"memory free percentage: (\d+)%", text or "")
    return int(m.group(1)) if m else None


def mem_verdict(pressure_level, free_pct):
    """None when the Mac can take more, else why it cannot. An unreadable number never trips alone."""
    if pressure_level is not None and pressure_level >= PRESSURE_CRITICAL:
        return f"kernel memory pressure critical (level {pressure_level})"
    if free_pct is not None and free_pct < MEM_FREE_PCT_MIN:
        return f"memory free {free_pct}% under the {MEM_FREE_PCT_MIN}% floor"
    return None


def disk_verdict(free_gb):
    if free_gb is not None and free_gb < DISK_FREE_GB_MIN:
        return f"disk free {free_gb} GB under {DISK_FREE_GB_MIN} GB: no room to swap into"
    return None


def _sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def read_mem():
    """(kernel pressure level, free percentage), each None where this OS has no reader."""
    level = _sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level").strip()
    return (int(level) if level.isdigit() else None), parse_free_pct(_sh("memory_pressure"))


def guard_tick(read, unload, abort_path, exit=os._exit):
    """One look. On a breach: unload the model (frees the memory at once), write why, stop the run with exit 7."""
    level, free_pct = read()
    why = mem_verdict(level, free_pct)
    if why is None:
        return
    try:
        unload()
    finally:
        json.dump({"why": why, "pressure_level": level, "free_pct": free_pct, "at": time.time(),
                   "swap_free_mb": parse_swap_free_mb(_sh("sysctl", "vm.swapusage"))}, open(abort_path, "w"))
        print("ABORT " + why, file=sys.stderr, flush=True)
        exit(7)


def start_guard(model, abort_path):
    def unload():
        post("/api/generate", {"model": model, "keep_alive": 0}, timeout=60)

    def loop():
        while True:
            guard_tick(read_mem, unload, abort_path)
            time.sleep(GUARD_EVERY_S)
    threading.Thread(target=loop, daemon=True).start()


def post(path, body, timeout=600):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def runner_rss_gb():
    """Resident memory of the Ollama model runner processes, from ps (KB)."""
    out = subprocess.run(["ps", "-axo", "rss=,command="], capture_output=True, text=True).stdout
    kb = sum(int(l.split(None, 1)[0]) for l in out.splitlines()
             if "ollama" in l and " runner" in l and l.strip().split(None, 1)[0].isdigit())
    return round(kb / 1048576, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--think", choices=["unset", "false"], default="false")
    ap.add_argument("--per-family", type=int, default=0, help="first N cases per job (0 = all)")
    ap.add_argument("--num-ctx", type=int, default=None, help="16384, or held-out v3's own window for holdout_v3")
    ap.add_argument("--set", choices=["s1", "holdout", "holdout2", "holdout_v3"], default="s1")
    a = ap.parse_args()
    request, score, cases_fn, suffix = suite.request, suite.score, suite.cases, ""
    if a.set == "holdout":
        sys.path.insert(0, os.path.join(HERE, "s2"))
        import holdout
        request, score, cases_fn, suffix = holdout.request_holdout, holdout.score_holdout, holdout.cases, "-holdout"
    if a.set == "holdout2":
        sys.path.insert(0, os.path.join(HERE, "s2"))
        import holdout_v2
        request, score, cases_fn, suffix = holdout_v2.request_v2, holdout_v2.score_v2, holdout_v2.cases, "-holdout2"
    v3 = None
    if a.set == "holdout_v3":
        sys.path.insert(0, os.path.join(HERE, "s2"))
        import holdout_v3 as v3
        cases_fn, suffix = v3.cases, "-holdout_v3"
        if a.num_ctx is not None:
            v3.NUM_CTX = a.num_ctx
        a.num_ctx = v3.NUM_CTX
    if a.num_ctx is None:
        a.num_ctx = 16384
    if a.think == "false":
        suite.THINK = False
    suite.NUM_CTX = a.num_ctx
    os.makedirs(a.run, exist_ok=True)
    cases = cases_fn()
    if a.per_family:
        seen = {}
        cases = [c for c in cases if seen.setdefault(c["family"], []).append(1) or len(seen[c["family"]]) <= a.per_family]
    stem = a.model.replace(":", "_").replace("/", "_")   # a host-qualified name holds slashes
    path = os.path.join(a.run, stem + suffix + ".jsonl")
    have = set()
    if os.path.exists(path):
        have = {json.loads(l)["case"] for l in open(path)}
    refuse = disk_verdict(round(shutil.disk_usage(a.run).free / 1e9, 1)) or mem_verdict(*read_mem())
    if refuse:
        sys.exit("REFUSED " + refuse)
    start_guard(a.model, os.path.join(a.run, stem + suffix + ".abort.json"))
    t0 = time.time()
    warm = post("/api/chat", {"model": a.model, "stream": False, "think": False,
                              "messages": [{"role": "user", "content": "ok"}],
                              "options": {"num_ctx": a.num_ctx, "num_predict": 4}})
    ps = post_get("/api/ps")
    m = next((x for x in ps.get("models", []) if x.get("name") == a.model), {})
    meta = {"model": a.model, "num_ctx": a.num_ctx, "think": a.think, "load_s": round(time.time() - t0, 1),
            "ollama_size_gb": round(m.get("size", 0) / 1e9, 2), "runner_rss_gb": runner_rss_gb(),
            "version": post_get("/api/version").get("version")}
    print(json.dumps(meta), flush=True)
    with open(path, "a") as out:
        for c in cases:
            if c["id"] in have:
                continue
            if v3 is not None:
                out.write(json.dumps(score_v3_row(v3, c, a.model)) + "\n")
                out.flush()
                continue
            req = request(c, a.model)
            t = time.time()
            try:
                raw = post(req["path"], req["body"])
                ok, why, got = score(c, raw, req["offered"]) if a.set == "holdout2" and "offered" in req \
                    else score(c, raw)
                row = {"case": c["id"], "family": c["family"], "status": "scored", "pass": ok, "why": why,
                       "got": got, "s": round(time.time() - t, 2),
                       "first_token_s": round((raw.get("prompt_eval_duration") or 0) / 1e9 + (raw.get("load_duration") or 0) / 1e9, 2),
                       "out_tokens": raw.get("eval_count")}
                if "offered" in req:
                    row["offered"] = req["offered"]
                    row["prompt_tokens"] = raw.get("prompt_eval_count")
            except Exception as e:
                row = {"case": c["id"], "family": c["family"], "status": "skipped", "error": str(e)[:200],
                       "s": round(time.time() - t, 2)}
            out.write(json.dumps(row) + "\n")
            out.flush()
    meta["peak_runner_rss_gb"] = runner_rss_gb()
    meta["wall_s"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(a.run, stem + suffix + ".meta.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)


def score_v3_row(v3, c, model):
    """One held-out v3 job, step by step (holdout_v3.run_case), as one row: pass, why, and each step's seconds and
    prompt tokens, so a prompt Ollama trimmed shows as a prompt_tokens number below the request's size."""
    t = time.time()
    try:
        ok, why, got, steps = v3.run_case(c, model, post)
    except Exception as e:
        return {"case": c["id"], "family": c["family"], "status": "skipped", "error": str(e)[:200],
                "s": round(time.time() - t, 2)}
    return {"case": c["id"], "family": c["family"], "status": "scored", "pass": ok, "why": why, "got": got,
            "s": round(time.time() - t, 2), "steps": steps, "prompt_tokens": [x["prompt_tokens"] for x in steps],
            "out_tokens": sum(x["out_tokens"] or 0 for x in steps)}


def post_get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return json.loads(resp.read().decode() or "{}")


if __name__ == "__main__":
    main()
