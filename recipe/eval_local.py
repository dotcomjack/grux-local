#!/usr/bin/env python3
"""S1 on a real Mac: the same suite, sent one request at a time to this Mac's own Ollama, as a
person's Grux would. Records pass, latency and the Ollama runner's resident memory, so a pick is
proven on the hardware it is for, not only on a GPU.

  python3 eval_local.py --model qwen3.5:4b --run results/mac-<stamp> [--think false] [--per-family N] [--set holdout]

--set holdout scores S2's frozen held-out set (s2/holdout.py) instead of the S1 suite.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suite  # noqa: E402

BASE = "http://127.0.0.1:11434"


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
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--set", choices=["s1", "holdout"], default="s1")
    a = ap.parse_args()
    request, score, cases_fn, suffix = suite.request, suite.score, suite.cases, ""
    if a.set == "holdout":
        sys.path.insert(0, os.path.join(HERE, "s2"))
        import holdout
        request, score, cases_fn, suffix = holdout.request_holdout, holdout.score_holdout, holdout.cases, "-holdout"
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
            req = request(c, a.model)
            t = time.time()
            try:
                raw = post(req["path"], req["body"])
                ok, why, got = score(c, raw)
                row = {"case": c["id"], "family": c["family"], "status": "scored", "pass": ok, "why": why,
                       "got": got, "s": round(time.time() - t, 2),
                       "first_token_s": round((raw.get("prompt_eval_duration") or 0) / 1e9 + (raw.get("load_duration") or 0) / 1e9, 2),
                       "out_tokens": raw.get("eval_count")}
            except Exception as e:
                row = {"case": c["id"], "family": c["family"], "status": "skipped", "error": str(e)[:200],
                       "s": round(time.time() - t, 2)}
            out.write(json.dumps(row) + "\n")
            out.flush()
    meta["peak_runner_rss_gb"] = runner_rss_gb()
    meta["wall_s"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(a.run, stem + suffix + ".meta.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)


def post_get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return json.loads(resp.read().decode() or "{}")


if __name__ == "__main__":
    main()
