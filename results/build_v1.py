"""Builds results/v1.json from the raw held-out eval files on the Mac mini (factory/gpu/results/mac-ship and
mac-base*), so no number in it is typed by hand. Run on the Mini from that results folder."""
import collections
import datetime
import json
import os


def load(p):
    return [json.loads(l) for l in open(p)]


def p95(xs):
    xs = sorted(xs)
    return round(xs[int(0.95 * (len(xs) - 1))], 2)   # the same index rule as factory/gpu/s2's reports


def fams(rows):
    f = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        f[r["family"]][0] += 1
        f[r["family"]][1] += bool(r["pass"])
    return {k: {"pass": v[1], "of": v[0]} for k, v in sorted(f.items())}


def summary(path):
    rows = load(path)
    assert all(r.get("status") == "scored" for r in rows), "unscored rows"
    fam = fams(rows)
    chat = [r["s"] for r in rows if r["family"].startswith("chat_") or r["family"] == "injection"]
    return {"measured": datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat(),
            "pass": sum(v["pass"] for v in fam.values()), "of": len(rows),
            "hidden_instructions_ignored": fam["injection"]["pass"], "hidden_instruction_cases": fam["injection"]["of"],
            "chat_p95_s": p95(chat), "families": fam}


tags = {}
for tag, size, made_for, aliases, digest in [
        ("2b", "2B", "8 GB Macs", [], "sha256:601cd23c66ef4c834dec0c3f518c3663f2abe2298f62249b7337eb0845fb3ee4"),
        ("4b", "4B", "16 GB Macs and up", ["latest"], "sha256:1de6472396f279b700f1b7be43dba34003189c72b19f441380e90fee761d2024")]:
    meta = json.load(open(f"mac-ship/grux-local_{tag}-holdout.meta.json"))
    base_dir = "mac-base" if tag == "4b" else "mac-base2b"
    tags[tag] = {"name": f"gruxai.com/grux/grux-local:{tag}", "aliases": aliases, "size": size, "made_for": made_for,
                 "base_model": f"Qwen/Qwen3.5-{size}", "manifest_digest": digest, "loaded_gb": meta["ollama_size_gb"],
                 "grux_local": summary(f"mac-ship/grux-local_{tag}-holdout.jsonl"),
                 "before_training": summary(f"{base_dir}/qwen3.5_{tag}-holdout.jsonl"),
                 "before_training_model": f"qwen3.5:{tag}, the Ollama library build, on the same Mac"}
print(json.dumps({
    "version": "v1",
    "machine": "Mac mini, Apple M4, 16 GB",
    "runtime": "Ollama 0.40.8, MLX runner, one request at a time, thinking off as Grux sends it",
    "task_set": "Grux held-out set: 464 tasks written separately and never used in training, scored by Grux's own parsers",
    "chat_latency": "95th percentile, seconds, over the chat and hidden-instruction tasks",
    "chat_percentile": 95,
    "requires": "Apple silicon, Ollama 0.40.8 or later",
    "tags": tags}, indent=1))
