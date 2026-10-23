"""Builds results/v1.json from the raw held-out eval files on the Mac mini (factory/gpu/results/mac-ship,
mac-base*, mac-grux-local-9b-mo and mbp-base9b), so no number in it is typed by hand. Run on the Mini from that
results folder."""
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


MAC_MINI = "Mac mini, Apple M4, 16 GB"
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
                 "before_training_model": f"qwen3.5:{tag}, the Ollama library build, on the same Mac",
                 "before_training_machine": MAC_MINI}
# The 9B: scored on the same Mac mini, but its before-training baseline on a MacBook Pro, because the Mini had no
# disk to spare for the stock 9B. Its holdout meta reads 0.0 GB (read after the model unloaded), so the loaded size
# comes from loaded.json, read while it was loaded. The weights are digest-identical to the published grux-local:9b.
loaded = json.load(open("mac-grux-local-9b-mo/loaded.json"))["models"]
assert [m["name"] for m in loaded] == ["grux-local-9b-mo:latest"], loaded
tags["9b"] = {"name": "gruxai.com/grux/grux-local:9b", "aliases": [], "size": "9B", "made_for": "24 GB Macs and up",
              "base_model": "Qwen/Qwen3.5-9B",
              "manifest_digest": "sha256:357e4bd45ff0030a6dbfdf5a7650b745418f1dcdb7eeed473b4078987f111837",
              "loaded_gb": loaded[0]["size_gb"],
              "grux_local": summary("mac-grux-local-9b-mo/grux-local-9b-mo-holdout.jsonl"),
              "before_training": summary("mbp-base9b/qwen3.5_9b-holdout.jsonl"),
              "before_training_model": "qwen3.5:9b, the Ollama library build, on a MacBook Pro (Apple M4 Pro, 48 GB) "
                                       "with Ollama 0.40.8, because the Mac mini had no disk to spare for it",
              # "Name, chip, memory" like the top-level machine: results/check.py compares the two, and when they
              # differ it requires the name next to every one of this tag's before-training numbers.
              "before_training_machine": "MacBook Pro, Apple M4 Pro, 48 GB"}
# The published 9B scored on that same MacBook Pro, once it is there: a same-machine pair for the baseline above.
pair = "mbp-grux9b/grux-local_9b-holdout.jsonl"
if os.path.exists(pair):
    tags["9b"]["grux_local_on_before_training_machine"] = summary(pair)
print(json.dumps({
    "version": "v1",
    "machine": MAC_MINI,
    "machine_note": "Every Grux Local number was measured on this machine. Each tag's before-training numbers were "
                    "measured on its before_training_machine, which for the 9b is a MacBook Pro.",
    "runtime": "Ollama 0.40.8, MLX runner, one request at a time, thinking off as Grux sends it",
    "task_set": "Grux held-out set: 464 tasks written separately and never used in training, scored by Grux's own parsers",
    "chat_latency": "95th percentile, seconds, over the chat and hidden-instruction tasks",
    "chat_percentile": 95,
    "requires": "Apple silicon, Ollama 0.40.8 or later",
    "tags": tags}, indent=1))
