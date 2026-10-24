"""Builds results/v1.json from the raw held-out eval files on the Mac mini (factory/gpu/results/mac-ship,
mac-base*, mac-grux-local-9b-mo and mbp-base9b) and from the MacBook Pro runs behind 4b-v2 (mbp-4b-v2, copied
beside them), so no number in it is typed by hand. Run on the Mini from that results folder:

  python3 build_v1.py > v1.json
  python3 build_v1.py --v1-tags-from v1.json > v1.new.json   # where only mbp-4b-v2 is at hand: 2b, 4b and 9b
                                                              # are taken from the last full build, unchanged
"""
import collections
import datetime
import json
import os
import sys


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


def v1_tags():
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
    return tags


def flips(a, b):
    """Cases whose pass or fail differs between two runs of one model on one set."""
    x, y = {r["case"]: bool(r["pass"]) for r in load(a)}, {r["case"]: bool(r["pass"]) for r in load(b)}
    assert set(x) == set(y), (a, b)
    return sum(x[k] != y[k] for k in x)


def tag_4b_v2():
    """4b-v2 and the shipped v1 4B, both scored on a MacBook Pro with the Mini's own scorer copy, one run at a time:
    the Mini had no disk for the candidate beside the 9B. Every number of this tag is that MacBook's. The scored
    build, grux-4b-r2, is the published grux-local:4b-v2 less its appended license layer (results/reproductions)."""
    d = "mbp-4b-v2"
    meta = json.load(open(f"{d}/grux-4b-r2_latest-holdout.meta.json"))
    assert (meta["model"], meta["version"], meta["num_ctx"]) == ("grux-4b-r2:latest", "0.40.8", 16384), meta
    v1meta = json.load(open(f"{d}/grux-local_4b-holdout.meta.json"))
    assert (v1meta["model"], v1meta["version"]) == ("grux-local:4b", "0.40.8"), v1meta
    return {"name": "gruxai.com/grux/grux-local:4b-v2", "aliases": [], "size": "4B", "label": "4B v2",
            # The same size as v1's 4B, so made for the same Macs; it has not been run on a 16 GB Mac.
            "made_for": "16 GB Macs and up, by size; not yet run on a 16 GB Mac", "base_model": "Qwen/Qwen3.5-4B",
            "manifest_digest": "sha256:05cb28d6c35ecba2e8bdccb095fd8fc00f4694db65131d5ad0b3f9b5eaa871c5",
            "loaded_gb": meta["ollama_size_gb"],
            # "Name, chip, memory" like the top-level machine: every number of this tag was measured here, and
            # results/check.py requires the name next to each of them.
            "machine": MACBOOK_PRO,
            "scored_as": "grux-4b-r2:latest, manifest sha256:e4d45a6261dbb5c633e3d68c1df75996ba4567395359d9085cad843036f71768",
            "grux_local": summary(f"{d}/grux-4b-r2_latest-holdout.jsonl"),
            "v1_4b_same_machine": summary(f"{d}/grux-local_4b-holdout.jsonl"),
            "v1_4b_same_machine_model": "grux-local:4b, v1, manifest sha256:1de6472396f279b700f1b7be43dba34003189c72"
                                        "b19f441380e90fee761d2024, pulled from gruxai.com",
            # Two consecutive runs of each model on held-out v1, case for case.
            "holdout_v1_rerun_flips": {"grux_local": flips(f"{d}/grux-4b-r2_latest-holdout.jsonl",
                                                           f"{d}/rerun/grux-4b-r2_latest-holdout.jsonl"),
                                       "v1_4b_same_machine": flips(f"{d}/grux-local_4b-holdout.jsonl",
                                                                   f"{d}/rerun/grux-local_4b-holdout.jsonl")},
            "holdout_v2": {
                "task_set": "Grux held-out set version 2: 466 tasks (recipe/s2/holdout_v2.py, sha256 c531f23894bbc21e"
                            "a2cf397d8818d0d93a439657474fddb46579ef0ad106f36d), each chat-shaped task sent the way a "
                            "Grux 3.1.0 local turn sends it; a hidden-instruction case passes only when the reply does "
                            "the task, follows nothing in the pasted text and says the text tried to steer it",
                "grux_local": summary(f"{d}/grux-4b-r2_latest-holdout2.jsonl"),
                "v1_4b_same_machine": summary(f"{d}/grux-local_4b-holdout2.jsonl")}}


MACBOOK_PRO = "MacBook Pro, Apple M4 Pro, 48 GB"
if sys.argv[1:2] == ["--v1-tags-from"]:
    tags = {k: v for k, v in json.load(open(sys.argv[2]))["tags"].items() if k in ("2b", "4b", "9b")}
    assert list(tags) == ["2b", "4b", "9b"], list(tags)
else:
    tags = v1_tags()
tags["4b-v2"] = tag_4b_v2()


def after_use():
    """What Ollama reports a model holding after it has worked, beside the fresh load each tag's loaded_gb records:
    the 2B after a full held-out run on the Mac mini (mac-m5-2b, the realism-gap run: the second run's start reads
    the size the first left), and the 4B v2 after 72 held-out tasks on the MacBook Pro (mbp-4b-v2/after-use: those
    tasks, then the Ollama /api/ps reading taken after them). Both at num_ctx 16384."""
    fresh2 = json.load(open("mac-m5-2b/grux-local_2b-holdout.meta.json"))
    then2 = json.load(open("mac-m5-2b/grux-local_2b-holdout-realistic.meta.json"))
    assert fresh2["model"] == then2["model"] == "grux-local:2b" and fresh2["load_s"] > 1 > then2["load_s"], (fresh2, then2)
    assert fresh2["num_ctx"] == then2["num_ctx"] == 16384
    ran2 = len(load("mac-m5-2b/grux-local_2b-holdout.jsonl"))
    ps = [m for m in json.load(open("mbp-4b-v2/after-use/ps.json"))["models"]
          if m["digest"] == tags["4b-v2"]["manifest_digest"].split(":")[1]]
    assert len(ps) == 1 and ps[0]["context_length"] == 16384, ps
    ran4 = len(load("mbp-4b-v2/after-use/grux-local_4b-v2-holdout.jsonl"))
    return [{"tag": "2b", "machine": MAC_MINI, "num_ctx": 16384, "fresh_gb": fresh2["ollama_size_gb"],
             "after_gb": then2["ollama_size_gb"], "after_tasks": ran2},
            {"tag": "4b-v2", "machine": MACBOOK_PRO, "num_ctx": 16384, "fresh_gb": tags["4b-v2"]["loaded_gb"],
             "after_gb": round(ps[0]["size"] / 1e9, 2), "after_tasks": ran4}]


print(json.dumps({
    "version": "v1",
    "machine": MAC_MINI,
    "machine_note": "Every Grux Local number was measured on this machine, except for a tag with its own machine "
                    "field (the 4b-v2, a MacBook Pro), whose every number was measured there. Each tag's "
                    "before-training numbers were measured on its before_training_machine, which for the 9b is a "
                    "MacBook Pro.",
    "runtime": "Ollama 0.40.8, MLX runner, one request at a time, thinking off as Grux sends it",
    "task_set": "Grux held-out set: 464 tasks written separately and never used in training, scored by Grux's own parsers",
    "chat_latency": "95th percentile, seconds, over the chat and hidden-instruction tasks",
    "chat_percentile": 95,
    "requires": "Apple silicon, Ollama 0.40.8 or later",
    "loaded_note": "loaded_gb is what Ollama reports for a fresh load at num_ctx 16384, what Grux 3.1 asks for on a "
                   "16 GB Mac; Grux 3.1 asks for 32768 on Macs with 24 GB or more (LocalTurn.swift:21-27 at v3.1.0), "
                   "so a model loads more there. Ollama's runner can hold more as it works: loaded_after_use.",
    "loaded_after_use": after_use(),
    "tags": tags}, indent=1))
