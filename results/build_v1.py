"""Builds results/v1.json from the raw held-out eval files on the Mac mini (factory/gpu/results/mac-ship,
mac-base*, mac-grux-local-9b-mo and mbp-base9b) and from the MacBook Pro runs behind 4b-v2 (mbp-4b-v2, copied
beside them), so no number in it is typed by hand. Run on the Mini from that results folder:

  python3 build_v1.py > v1.json
  python3 build_v1.py --v1-tags-from v1.json > v1.new.json   # where only mbp-4b-v2 is at hand: 2b, 4b and 9b
                                                              # are taken from the last full build, unchanged

The 2B that replaced the first one (2026-11-24, manifest 1622ac1b) is built from the raw rows in this repository, in
mini-r5-20261123/ (copied from the Mac mini with their sha256 and write times in mini-r5-20261123/SOURCES.json,
which mini-r5-20261123/sources.py printed there). From a clone, in results/:

  python3 build_v1.py --from v1.json --2b-r5 mini-r5-20261123 > v1.new.json && mv v1.new.json v1.json

It takes every other tag, the top-level fields and the other loaded_after_use entries from v1.json unchanged
(asserted), moves the 2B it replaces into "replaced" with its after-use reading, and is a fixed point: run on its own
output it writes the same bytes.
"""
import collections
import datetime
import json
import os
import re
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


def summary(path, measured=None, scored_only=False):
    """measured: the date the run was written, when the file's own mtime is not it (a copy in a git clone)."""
    rows = load(path)
    sent = [r for r in rows if r.get("status") == "scored"]
    if scored_only:   # the "local" set: a turn Grux 3.3.1 answers without a model is "not_sent", never scored
        assert all(r.get("status") in ("scored", "not_sent") for r in rows), "rows neither scored nor not sent"
    else:
        assert len(sent) == len(rows), "unscored rows"
    fam = fams(sent)
    chat = [r["s"] for r in sent if r["family"].startswith("chat_") or r["family"] == "injection"]
    out = {"measured": measured or datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat(),
           "pass": sum(v["pass"] for v in fam.values()), "of": len(sent),
           "hidden_instructions_ignored": fam["injection"]["pass"], "hidden_instruction_cases": fam["injection"]["of"],
           "chat_p95_s": p95(chat), "families": fam}
    if scored_only:
        out["not_sent"] = len(rows) - len(sent)
    return out


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
R5_DIGEST = "sha256:1622ac1b80afb8841fc588fbc57eff60284f318d021311322a84df4aa097251e"
R5_SCORED = "sha256:03ac1e39c81e3451e557662f0cb68d6917b0a0e4743b713b326228bb7d50270c"
R5_RUNTIME = "Ollama 0.40.8, llama.cpp runner (llama-server), GGUF Q4_K_M, one request at a time, thinking off as Grux sends it"


def r5_2b(d, replaced, v2_task_set):
    """The 2B built from the round 5 pilot: the scored build grux-local-2b-r5pilot (manifest 03ac1e39) with the
    shipped license layer appended (results/reproductions/2026-11-24-2b-v2-license.md), published as
    gruxai.com/grux/grux-local:2b, manifest 1622ac1b. Held-out v1 and the loaded sizes are read from the published
    bytes (r5-published); held-out v2, the "local" set and file naming from the scored build, beside the 2B it
    replaces scored on the same Mac mini the same evening. `replaced` is that 2B's v1.json record: its
    before-training run is this one's too (the same qwen3.5:2b on the same Mac)."""
    import hashlib
    src = json.load(open(f"{d}/SOURCES.json"))

    def path(rel):   # a raw file, checked against the hash recorded on the Mac mini
        p = f"{d}/{rel}"
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == src[rel]["sha256"], p
        return p

    def day(rel):
        return src[rel]["written"][:10]

    def run(rel, **kw):
        return summary(path(rel), measured=day(rel), **kw)

    def meta(rel):
        m = json.load(open(path(rel)))
        assert m["version"] == "0.40.8" and m["think"] == "false", m
        return m

    pub = "runs/r5-published/grux-local-2b-r5-holdout"
    assert (meta(pub + ".meta.json")["model"], meta(pub + ".meta.json")["num_ctx"]) == ("grux-local-2b-r5", 16384)
    ps = {}
    for k in ("fresh", "after"):   # eval_local's meta reads 0.0 GB here (its runner lookup misses llama-server)
        rel = f"runs/r5-published/ps-{k}.json"
        (m,) = json.load(open(path(rel)))["models"]
        assert "sha256:" + m["digest"] == R5_DIGEST and m["context_length"] == 16384, m
        assert (m["details"]["format"], m["details"]["quantization_level"], m.get("runner")) == ("gguf", "Q4_K_M", "llamacpp")
        ps[k] = round(m["size"] / 1e9, 2)
    # What macOS's footprint tool read for the llama-server process after those 464 tasks: more than Ollama reports.
    fp = open(path("runs/r5-published/footprint-after.txt")).read()
    assert fp.lstrip("= \n").startswith("llama-server ["), fp[:120]
    footprint_mb = int(re.search(r"phys_footprint: (\d+) MB", fp).group(1))
    show = json.load(open(f"{d}/show.json"))
    assert show["model"] == "grux-local-2b-r5" and show["details"]["runner"] == "llamacpp", show
    assert "vision" not in show["capabilities"], show["capabilities"]

    def pair(p, s, **kw):
        new, old = f"runs/{p}/grux-local-2b-r5pilot-{s}", f"runs/{p}/grux-local_2b-{s}"
        mn, mo = meta(new + ".meta.json"), meta(old + ".meta.json")
        assert (mn["model"], mo["model"]) == ("grux-local-2b-r5pilot", "grux-local:2b") and mn["num_ctx"] == mo["num_ctx"]
        return mn["num_ctx"], run(new + ".jsonl", **kw), run(old + ".jsonl", **kw)

    _, v1_pilot, v1_old = pair("r5-pass1", "holdout")
    _, v2_new, v2_old = pair("r5-pass1", "holdout2")
    # The second pass of both models on held-out v1 and v2 gives the same totals as the first.
    for s, a, b in (("holdout", v1_pilot, v1_old), ("holdout2", v2_new, v2_old)):
        _, a2, b2 = pair("r5-pass2", s)
        assert (a2["pass"], b2["pass"], a2["hidden_instructions_ignored"], b2["hidden_instructions_ignored"]) == \
            (a["pass"], b["pass"], a["hidden_instructions_ignored"], b["hidden_instructions_ignored"]), s
    local = []
    for p in ("r5-pass1", "r5-pass2", "r5-pass3", "r5-8k"):
        ctx, a, b = pair(p, "local", scored_only=True)
        local.append({"num_ctx": ctx, "grux_local": a, "previous_2b_same_machine": b})
    # The 7 chat-honesty turns the new build failed in the last local run (the same 7 in all four), by reason.
    last = [r for r in load(path("runs/r5-8k/grux-local-2b-r5pilot-local.jsonl"))
            if r["family"] == "chat_honesty" and r["status"] == "scored" and not r["pass"]]
    for p in ("r5-pass1", "r5-pass2", "r5-pass3"):
        assert {r["case"] for r in load(path(f"runs/{p}/grux-local-2b-r5pilot-local.jsonl")) if r["family"] == "chat_honesty"
                and r["status"] == "scored" and not r["pass"]} == {r["case"] for r in last}, p
    honesty_fails = dict(sorted(collections.Counter(r["why"].split(":")[0] for r in last).items()))

    def ttft(rel):   # median seconds to the first token, over every task sent
        xs = sorted(r["first_token_s"] for r in load(path(rel)) if r.get("status") == "scored")
        return round((xs[len(xs) // 2] + xs[(len(xs) - 1) // 2]) / 2, 2)

    first_token = {s: {"grux_local": ttft(n), "previous_2b_same_machine": ttft(o)} for s, n, o in (
        ("holdout_v1", pub + ".jsonl", "runs/r5-pass1/grux-local_2b-holdout.jsonl"),
        ("holdout_v2", "runs/r5-pass1/grux-local-2b-r5pilot-holdout2.jsonl", "runs/r5-pass1/grux-local_2b-holdout2.jsonl"),
        ("local_num_ctx_8192", "runs/r5-8k/grux-local-2b-r5pilot-local.jsonl", "runs/r5-8k/grux-local_2b-local.jsonl"))}
    naming = {}
    for k, name in (("grux_local", "grux-local-2b-r5pilot"), ("previous_2b_same_machine", "grux-local_2b")):
        line = json.loads(open(path(f"fname-ab/fn8k-{name}.txt")).readline())
        assert line["model"] == name.replace("_", ":"), line
        naming[k] = {"measured": day(f"fname-ab/fn8k-{name}.txt"), "pass": line["useful"], "of": line["of"]}
    # recipe/s2/contamination.py's TrainGate on round 5's rows (against held-out v3 and v4 only), its report as written
    # with the source path cut to the file name.
    gate = json.load(open(f"{d}/train-mix-r5.gate.json"))
    assert gate["source"] == "train-mix-r5.jsonl" and gate["dropped"] == 0 and gate["kept"] == gate["rows"], gate
    grux_local = run(pub + ".jsonl")
    return {"name": "gruxai.com/grux/grux-local:2b", "aliases": [], "size": "2B", "made_for": "8 GB Macs",
            "base_model": "Qwen/Qwen3.5-2B", "manifest_digest": R5_DIGEST, "loaded_gb": ps["fresh"],
            # Its own runtime: the top-level one is the MLX runner, which every other tag runs on.
            "runtime": R5_RUNTIME,
            "input": "text only",
            "capabilities": show["capabilities"],
            "text_only_note": "Ollama lists no vision capability for it, and a chat request that carries an image "
                              "answers HTTP 400 \"Multimodal data provided, but model does not support multimodal "
                              "requests.\" The 2B it replaced accepted images.",
            "scored_as": f"grux-local-2b-r5pilot, manifest {R5_SCORED}: the published manifest less its appended "
                         "license layer (results/reproductions/2026-11-24-2b-v2-license.md)",
            "grux_local": grux_local,
            "runner_footprint_after_use_mb": footprint_mb,
            "runner_footprint_note": "the llama-server process's phys_footprint, as macOS's footprint tool printed it "
                                     "after the 464 held-out v1 tasks on the published bytes; Ollama's /api/ps size "
                                     "(loaded_gb, loaded_after_use) leaves part of it out",
            "before_training": replaced["before_training"],
            "before_training_model": replaced["before_training_model"],
            "before_training_machine": replaced["before_training_machine"],
            # Held-out v1 on the published bytes against the scored build, case for case.
            "holdout_v1_scored_build_flips": flips(path(pub + ".jsonl"), path("runs/r5-pass1/grux-local-2b-r5pilot-holdout.jsonl")),
            "previous_2b_same_machine": v1_old,
            "previous_2b_same_machine_model": f"grux-local:2b, the 2B this one replaced, manifest "
                                              f"{replaced['manifest_digest']}, scored on the same Mac mini the evening "
                                              "before",
            # The same set the 4b-v2 was scored on, in the same words: a reader of v1.json finds it described once.
            "holdout_v2": {"task_set": v2_task_set, "grux_local": v2_new, "previous_2b_same_machine": v2_old},
            "local_turns": {"task_set": "held-out v2's 265 chat turns sent the way Grux 3.3.1 sends a local turn "
                                        "(recipe/eval_local.py --set local); a turn Grux answers without a model is "
                                        "not sent, so each run scores the rest. Four runs of each model, the last "
                                        "at num_ctx 8,192, what Grux asks for on an 8 GB Mac",
                            "runs": local,
                            "chat_honesty_failures_last_run": honesty_fails},
            "first_token_median_s": first_token,
            "training_gate": {"rows": gate["rows"], "checked_against": ["held-out v3", "held-out v4"],
                              "held_out_touched": len(gate["held_out_touched"]),
                              "note": "no check of round 5's rows against held-out v1 or v2 is in this repository"},
            "file_naming": {"task_set": "Grux 3.3.1's own file-naming prompt on held-out file naming's 92 text "
                                        "items, at num_ctx 8,192; a name counts when it carries the item's key words",
                            **naming}}


if sys.argv[1:2] == ["--from"]:
    # Every tag but the 2B, and everything else, from the last v1.json: the raw files behind them are not in
    # this repository (they are on the Mac mini and the MacBook Pro, see above).
    v1 = json.load(open(sys.argv[2]))
    assert sys.argv[3] == "--2b-r5", sys.argv
    old = v1["tags"]["2b"]
    if old["manifest_digest"] == R5_DIGEST:   # run on its own output
        (replaced,) = [r for r in v1["replaced"] if r["tag"] == "2b"]
    else:
        (used,) = [u for u in v1["loaded_after_use"] if u["tag"] == "2b"]
        replaced = {"tag": "2b", "replaced_by": R5_DIGEST, **old, "runtime": v1["runtime"], "input": "text and images",
                    "loaded_after_use": {k: v for k, v in used.items() if k != "tag"}}
    new = r5_2b(sys.argv[4], replaced, v1["tags"]["4b-v2"]["holdout_v2"]["task_set"])
    assert new["before_training"] == replaced["before_training"]
    out = dict(v1)
    out["runtime"] = "Ollama 0.40.8, MLX runner, one request at a time, thinking off as Grux sends it, except for a " \
                     "tag with its own runtime field (the 2b, on the llama.cpp runner)"
    out["loaded_after_use"] = [{"tag": "2b", "machine": MAC_MINI, "num_ctx": 16384, "fresh_gb": new["loaded_gb"],
                                "after_gb": round(json.load(open(f"{sys.argv[4]}/runs/r5-published/ps-after.json"))
                                                  ["models"][0]["size"] / 1e9, 2),
                                "after_tasks": new["grux_local"]["of"]} if u["tag"] == "2b" else u
                               for u in v1["loaded_after_use"]]
    out["tags"] = {k: (new if k == "2b" else v) for k, v in v1["tags"].items()}
    out["replaced"] = [replaced] + [r for r in v1.get("replaced", []) if r["tag"] != "2b"]
    # What must not change: every other tag, every other after-use reading, every other top-level field.
    for k, v in v1.items():
        if k not in ("runtime", "loaded_after_use", "tags", "replaced"):
            assert out[k] == v, k
    assert {k: v for k, v in out["tags"].items() if k != "2b"} == {k: v for k, v in v1["tags"].items() if k != "2b"}
    assert [u for u in out["loaded_after_use"] if u["tag"] != "2b"] == [u for u in v1["loaded_after_use"] if u["tag"] != "2b"]
    print(json.dumps(out, indent=1))
    sys.exit(0)
if sys.argv[1:2] == ["--v1-tags-from"]:
    tags = {k: v for k, v in json.load(open(sys.argv[2]))["tags"].items() if k in ("2b", "4b", "9b")}
    assert list(tags) == ["2b", "4b", "9b"], list(tags)
else:
    if "--first-2b" not in sys.argv:   # v1_tags() builds the 2B that was replaced on 2026-11-24
        sys.exit("refused: a full build writes the first 2B (601cd23c) as the 2b tag. Run it with --first-2b, then "
                 "`build_v1.py --from <that output> --2b-r5 mini-r5-20261123` to put the current 2B back")
    tags = v1_tags()
tags["4b-v2"] = tag_4b_v2()


def after_use():
    """What Ollama reports for a model (its /api/ps size) after it has worked, beside the fresh load each tag's
    loaded_gb records: the published 2B and 4B after a full held-out run on the Mac mini (mac-m5-step1, the
    realism-gap run: the second run's start reads the size the first left), and the 4B v2 after 72 held-out tasks on
    the MacBook Pro (mbp-4b-v2/after-use: those tasks, then the Ollama /api/ps reading taken after them). All at
    num_ctx 16384. The runner's resident memory did not grow in any of them."""
    out = []
    for tag in ("2b", "4b"):
        fresh = json.load(open(f"mac-m5-step1/grux-local_{tag}-holdout.meta.json"))
        then = json.load(open(f"mac-m5-step1/grux-local_{tag}-holdout-realistic.meta.json"))
        assert fresh["model"] == then["model"] == f"grux-local:{tag}" and fresh["load_s"] > 1 > then["load_s"], (fresh, then)
        assert fresh["num_ctx"] == then["num_ctx"] == 16384 and fresh["ollama_size_gb"] == tags[tag]["loaded_gb"]
        out.append({"tag": tag, "machine": MAC_MINI, "num_ctx": 16384, "fresh_gb": fresh["ollama_size_gb"],
                    "after_gb": then["ollama_size_gb"],
                    "after_tasks": len(load(f"mac-m5-step1/grux-local_{tag}-holdout.jsonl"))})
    ps = [m for m in json.load(open("mbp-4b-v2/after-use/ps.json"))["models"]
          if m["digest"] == tags["4b-v2"]["manifest_digest"].split(":")[1]]
    assert len(ps) == 1 and ps[0]["context_length"] == 16384, ps
    out.append({"tag": "4b-v2", "machine": MACBOOK_PRO, "num_ctx": 16384, "fresh_gb": tags["4b-v2"]["loaded_gb"],
                "after_gb": round(ps[0]["size"] / 1e9, 2),
                "after_tasks": len(load("mbp-4b-v2/after-use/grux-local_4b-v2-holdout.jsonl"))})
    return out


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
                   "so a model loads more there. Ollama can report more memory for the model as it works (its "
                   "/api/ps size; the runner's resident memory did not grow): loaded_after_use.",
    "loaded_after_use": after_use(),
    "tags": tags}, indent=1))
