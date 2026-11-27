"""The num_batch sweep behind the 2B published on 2026-11-27, from the raw rows and the probe log in this repository.
Run from results/reproductions: python3 2026-11-27-2b-num-batch-sweep.py

Part 1 reads ../mini-nb256-20261127/probe/batch-extract.log: per num_batch, the first request (cold) and then the
repeats, each sharing a ~5,400-token system prompt with the one before, as llama-server logged them.
Part 2 scores 1622ac1b (num_batch 1024, its default), 256 (the published 86255f20) and 128 (never published) on the
sets each ran, case for case against 1622ac1b, from the eval rows."""
import collections, hashlib, json, re
N, R = "../mini-nb256-20261127", "../mini-r5-20261123"
for d in (N, R):   # every row read below is the copy SOURCES.json hashed on the Mac mini
    src = json.load(open(f"{d}/SOURCES.json"))
    for rel, v in src.items():
        assert hashlib.sha256(open(f"{d}/{rel}", "rb").read()).hexdigest() == v["sha256"], rel

print("Part 1: llama-server on the Mac mini, Ollama 0.40.8, num_ctx 8192, 1622ac1b's GGUF")
nb, rows = None, collections.defaultdict(list)
for line in open(f"{N}/probe/batch-extract.log"):
    if m := re.search(r"n_batch += (\d+)", line):
        nb = int(m.group(1))
    elif m := re.search(r"prompt eval time = +([\d.]+) ms / +(\d+) tokens", line):
        rows[nb].append((int(m.group(2)), float(m.group(1))))
for nb, xs in rows.items():
    cold, rest = xs[0], xs[1:]
    print(f"  num_batch {nb:>4}: cold {cold[0]} tokens in {cold[1]:.0f} ms; then evaluated again "
          f"{[t for t, _ in rest]} tokens in {[round(ms) for _, ms in rest]} ms")


def load(p):
    return {(r := json.loads(l))["case"]: r for l in open(p)}


def med(xs):
    xs = sorted(xs)
    return round((xs[len(xs) // 2] + xs[(len(xs) - 1) // 2]) / 2, 2)


print("Part 2: pass on each set, cases that differ from num_batch 1024, median first word over the tasks sent")
sets = [("held-out v1", f"{R}/runs/r5-published/grux-local-2b-r5-holdout.jsonl", f"{N}/runs/nb256/grux-local-2b-r5-nb256-holdout.jsonl", f"{N}/runs/nb128/grux-local-2b-r5-nb128-holdout.jsonl"),
        ("held-out v2", f"{R}/runs/r5-pass1/grux-local-2b-r5pilot-holdout2.jsonl", f"{N}/runs/nb256/grux-local-2b-r5-nb256-holdout2.jsonl", f"{N}/runs/nb128/grux-local-2b-r5-nb128-holdout2.jsonl"),
        ("S1 suite", f"{N}/runs/r5-pass1/grux-local-2b-r5pilot.jsonl", f"{N}/runs/nb256/grux-local-2b-r5-nb256.jsonl", f"{N}/runs/nb128/grux-local-2b-r5-nb128.jsonl"),
        ("local turns, num_ctx 8192", f"{R}/runs/r5-8k/grux-local-2b-r5pilot-local.jsonl", f"{N}/runs/nb256-8k/grux-local-2b-r5-nb256-local.jsonl", f"{N}/runs/nb128-8k/grux-local-2b-r5-nb128-local.jsonl"),
        ("local turns, num_ctx 16384", f"{R}/runs/r5-pass1/grux-local-2b-r5pilot-local.jsonl", f"{N}/runs/nb256-published/grux-local-2b-r5-nb256-local.jsonl", f"{N}/runs/nb128/grux-local-2b-r5-nb128-local.jsonl")]
for name, *paths in sets:
    base = load(paths[0])
    print(f"  {name}:")
    for nb, p in zip((1024, 256, 128), paths):
        x = load(p)
        assert set(x) == set(base)
        sent = [r for r in x.values() if r["status"] == "scored"]
        lost = sorted(k for k in x if base[k].get("pass") and not x[k].get("pass"))
        gained = sorted(k for k in x if x[k].get("pass") and not base[k].get("pass"))
        print(f"    num_batch {nb:>4}: {sum(bool(r['pass']) for r in sent)} of {len(sent)}, first word {med([r['first_token_s'] for r in sent]):.2f} s"
              + ("" if nb == 1024 else f", lost {lost or 'none'}, gained {gained or 'none'}"))
