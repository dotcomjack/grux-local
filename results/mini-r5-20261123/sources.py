"""Run on the Mac mini from ~/grux-eval: prints, for each raw file results/v1.json's 2B is built from, its sha256,
size and the date it was last written (US Eastern, the Mini's clock), so results/build_v1.py can verify the copies in
this folder and date each run without trusting a checkout's file times.

  python3 sources.py > SOURCES.json
"""
import datetime, hashlib, json, os
FILES = [f"runs/r5-published/{n}" for n in ("grux-local-2b-r5-holdout.jsonl", "grux-local-2b-r5-holdout.meta.json",
                                             "ps-fresh.json", "ps-after.json", "footprint-after.txt")]
for p in ("r5-pass1", "r5-pass2"):
    for m in ("grux-local-2b-r5pilot", "grux-local_2b"):
        for s in ("holdout", "holdout2", "local"):
            FILES += [f"runs/{p}/{m}-{s}.jsonl", f"runs/{p}/{m}-{s}.meta.json"]
for p in ("r5-pass3", "r5-8k"):
    for m in ("grux-local-2b-r5pilot", "grux-local_2b"):
        FILES += [f"runs/{p}/{m}-local.jsonl", f"runs/{p}/{m}-local.meta.json"]
for m in ("grux-local-2b-r5", "grux-local-2b-r5pilot"):
    FILES += [f"runs/r5-smoke/{m}-holdout2.jsonl", f"runs/r5-smoke/{m}-holdout2.meta.json"]
FILES += ["fname-ab/fn8k-grux-local-2b-r5pilot.txt", "fname-ab/fn8k-grux-local_2b.txt"]
out = {}
for f in FILES:
    b = open(f, "rb").read()
    out[f] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b),
              "written": datetime.datetime.fromtimestamp(os.path.getmtime(f)).isoformat(timespec="seconds")}
print(json.dumps(out, indent=1))
