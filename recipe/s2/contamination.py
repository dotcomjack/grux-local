#!/usr/bin/env python3
"""Training cases that touch the frozen held-out set, by id only (the orchestrator never reads either
side's text). A training case is excluded when its user-facing text equals a held-out case's, or shares
any 8-word run with one. Writes s2/train_exclude.json; gen_data.py skips those ids.

  python3 s2/contamination.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import holdout  # noqa: E402
import train_bank  # noqa: E402

SKIP = {"id", "family", "now", "tool", "args", "expect_keys"}


def texts(c):
    out = []
    for k, v in c.items():
        if k in SKIP:
            continue
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out += [x for x in v if isinstance(x, str)]
    return re.findall(r"[a-z0-9']+", " ".join(out).lower())


def grams(ws, n=8):
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def main():
    held = [texts(c) for c in holdout.cases()]
    whole = {" ".join(w) for w in held}
    runs = set().union(*[grams(w) for w in held])
    out = sorted(c["id"] for c in train_bank.cases()
                 if " ".join(texts(c)) in whole or grams(texts(c)) & runs)
    json.dump({"rule": "exact text or any shared 8-word run with s2/holdout.py (sha 1623e0e7)",
               "excluded": out}, open(os.path.join(HERE, "train_exclude.json"), "w"), indent=1)
    print(f"EXCLUDED {len(out)} training cases")


if __name__ == "__main__":
    main()
