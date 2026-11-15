#!/usr/bin/env python3
"""contamination.py checks training text against every frozen held-out set the charter names: v1, v2 and v3.
A training case that copies an 8-word run from any of them must be excluded. No box, no model.

  GRUX_SRC_V3=<Grux v3.3.1 Sources/Grux> python3 recipe/s2/test_contamination.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import contamination as ct  # noqa: E402
import holdout  # noqa: E402
import holdout_v2  # noqa: E402
import holdout_v3  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name)
    fails += 0 if ok else 1


held = ct.held_out()
runs = set().union(*[ct.grams(w) for w in held])


def caught(case):
    return bool(ct.grams(ct.texts(case)) & runs)


for name, mod in (("v1", holdout), ("v2", holdout_v2)):
    cs = [c for c in mod.cases() if len(ct.texts(c)) >= 8]
    hit = sum(caught({"id": "train-x", **{k: v for k, v in c.items() if k not in ct.SKIP}}) for c in cs)
    check(f"a training case copying a held-out {name} case is excluded ({hit} of {len(cs)})", hit == len(cs))
v3 = [t for c in holdout_v3.cases() for t in holdout_v3.case_texts(c) if len(t.split()) >= 8]
hit = sum(caught({"id": "train-x", "user": t}) for t in v3)
check(f"a training case copying held-out v3 text is excluded ({hit} of {len(v3)} texts)", hit == len(v3))
check("an unrelated sentence is not excluded",
      not caught({"id": "train-y", "user": "please water the ferns on the north balcony before the frost arrives tonight"}))

print("SELF-TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
