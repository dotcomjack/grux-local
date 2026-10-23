#!/usr/bin/env python3
"""Prove the scorer sends exactly the requests the v1 numbers were measured with.

  GRUX_SRC=... python3 recipe/check_prompts.py

Builds the request body for every one of the 464 held-out cases from the Grux source at GRUX_SRC and
hashes them. v1 (results/v1.json) was scored with requests that hash to EXPECTED; any change to a prompt
in the source, to the cases or to the request shape changes the hash. Exit 0 on a match, 1 otherwise.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "s2"))
import holdout  # noqa: E402

EXPECTED = "7c3f2a9ff3316ecc"   # measured 2026-10-23 on the tree that scored v1, and on public Grux v3.1.0

cases = holdout.build() if hasattr(holdout, "build") else holdout.cases()
h = hashlib.sha256()
for c in cases:
    h.update(json.dumps(holdout.request_holdout(c, "M"), sort_keys=True).encode())
got = h.hexdigest()[:16]
print(f"{len(cases)} held-out requests, hash {got}, expected {EXPECTED}: {'MATCH' if got == EXPECTED else 'DIFFERENT'}")
sys.exit(0 if got == EXPECTED and len(cases) == 464 else 1)
