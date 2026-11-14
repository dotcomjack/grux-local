#!/usr/bin/env python3
"""Prove the scorer sends exactly the requests the published numbers were measured with.

  GRUX_SRC=... python3 recipe/check_prompts.py

Builds the request body for every held-out case from the Grux source at GRUX_SRC and hashes them, for both
sets. Held-out v1 (464 cases, s2/holdout.py) scored every v1 tag; held-out v2 (466 cases, s2/holdout_v2.py,
sent the way a Grux 3.1.0 local turn sends them) scored 4b-v2. Any change to a prompt or a tool in the source,
to the cases or to the request shape changes a hash. Exit 0 when both match, 1 otherwise.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "s2"))
import holdout  # noqa: E402
import holdout_v2  # noqa: E402

EXPECTED = "7c3f2a9ff3316ecc"      # measured 2026-10-23 on the tree that scored v1, and on public Grux v3.1.0
EXPECTED_V2 = "da6577c7178a0963"   # measured 2026-10-24 on the tree that scored 4b-v2, and on public Grux v3.1.0
EXPECTED_V3 = "fd125548504ade65"   # every step's request of held-out v3, measured 2026-11-14 on public Grux v3.3.1

cases = holdout.build() if hasattr(holdout, "build") else holdout.cases()
h = hashlib.sha256()
for c in cases:
    h.update(json.dumps(holdout.request_holdout(c, "M"), sort_keys=True).encode())
got = h.hexdigest()[:16]
ok = got == EXPECTED and len(cases) == 464
print(f"held-out v1: {len(cases)} requests, hash {got}, expected {EXPECTED}: {'MATCH' if ok else 'DIFFERENT'}")

cases2 = holdout_v2.cases()
got2 = holdout_v2.fingerprint(cases2)[:16]   # the set's own fingerprint: every case with its request body
ok2 = got2 == EXPECTED_V2 and len(cases2) == 466
print(f"held-out v2: {len(cases2)} requests, hash {got2}, expected {EXPECTED_V2}: {'MATCH' if ok2 else 'DIFFERENT'}")

# Held-out v3 reads its own tag (holdout_v3.GRUX_TAG) from GRUX_SRC_V3; it has no published number yet, so without
# that tree it is reported and not counted.
ok3 = True
if os.environ.get("GRUX_SRC_V3"):
    import holdout_v3  # noqa: E402
    cases3 = holdout_v3.cases()
    got3 = holdout_v3.fingerprint(cases3)[:16]
    ok3 = got3 == EXPECTED_V3 and len(cases3) == 321
    print(f"held-out v3: {len(cases3)} jobs at Grux {holdout_v3.GRUX_TAG}, hash {got3}, expected {EXPECTED_V3}: "
          f"{'MATCH' if ok3 else 'DIFFERENT'}")
else:
    print('held-out v3: not checked; export GRUX_SRC_V3="$(GRUX_TAG=v3.3.1 sh recipe/get_grux_source.sh)"')
sys.exit(0 if ok and ok2 and ok3 else 1)
