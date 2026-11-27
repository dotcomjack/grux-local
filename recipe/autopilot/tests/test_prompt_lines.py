#!/usr/bin/env python3
"""score.prompt_lines_ok accepts check_prompts.py's real outputs and refuses a mismatch or a missing set. No model runs.

  python3 recipe/autopilot/tests/test_prompt_lines.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import score  # noqa: E402

V1 = "held-out v1: 464 requests, hash 7c3f2a9ff3316ecc, expected 7c3f2a9ff3316ecc: MATCH"
V2 = "held-out v2: 466 requests, hash da6577c7178a0963, expected da6577c7178a0963: MATCH"
V3 = "held-out v3: 321 jobs at Grux v3.3.1, hash fd125548504ade65, expected fd125548504ade65: MATCH"
V4 = "held-out v4: 323 jobs at Grux v3.3.1, hash 00a87f51a5f63f19, expected 00a87f51a5f63f19: MATCH"
NOTE = 'held-out v3 and v4: not checked; export GRUX_SRC_V3="$(GRUX_TAG=v3.3.1 sh recipe/get_grux_source.sh)"'
results = []


def check(name, ok):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name)


check("1. v1, v2 and the not-checked note (GRUX_SRC_V3 unset) pass", score.prompt_lines_ok([V1, V2, NOTE]))
check("2. v1 to v4 all MATCH (GRUX_SRC_V3 set) pass", score.prompt_lines_ok([V1, V2, V3, V4]))
check("3. the old two-line output still passes", score.prompt_lines_ok([V1, V2]))
check("4. a v2 MISMATCH refuses", not score.prompt_lines_ok([V1, V2.replace(": MATCH", ": MISMATCH"), NOTE]))
check("5. a v3 MISMATCH refuses", not score.prompt_lines_ok([V1, V2, V3.replace(": MATCH", ": MISMATCH"), V4]))
check("6. a missing v2 refuses", not score.prompt_lines_ok([V1, NOTE]))
check("7. a stray line refuses", not score.prompt_lines_ok([V1, V2, "Traceback (most recent call last):"]))
check("8. a duplicated v1 refuses", not score.prompt_lines_ok([V1, V1, V2]))
if not results:
    print("FAIL no check ran")
    sys.exit(1)
sys.exit(0 if all(results) else 1)
