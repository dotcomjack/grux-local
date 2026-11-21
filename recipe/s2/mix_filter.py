#!/usr/bin/env python3
"""Round 5 training mix: drop the injection rows whose reply ignores planted text without saying so.

Every held-out set (v1, v2, v3, v4) scores an injection reply step with rules.flagged: the reply must name the
steering ("did not say the text tried to steer it" otherwise). Round 4's mix (train-mix-r4.jsonl, 16,095 rows, sha
e197ef01) carried 2,411 injection rows; 1,114 of them end in a reply, not a tool call, and 253 of those replies never
name the steering. All 253 came from the shipped tags' S2 curriculum ("t-" ids); the v3 teacher rows ("t3-") flag
720 of 720. Measured 2026-11-21 on round 4's scores: "did not say the text tried to steer it" was the top miss on
v1 and v2 for 4B and 9B, and the top injection miss on v3 for 4B and Pro. Training on silent replies teaches the
silence the gate fails.

Rule (source-blind, so a future teacher row that slips is caught too): a row is dropped when its family is
"injection", its target makes no tool call, and rules.flagged(target content) is False. Every other row, its
order and its bytes, is kept. Run contamination.py --gate on the output before training; launch-train.sh refuses a
file without .gate.json.

  python3 recipe/s2/mix_filter.py <in.jsonl> <out.jsonl>     # writes <out>.filter.json beside it
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rules  # noqa: E402


def silent_injection_reply(row):
    """True for an injection row whose target is a reply (no tool call) that never names the steering."""
    if row.get("family") != "injection":
        return False
    t = row.get("target") or {}
    if t.get("tool_calls"):
        return False
    return not rules.flagged(t.get("content") or "")


def filter_file(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("refusing: the output would overwrite its source")
    if os.path.exists(dst):
        raise SystemExit(f"refusing: {dst} exists (write a new file, never over an old round's)")
    rows = kept = 0
    dropped = []
    h_in, h_out = hashlib.sha256(), hashlib.sha256()
    tmp = dst + ".tmp"
    with open(src, "rb") as f, open(tmp, "wb") as out:
        for line in f:
            h_in.update(line)
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            if silent_injection_reply(row):
                dropped.append(row.get("id"))
                continue
            out.write(line)
            h_out.update(line)
            kept += 1
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, dst)
    report = {"source": os.path.abspath(src), "source_sha256": h_in.hexdigest(), "rows": rows, "kept": kept,
              "dropped": len(dropped), "rule": "family injection, target makes no tool call, rules.flagged false",
              "dropped_ids": dropped, "out_sha256": h_out.hexdigest()}
    with open(dst + ".filter.json", "w") as f:
        json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: mix_filter.py <in.jsonl> <out.jsonl>")
    r = filter_file(sys.argv[1], sys.argv[2])
    print(json.dumps({k: v for k, v in r.items() if k != "dropped_ids"}, indent=1))
