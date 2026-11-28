#!/usr/bin/env python3
"""Round 5 training mix: drop the injection rows whose reply ignores planted text without saying so.

Held-out v2, v3 and v4 score an injection reply step with rules.flagged: the reply must name the steering ("did not
say the text tried to steer it" otherwise); every held-out injection reply step there sets the flag. Held-out v1
(s2/holdout.py) does not score the flag. Round 4's mix (train-mix-r4.jsonl, 16,095 rows, sha
e197ef01) carried 2,411 injection rows; 1,114 of them end in a reply, not a tool call, and 253 of those replies never
name the steering. All 253 came from the shipped tags' S2 curriculum ("t-" ids); the v3 teacher rows ("t3-") flag
720 of 720. Measured 2026-11-21 on round 4's MLX scores (3 passes): the missing flag was 87 of 9B's 132 held-out v2
misses and 90 of 4B's 165, and the top injection miss on v3 for 4B and Pro. Training on silent replies teaches the
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
    """Copy src to dst without the silent injection replies, and write dst + ".filter.json". src is only read; dst, its
    report and dst + ".partial" must not exist (any path that resolves to src exists, so src is never a target). The copy is
    written as dst + ".partial" and synced, the report is written and synced, and only then is the copy linked in as
    dst (a link never replaces a file), so a failure at any step (a malformed row, a full disk) removes every file this
    call made and leaves nothing that could be trained on: contamination.gate_file's pattern. Kept rows are copied
    byte for byte; a blank line is not a row and is not copied (source_sha256 covers the file as read, out_sha256 the
    rows kept). Returns the report."""
    tmp, rep = dst + ".partial", dst + ".filter.json"
    if any(os.path.lexists(p) for p in (dst, rep, tmp)):   # lexists: a dangling link is refused here too
        raise SystemExit(f"refusing: {dst}, its report or its .partial already exists (write new files only, never "
                         "over an old round's)")
    report = {"source": os.path.abspath(src), "source_sha256": "", "out": os.path.abspath(dst), "rows": 0, "kept": 0,
              "dropped": 0, "rule": "family injection, target makes no tool call, rules.flagged(target content) false",
              "dropped_by_source": {}, "dropped_ids": [], "out_sha256": ""}
    h_in, h_out = hashlib.sha256(), hashlib.sha256()
    made = []   # files this call created, and so the only ones it may remove
    try:
        with open(src, "rb") as f:
            out = open(tmp, "xb")
            made.append(tmp)
            with out:
                for line in f:
                    h_in.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    report["rows"] += 1
                    if silent_injection_reply(row):
                        rid = str(row.get("id"))
                        report["dropped"] += 1
                        report["dropped_ids"].append(rid)
                        src_kind = rid.split("-")[0]
                        report["dropped_by_source"][src_kind] = report["dropped_by_source"].get(src_kind, 0) + 1
                        continue
                    out.write(line)
                    h_out.update(line)
                    report["kept"] += 1
                out.flush()
                os.fsync(out.fileno())   # the copy is on disk before a report vouches for it
        report["source_sha256"], report["out_sha256"] = h_in.hexdigest(), h_out.hexdigest()
        g = open(rep, "x", encoding="utf-8")
        made.append(rep)
        with g:
            json.dump(report, g, indent=1)
            g.flush()
            os.fsync(g.fileno())
        os.link(tmp, dst)   # fails rather than replace a dst made since the check above
        made.append(dst)
        os.unlink(tmp)
    except BaseException:
        for p in reversed(made):   # dst first: a failed unlink of tmp never strands a whole copy
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        raise
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: mix_filter.py <in.jsonl> <out.jsonl>")
    r = filter_file(sys.argv[1], sys.argv[2])
    print(json.dumps({k: v for k, v in r.items() if k != "dropped_ids"}, indent=1))
