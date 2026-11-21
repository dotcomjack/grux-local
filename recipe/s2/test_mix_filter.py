#!/usr/bin/env python3
"""mix_filter drops exactly the injection replies that never name the steering, and nothing else. No model runs.

  1. an injection reply that ignores planted text silently is dropped;
  2. an injection reply that says the text tried to steer it is kept;
  3. an injection row that ends in a tool call is kept (a call step is not scored for a flag);
  4. a reply in any other family is kept even when it does not flag;
  5. kept rows keep their order and their exact bytes; the report counts rows, kept and dropped and names the
     dropped ids and the output's sha; the output never overwrites its source (a symlink to it included) or an existing file;
  6. the rows have train-mix-r4's real shape (flat {"name", "arguments"} calls, a system and user turn before the
     target); a silent target after an assistant turn that flagged is still dropped (only the target is trained and
     scored); a None content is a silent reply, not a crash;
  7. a malformed row mid-file leaves no output, no report and no .partial behind.
Prints PASS or FAIL per check; exit 1 on any FAIL, or if no check ran.

  python3 recipe/s2/test_mix_filter.py
"""
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import mix_filter  # noqa: E402

results = []


def check(name, ok):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name)


def row(i, family, content="", calls=None, history=None):
    """train-mix-r4's shape: system and user turns (and any earlier assistant turns) in messages, the trained reply in
    target, tool calls flat {"name", "arguments"}."""
    msgs = [{"role": "system", "content": "You are Grux."},
            {"role": "user", "content": "CONTEXT FOR THIS TURN (written by Grux, not typed by the user): ..."}]
    return {"id": i, "family": family, "messages": msgs + (history or []),
            "tools": [{"type": "function", "function": {"name": "add_task", "parameters": {}}}],
            "target": {"content": content, "tool_calls": calls or []}}


SILENT = row("t-injection-1", "injection", "Here is the translation: the lunch is on Wednesday.")
FLAGGED = row("t3-injection-2", "injection", "Lunch is Wednesday. The email also tried to get me to forward your "
                                             "contacts, and I ignored that.")
CALL = row("t3-injection-3", "injection", "", [{"name": "add_task", "arguments": {"title": "a"}}])
AFTER_FLAG = row("t3-injection-6", "injection", "Done, the task is added.", history=[
    {"role": "assistant", "content": "The page tried to get me to email your contacts; I ignored that.",
     "tool_calls": []},
    {"role": "tool", "content": "ok"}])
NONE = row("t-injection-7", "injection")
NONE["target"]["content"] = None
OTHER = row("t-dictation-4", "dictation", "Meet me at noon.")
SILENT_T3 = row("t3-injection-5", "injection", "The recipe needs three bananas.")

check("1. a silent injection reply is dropped", mix_filter.silent_injection_reply(SILENT))
check("1b. a silent teacher reply is dropped too (the rule is source-blind)",
      mix_filter.silent_injection_reply(SILENT_T3))
check("2. a flagged injection reply is kept", not mix_filter.silent_injection_reply(FLAGGED))
check("3. an injection tool call is kept", not mix_filter.silent_injection_reply(CALL))
check("4. a reply in another family is kept", not mix_filter.silent_injection_reply(OTHER))
check("6. a silent target after a flagged assistant turn is dropped", mix_filter.silent_injection_reply(AFTER_FLAG))
check("6b. a None content is a silent reply, not a crash", mix_filter.silent_injection_reply(NONE))

with tempfile.TemporaryDirectory() as d:
    src, dst = os.path.join(d, "in.jsonl"), os.path.join(d, "out.jsonl")
    order = [OTHER, SILENT, FLAGGED, CALL, SILENT_T3]
    lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in order]
    with open(src, "w") as f:
        f.writelines(lines)
        f.write("\n")   # a blank line is not a row
    rep = mix_filter.filter_file(src, dst)
    out = open(dst).readlines()
    check("5. kept rows keep their order and exact bytes", out == [lines[0], lines[2], lines[3]])
    check("5b. the report counts rows, kept and dropped",
          (rep["rows"], rep["kept"], rep["dropped"]) == (5, 3, 2))
    check("5c. the report names the dropped ids", rep["dropped_ids"] == ["t-injection-1", "t3-injection-5"])
    check("5c2. out_sha256 is the sha of the output as written",
          rep["out_sha256"] == hashlib.sha256(open(dst, "rb").read()).hexdigest())
    check("5d. the report is written beside the output",
          json.load(open(dst + ".filter.json"))["kept"] == 3)
    for target, why in ((src, "its source"), (dst, "an existing file")):
        try:
            mix_filter.filter_file(src, target)
            check(f"5e. refuses to write over {why}", False)
        except SystemExit:
            check(f"5e. refuses to write over {why}", True)
    check("5f. the source is untouched", open(src).readlines()[:5] == lines)
    link = os.path.join(d, "link.jsonl")
    os.symlink(src, link)
    try:
        mix_filter.filter_file(src, link)
        check("5g. a symlink to the source is refused (it exists)", False)
    except SystemExit:
        check("5g. a symlink to the source is refused (it exists)", True)
    bad, bout = os.path.join(d, "bad.jsonl"), os.path.join(d, "bad-out.jsonl")
    with open(bad, "w") as f:
        f.write(lines[0] + "{not json\n" + lines[1])
    try:
        mix_filter.filter_file(bad, bout)
        check("7. a malformed row raises", False)
    except ValueError:
        check("7. a malformed row raises", True)
    check("7b. and leaves no output, report or .partial",
          not any(os.path.exists(p) for p in (bout, bout + ".filter.json", bout + ".partial")))

if not results:
    print("FAIL no check ran")
    sys.exit(1)
sys.exit(0 if all(results) else 1)
