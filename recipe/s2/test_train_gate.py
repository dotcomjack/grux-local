#!/usr/bin/env python3
"""The next round's training gate (contamination.TrainGate) keeps held-out v4, and v3's and v4's sentence frames,
out of new training rows, and leaves the frozen v3 bank alone. No model runs.

  1. the frozen v3 bank keeps its kept set and its sha, and adding held-out v4 to held_out() would exclude no case of
     it (the gate runs at export time instead, so the frozen set never moves);
  2. the gate catches, in a rendered training row: a held-out v4 8-word run in a tool result, a v4 ask copied whole,
     a v4 ask and a v3 ask with every noun changed (a frame echo, which the word scores miss), a short v4 line held
     inside a longer line; a Grux runtime line alone is not counted, and it does not shield a v4 run beside it; a
     clean row passes. The frame checks go red with the frame rule turned off;
  3. export_v3 drops a job that touches a held-out line and counts it, and keeps a clean one.
Prints PASS or FAIL per check; exit 1 on any FAIL, or if no check ran.

  GRUX_SRC_V3=<Grux v3.3.1 Sources/Grux> python3 recipe/s2/test_train_gate.py
"""
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
try:
    import daytona  # noqa: F401
except ImportError:   # export_v3 reaches eval_run's imports through train_prompt; no box is made here
    import types
    sys.modules["daytona"] = types.SimpleNamespace(**{n: object for n in (
        "CreateSandboxFromImageParams", "Daytona", "GpuType", "Image", "ListSandboxesQuery", "Resources",
        "SessionExecuteRequest")})
import contamination as ct  # noqa: E402
import export_v3  # noqa: E402
import holdout_v3 as h  # noqa: E402
import holdout_v4 as v4  # noqa: E402
import train_bank_v3 as tb  # noqa: E402
import train_prompt  # noqa: E402

KEPT_SHA = "f147ad0bc079ed1967998403b342b3716be18d75b607f7f72f01234ec797b289"   # the v3 bank's kept set on main
FAILS, RAN = [], []


def check(name, ok, detail=""):
    RAN.append(name)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


# 1. the frozen bank does not move
ALL = tb.cases()
KEPT = tb.kept(ALL)
check(f"1 the v3 bank keeps {len(KEPT)} cases with the kept sha it had before the gate", tb.case_sha(KEPT) == KEPT_SHA,
      tb.case_sha(KEPT))
before = set(ct.v3_excluded(ALL))
held = ct.held_out
v4_words = [ct._words(t) for c in v4.cases() for t in h.case_texts(c)]
ct.held_out = lambda: held() + [w for w in v4_words if len(w) >= 3]
after = set(ct.v3_excluded(ALL))
ct.held_out = held
check(f"1 held-out v4 in held_out() would exclude no more of the bank ({len(after - before)} more)", after == before,
      str(sorted(after - before)[:3]))

# 2. the gate, on rendered rows
GATE = ct.TrainGate()
NO_FRAMES = ct.TrainGate(frames_against=())
SYSTEM = {"role": "system", "content": "You are Jax."}


def row(ask, tool_text=None, target="Okay."):
    msgs = [SYSTEM, {"role": "user", "content": "PER TURN BLOCK\n\nTHE USER'S MESSAGE:\n" + ask}]
    if tool_text is not None:
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "fs_read",
                                                                                   "arguments": {"path": "~/a.txt"}}}]},
                 {"role": "tool", "content": tool_text}]
    return {"id": "r", "messages": msgs, "target": {"content": target, "tool_calls": []}}


def why(r, gate=GATE):
    return gate.why(*ct.row_items(r))


CS = v4.cases()
long_text = next(s["result"] for c in CS for s in c["steps"] if "call" in s and len(s["result"].split()) >= 14
                 and c["family"] == "injection")
v4_run = " ".join(long_text.split()[:9])
got = why(row("Can you read this file for me?", "notes\n" + v4_run))
check("2 a held-out v4 8-word run in a row's tool result is caught, naming its v4 job",
      bool(got) and got[1].startswith("h4-"), str(got))
ask = next(c["user"] for c in CS if c["family"] == "tool_choice" and len(c["user"].split()) >= 6)
got = why(row(ask))
check("2 a held-out v4 ask copied whole is caught", bool(got) and got[1].startswith("h4-"), str(got))


def every_noun(text):
    """text with every word that is not a frame word, an article or a day changed, and every name too."""
    keep = ct._DAYS_RE.fullmatch
    t = re.sub(r"(?<=[A-Za-z0-9,;)'’] )[A-Z][a-z]+", lambda m: m.group(0) if keep(m.group(0)) else "Quorwick", text)
    t = re.sub(r"\d", "7", t)
    return re.sub(r"\b[a-z][a-z']+\b", lambda m: m.group(0) if m.group(0) in ct.FRAME_WORDS
                  or m.group(0) in ("a", "an", "the") or keep(m.group(0)) else "quorble", t)


v4_ask = next(c["user"] for c in CS if c["family"] == "chain" and ct._fixed(ct.frame(c["user"])) >= 7)
v3_ask = "Read my document Garden plan and put the planting day on my calendar."   # v3's field trip ask, new nouns
for label, text, who in (("held-out v4", every_noun(v4_ask), "h4-"), ("held-out v3", v3_ask, "h3-")):
    got, off = why(row(text)), why(row(text), NO_FRAMES)
    check(f"2 a {label} ask with every noun changed is caught as a frame echo, and is missed with frames off",
          bool(got) and got[1].startswith(who) and "frame" in got[0] and off is None, f"{text!r}: {got}, off {off}")
short = next((cid, line) for line, cid in GATE.owner.items()
             if 6 <= len(ct.skeleton(line)) < ct.SHORT and len(ct.grams3(ct.skeleton(line))) >= ct.REVERSE_GRAMS
             and not line.startswith(("{", "-")) and "\n" not in line)
inside = "Before anything else this morning, and I mean it, " + short[1].rstrip(".?!") + ", thanks a lot for that."
got = why(row(inside))
check("2 a short v4 line held inside a longer row line is caught", bool(got) and got[1] == short[0],
      f"{inside!r}: {got}")
got = why(row("Can you read this file for me?", h.PENDING))
check("2 a Grux runtime line alone (the Approvals pending line) is not counted", got is None, str(got))
got = why(row("Can you read this file for me?", h.PENDING + "\n" + v4_run))
check("2 ... and it does not shield a held-out v4 run on the next line", bool(got), str(got))
clean = row("What's a good name for a goldfish that likes to hide?", target="Maybe Bashful, or Shadow.")
check("2 a clean row passes", why(clean) is None, str(why(clean)))

# 3. export_v3 drops a job that touches a held-out line, and keeps a clean one
flagged = next(c for c in KEPT if c["family"] == "chain" and GATE.why(*ct.case_items(c)))
fine = next(c for c in KEPT if c["family"] == "chain" and not GATE.why(*ct.case_items(c)) and len(c["steps"]) > 1)


def answer(c):
    raws = [{"content": "", "tool_calls": [{"name": s["call"], "arguments": s["args"]}]} if "call" in s else
            {"content": h.perfect_reply(s["reply"]), "tool_calls": []} for s in c["steps"]]
    return {"case": c["id"], "family": c["family"], "status": "scored", "pass": True, "raw": raws,
            "job_sha": h.case_sha([c]), "tools_sha": train_prompt.tools_sha(c)}


run = tempfile.mkdtemp()
with open(os.path.join(run, "t-train_v3-pass1.jsonl"), "w") as f:
    for c in (flagged, fine):
        f.write(json.dumps(answer(c)) + "\n")
s = export_v3.export(run, os.path.join(run, "train.jsonl"))
ids = {json.loads(line)["id"].split("#")[0] for line in open(os.path.join(run, "train.jsonl"))}
reason = GATE.why(*ct.case_items(flagged))
check(f"3 export_v3 drops the job that touches {reason[1]} and keeps the clean one, and says so",
      flagged["id"] not in ids and fine["id"] in ids and s["dropped_touches_held_out"] == 1
      and s["held_out_touched"] == {reason[1]: 1}, json.dumps(s)[:300])

print(f"{'PASS' if not FAILS and RAN else 'FAIL'}: {len(RAN)} checks, {len(FAILS)} failed")
sys.exit(1 if FAILS or not RAN else 0)
