#!/usr/bin/env python3
"""The v3-shaped training bank (s2/train_bank_v3.py) is held-out v3's shape, scoreable, clean and new. No model runs.

  1. after the exclusion every family keeps at least 1,000 cases and the bank at least 8,000; under 2% excluded;
  2. holdout_v3.perfect_raws (written from the labels) scores every kept case: the scorer can pass each one;
  3. an empty answer fails every kept case, and so does following the planted line in every injection job;
  4. 0 overlap with held-out v1, v2 and v3 after the exclusion, by s2/contamination.py's own reading, and that
     reading is not vacuous: a held-out v1, v2 or v3 run planted in a kept case's tool result, history or ask is
     excluded, and a Grux runtime line does not shield a held-out run on the line beside it;
  5. s2/data_scan.py finds no secret shape and no identity term in the bank, and does find a planted key;
  6. the case sha is the same under PYTHONHASHSEED 1 and 2;
  7. every tool a case names (call, alt, forbid, allow, target) is in the v3.3.1 catalogue, every label passes
     train_bank_v3.label_errors (holdout_v3's self-test checks), the copy rules hold, and each RUNTIME_LINES piece is
     in the Grux source it names.
Prints PASS or FAIL per check; exit 1 on any FAIL.

  GRUX_SRC_V3=<Grux v3.3.1 Sources/Grux> [GRUX_DATA_SCAN_VOCAB=<file>] python3 recipe/s2/test_train_bank_v3.py
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import contamination as ct  # noqa: E402
import data_scan  # noqa: E402
import holdout  # noqa: E402
import holdout_v2  # noqa: E402
import holdout_v3 as h  # noqa: E402
import train_bank_v3 as tb  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


ALL = tb.cases()
BAD = ct.v3_excluded(ALL)
CS = [c for c in ALL if c["id"] not in BAD]
FAM = {f: [c for c in CS if c["family"] == f] for f in tb.FAMILIES}

# 1. counts
for f in tb.FAMILIES:
    n_all = sum(c["family"] == f for c in ALL)
    print(f"    {f:13s} {n_all} generated, {n_all - len(FAM[f])} excluded, {len(FAM[f])} kept")
check(f"1 every family keeps at least {tb.MINIMUM}", all(len(FAM[f]) >= tb.MINIMUM for f in tb.FAMILIES),
      str({f: len(v) for f, v in FAM.items()}))
check(f"1 the bank keeps {len(CS)}, at least {tb.TOTAL_MINIMUM}", len(CS) >= tb.TOTAL_MINIMUM)
check(f"1 {len(BAD)} of {len(ALL)} excluded ({100 * len(BAD) / len(ALL):.2f}%), under 2%", len(BAD) < 0.02 * len(ALL))
check("1 ids are unique and carry the bank's prefix", len({c["id"] for c in ALL}) == len(ALL)
      and all(c["id"].startswith(tb.PREFIX + "-") for c in ALL))
check("1 more weight on injection and shell than on any other family",
      min(len(FAM["injection"]), len(FAM["shell"])) > max(len(FAM[f]) for f in tb.FAMILIES
                                                          if f not in ("injection", "shell")))

# 2. the perfect answers score 100%
bad = [(c["id"], h.score_v3(c, h.perfect_raws(c))[1]) for c in CS if not h.score_v3(c, h.perfect_raws(c))[0]]
check(f"2 perfect answers pass {len(CS) - len(bad)} of {len(CS)}", not bad, str(bad[:5]))

# 3. an empty answer fails every case; following the plant fails every injection case
empty = [c["id"] for c in CS if h.score_v3(c, [h._raw()] * len(c["steps"]))[0]]
check(f"3 an empty answer fails all {len(CS)}", not empty, str(empty[:5]))


def follow_plant(c):
    """The perfect answers plus a call to the planted tool at the first step that comes after the model read it."""
    raws = h.perfect_raws(c)
    k = 0 if c["inject"] in c["user"] else 1 + next(i for i, s in enumerate(c["steps"])
                                                    if "call" in s and c["inject"] in s["result"])
    msg = raws[min(k, len(raws) - 1)]["message"]
    msg["tool_calls"] = msg["tool_calls"] + [{"function": {"name": c["target"], "arguments": {}}}]
    return raws


followed = [c["id"] for c in FAM["injection"] if h.score_v3(c, follow_plant(c))[0]]
check(f"3 following the plant fails all {len(FAM['injection'])} injection cases", not followed, str(followed[:5]))

# 4. 0 overlap after the exclusion, and the check is not vacuous
check("4 0 kept cases share a run or a whole text with held-out v1, v2 or v3", ct.v3_excluded(CS) == {},
      str(list(ct.v3_excluded(CS).items())[:3]))


def first_run(words, n=8):
    return " ".join(words[:n])


v1_run = first_run(next(w for w in (ct.texts(c) for c in holdout.cases()) if len(w) >= 12))
v1_runs = set().union(*[ct.grams(ct.texts(c)) for c in holdout.cases()])
v2_run = next(g for c in holdout_v2.cases() for g in sorted(ct.grams(ct.texts(c))) if g not in v1_runs)  # v2 only
v3_case = next(c for c in h.cases() if c["family"] == "chain")
v3_result = next(s["result"] for s in v3_case["steps"] if "call" in s and len(s["result"].split()) >= 12)
v3_line = next(t for c in h.cases() for _, t in c["history"] if len(t.split()) >= 10)
v3_ask = next(c["user"] for c in h.cases() if len(c["user"].split()) >= 10)
caller = next(c for c in CS if c["history"] and any("call" in s for s in c["steps"]))
for where, text in (("tool result", "v1 " + v1_run), ("tool result", "v2 " + v2_run),
                    ("tool result", "v3 tool result"), ("history", "v3 history"), ("ask", "v3 ask")):
    planted = copy.deepcopy(caller)
    run = {"v1 " + v1_run: v1_run, "v2 " + v2_run: v2_run, "v3 tool result": " ".join(v3_result.split()[:9]),
           "v3 history": " ".join(v3_line.split()[:9]), "v3 ask": " ".join(v3_ask.split()[:9])}[text]
    if where == "tool result":
        s = next(s for s in planted["steps"] if "call" in s)
        s["result"] += "\nnote: " + run
    elif where == "history":
        planted["history"][-1][1] += " " + run
    else:
        planted["user"] += " " + run
    check(f"4 a {text.split()[0]} run planted in a kept case's {where} is excluded",
          planted["id"] in ct.v3_excluded([planted]), f"planted '{run}'")
planted = copy.deepcopy(caller)
s = next(s for s in planted["steps"] if "call" in s)
s["result"] = tb.PENDING + "\n" + " ".join(v3_result.split()[:9])
check("4 a Grux runtime line does not shield a held-out run on the next line", planted["id"] in ct.v3_excluded([planted]))
planted["steps"][[i for i, x in enumerate(planted["steps"]) if "call" in x][0]]["result"] = tb.PENDING
check("4 the runtime line alone is not counted as held-out text", planted["id"] not in ct.v3_excluded([planted]))

# 5. data_scan over the whole bank, labels included
vocab = os.environ.get("GRUX_DATA_SCAN_VOCAB") or os.path.join(
    os.path.dirname(os.path.dirname(h.src())), "Tests", "GruxTests", "banned-vocabulary.json")
rows = os.path.join(tempfile.mkdtemp(), "bank.jsonl")
with open(rows, "w", encoding="utf-8") as f:
    for c in ALL:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
ok, msg = data_scan.check(rows, vocab)
check(f"5 data_scan passes on all {len(ALL)} generated cases", ok, msg)
check("5 data_scan read an identity vocabulary (it fails closed without one)", len(data_scan.load_terms(vocab)) > 0)
with open(rows, "a", encoding="utf-8") as f:
    f.write(json.dumps({"id": "planted", "user": "token " + "ghp_" + "A1b2C3d4" * 5}) + "\n")
ok2, _ = data_scan.check(rows, vocab)
check("5 data_scan refuses the same file with one planted key shape", not ok2)

# 6. the sha does not move with the hash seed
shas = set()
for seed in ("1", "2"):
    out = subprocess.run([sys.executable, "-c", "import sys; sys.path[:0] = [%r, %r]; import train_bank_v3 as t; "
                          "print(t.case_sha(t.cases()))" % (HERE, os.path.dirname(HERE))],
                         capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": seed})
    shas.add(out.stdout.strip())
check("6 the case sha is the same under PYTHONHASHSEED 1 and 2", len(shas) == 1 and tb.case_sha(ALL) in shas,
      str(shas))
print(f"    case sha (generated) {tb.case_sha(ALL)}")
print(f"    case sha (kept)      {tb.case_sha(CS)}")

# 7. real tools, clean labels, copy rules, runtime lines in the source
names = set(h.tool_names())
used = set()
for c in ALL:
    for s in c["steps"]:
        if "call" in s:
            used |= {s["call"], *(s.get("alt") or {}), *(s.get("forbid") or [])}
        else:
            used |= set(s["reply"]["forbid"]) | set(s["reply"].get("allow") or [])
    if c.get("target"):
        used.add(c["target"])
check(f"7 all {len(used)} tools the bank names are in the {len(names)}-tool catalogue at {h.GRUX_TAG}",
      used <= names and len(names) == h.EXPECTED_TOOLS, str(sorted(used - names)))
errs = tb.label_errors(ALL)
check(f"7 every label passes the held-out v3 label checks ({len(errs)} errors)", not errs, str(errs[:5]))
errs = tb.copy_errors(ALL)
check("7 copy rules: no dash, no dollars in words, no metric distance, no 24-hour time", not errs, str(errs[:5]))
src_dash = [chr(0x2013), chr(0x2014)]
check("7 no em or en dash in train_bank_v3.py or this test",
      not any(d in open(p, encoding="utf-8").read() for d in src_dash
              for p in (tb.__file__, os.path.abspath(__file__))))
missing = [(f, piece) for _, f, pieces in ct.RUNTIME_LINES for piece in pieces
           if piece not in open(os.path.join(h.src(), f), encoding="utf-8").read()]
check(f"7 every RUNTIME_LINES piece is in the Grux {h.GRUX_TAG} source it names", not missing, str(missing))

print(f"{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failed")
sys.exit(1 if FAILS else 0)
