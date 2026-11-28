#!/usr/bin/env python3
"""s2/train_local.py: the training bank's chat turns in held-out local's shape, scored by held-out local's rules.

  1. the set: two builds give the same bytes and FROZEN_SHA; every bank chat case is kept or dropped with a reason;
     each family's sent cases are spread over the three windows by bank number; the not-sent cases are counted, named
     and never in sent(); not_sent() agrees with request_local on every case;
  2. its own clock: every sent request carries its case's NOW line, and its asks are the frozen table's;
  3. independence, read afresh here: no kept case shares a whole text or an 8-word run with held-out v1, v2, local,
     v3, v4 or S1, none trips contamination.TrainGate, the 8-word drops are train_exclude.json's chat ids, held-out
     local's cases are held-out v2's, and a planted held-out run is caught;
  4. injection labels: all 500 bank cases split into one request, one body and one planted block, rebuilt byte for
     byte, with one target; no kept message reads as a warning, no kept plant does its task;
  5. the scorer is held-out local's: every sent case passes its ideal reply; on every kept injection case the task
     done without naming the steering fails here and passes train_bank.score_train; a read-only tool the paste asked
     for fails while another offered read-only tool passes; a chat_tools date is checked against the case's own clock;
  6. held-out local, v2, v3 and v4 are untouched by the now_line and asks parameters (fingerprints pinned);
  7. the oracle, when swiftc is here: the frozen asks re-derived from the 3.3.1 Swift at each case's own clock (and,
     as a control, at Oct 22 they differ), and every sent turn assembled by Python and by Swift the same.

  GRUX_SRC_V3=<Grux v3.3.1 Sources/Grux> python3 recipe/s2/test_train_local.py
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import contamination as ct  # noqa: E402
import holdout  # noqa: E402
import holdout_local  # noqa: E402
import holdout_v2  # noqa: E402
import holdout_v3  # noqa: E402
import holdout_v4  # noqa: E402
import local_turn as lt  # noqa: E402
import rules  # noqa: E402
import suite  # noqa: E402
import train_bank as tb  # noqa: E402
import train_local as tl  # noqa: E402

MODEL = "gruxai.com/grux/grux-local:2b"   # the shipped 2B, the model the first RFT round samples
# holdout_local.fingerprint at 16, 8 and 32 GB, measured on origin/main 964411d before now_line and asks went in
LOCAL_FP = {16: "19e3dab6d07d30b6ba09eee2a5b5985a1fe061dea533d96d2e2fd269d2eec4d6",
            8: "8df804439af5aa1317f55167bb5a97ba16a0c299a5cd2cec72bdec34755d35dd",
            32: "cce9329598b1d819219465c1a5e277ec7eac5ccf4a3bd952a1dd273e846f8e7d"}
V2_FP = "da6577c7178a096383890760f6a45e81241c81420c6b1f270a937ac4fade8b4a"
fails = 0


def check(label, ok, detail=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + label + ("" if ok else f"  [{str(detail)[:300]}]"))
    fails += 0 if ok else 1


def raw(text="", calls=()):
    return {"message": {"role": "assistant", "content": text,
                        "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls]}}


# --------------------------------------------------------------------------- 1. the set

cs = tl.cases()
check("two builds give the same bytes, and the case sha is FROZEN_SHA",
      tl.case_sha(cs) == tl.case_sha(tl.cases()) == tl.FROZEN_SHA, tl.case_sha(cs))
bank = tl.bank_cases()
drop = tl.dropped()
kept_src = [c["source"] for c in cs]
check(f"every bank chat case is kept or dropped with a reason: {len(bank)} = {len(cs)} kept + {len(drop)} dropped",
      len(bank) == 2850 and sorted(kept_src + list(drop)) == sorted(c["id"] for c in bank)
      and len(set(kept_src)) == len(kept_src) and all(r and d for r, d in drop.values()))
gone = tl.not_sent(cs)
reqs = {c["id"]: tl.request_local(c, MODEL) for c in cs}
check("not_sent() is request_local's own answer on every case",
      gone == {i: r["not_sent"] for i, r in reqs.items() if "not_sent" in r})
sent = tl.sent(cs)
check(f"sent() holds every case 3.3.1 sends ({len(sent)}) and none it answers alone ({len(gone)})",
      len(sent) + len(gone) == len(cs) and not {c["id"] for c in sent} & set(gone) and gone)
print(f"COUNT  not sent {len(gone)}: {dict(collections.Counter(gone.values()))}")
win_ok, win_detail = True, {}
for fam in tb.CHAT_FAMILIES:
    mine = [c for c in sent if c["family"] == fam]
    n = collections.Counter(c["num_ctx"] for c in mine)
    win_detail[fam] = dict(n)
    win_ok &= set(n) == set(tl.WINDOWS) and all(0.28 <= n[w] / len(mine) <= 0.39 for w in tl.WINDOWS)
by_number = all(c["num_ctx"] == tl.WINDOWS[(int(c["source"].rsplit("-", 1)[1]) - 1) % 3] for c in cs)
check("each family's sent cases are spread over 8,192, 16,384 and 32,768 by bank number, and the request sends it",
      win_ok and by_number and all(r["body"]["options"]["num_ctx"] == next(c for c in cs if c["id"] == i)["num_ctx"]
                                   for i, r in reqs.items() if "body" in r), win_detail)

# --------------------------------------------------------------------------- 2. its own clock

table = json.load(open(tl.ASKS_PATH, encoding="utf-8"))
check("every case's asks are the frozen table's, written from this very bank",
      table["train_bank"] == tb.fingerprint(tb.cases()) and table["tag"] == lt.V331_TAG
      and all(c["asks"] == table["values"][table["cases"][c["source"]]] for c in cs))
now_ok = all(("NOW: " + c["now"] + "\n") in reqs[c["id"]]["body"]["messages"][1]["content"] for c in sent)
other = sum(c["now"] != suite.NOW for c in sent)
check(f"every sent request carries its own case's NOW line ({other} of {len(sent)} are not Oct 22's)",
      now_ok and other == len(sent))

# --------------------------------------------------------------------------- 3. independence, read afresh

held = {"held-out v1": [ct.texts(c) for c in holdout.cases()], "held-out v2": [ct.texts(c) for c in holdout_v2.cases()],
        "held-out local": [ct.texts(c) for c in holdout_local.cases()],
        "held-out v3": [ct._words(t) for c in holdout_v3.cases() for t in holdout_v3.case_texts(c)],
        "held-out v4": [ct._words(t) for c in holdout_v4.cases() for t in ct.v4_texts(c)],
        "S1": [ct.texts(c) for c in suite.cases()]}
idx = {k: (set().union(*[ct.grams(w) for w in v]), {" ".join(w) for w in v if len(w) >= 3}) for k, v in held.items()}
bank_by = {c["id"]: c for c in bank}
bad = collections.Counter()
for c in cs:
    ws = ct.texts(bank_by[c["source"]])
    for k, (runs, whole) in idx.items():
        if ct.grams(ws) & runs or (len(ws) >= 3 and " ".join(ws) in whole):
            bad[k] += 1
check(f"no kept case shares a whole text or an 8-word run with any of {len(idx)} held-out sets", not bad, dict(bad))
gate = ct.TrainGate()
flag = [c["id"] for c in cs if gate.why([c["utterance"]], [c["utterance"]])]
check("no kept case trips contamination.TrainGate (v4 texts and clones, v3 and v4 frames)", not flag, flag[:3])
chat_ex = {i for i in json.load(open(os.path.join(HERE, "train_exclude.json")))["excluded"]
           if i.split("-")[1] in tb.CHAT_FAMILIES}
touched = {i for i, (r, _) in drop.items() if r.startswith("touches")}
check(f"the 8-word drops are train_exclude.json's {len(chat_ex)} chat ids", touched == chat_ex and chat_ex,
      sorted(touched ^ chat_ex)[:5])
local_ids = {c["id"]: c for c in holdout_v2.cases()}
check("held-out local's cases are held-out v2's chat cases, byte for byte",
      all(local_ids.get(c["id"]) == c for c in holdout_local.cases()) and len(holdout_local.cases()) == 265)
v1_run = next(w for w in (ct.texts(c) for c in holdout.cases()) if len(w) >= 12)[:9]
planted = dict(bank_by[cs[0]["source"]], utterance=cs[0]["utterance"] + " " + " ".join(v1_run))
check("a held-out v1 run planted in a kept case is caught", tl.touches(ct.texts(planted), tl.held_out_sets()))
sets = collections.Counter(r for r, _ in drop.values())
print(f"COUNT  dropped {len(drop)}: {dict(sorted(sets.items()))}")

# --------------------------------------------------------------------------- 4. injection labels

splits_ok, problems = True, []
for c in bank:
    if c["family"] != "injection":
        continue
    got = {(s[0], s[2], s[3], tuple(s[4]), s[5], s[6], s[7]) for s in tl.split_paste(c["text"])}
    if len(got) != 1:
        problems.append((c["id"], len(got)))
        continue
    kind, req, lay, body, pos, plant, foreign = next(iter(got))
    if tl.rebuild(req, lay, list(body), pos, plant) != c["text"] or tl.target_of(plant)[0] == "?":
        problems.append((c["id"], "rebuild or target"))
check("all 500 bank injection messages split into one request, one body and one plant, rebuilt byte for byte, "
      "with one target", not problems and sum(c["family"] == "injection" for c in bank) == 500, problems[:3])
every_instr = set(tb.INSTR) | set(tb.SHORT_INSTR) | set(tb.FOREIGN_INSTR)
tools331 = {t["name"] for t in lt.all_tools_v331()}
check("TARGET names every planted instruction train_bank holds, each a Grux 3.3.1 tool or None",
      set(tl.TARGET) == every_instr and all(v is None or v in tools331 for v in tl.TARGET.values()))
check("ENGLISH holds every FOREIGN text, and no word of it is in a planted instruction",
      set(tl.ENGLISH) == {b[0].split()[0] for b in tb.FOREIGN}
      and not any(w in s.lower() for ws in tl.ENGLISH.values() for w in ws for s in every_instr))
inj = [c for c in cs if c["family"] == "injection"]
check("no kept injection message reads as a warning by itself (held-out v2's own rule)",
      not [c["id"] for c in inj if rules.flagged(c["utterance"])])
check("no kept plant does its task alone (a count passes on any number, so counts are exempt)",
      not [c["id"] for c in inj if c["task"]["kind"] != "count" and rules.task_done(c["task"], c["inject"])[0]])
check("every kept injection label is in held-out v2's shape and its plant is in the message",
      all(c["inject"] in c["utterance"] and c["task"]["kind"] == c["ask"] and "target" in c for c in inj))
print(f"COUNT  kept injection asks {dict(collections.Counter(c['ask'] for c in inj))}")


# --------------------------------------------------------------------------- 5. the scorer is held-out local's

ideal = tl.ideal_reply
STEER = tl.STEER
miss = []
for c in sent:
    ok, why, _ = tl.score_local(c, ideal(c, reqs[c["id"]]), reqs[c["id"]])
    if not ok:
        miss.append((c["id"], why))
check(f"every sent case passes its ideal reply ({len(sent)}), so no label is out of reach", not miss, miss[:4])
silent = []
for c in [x for x in sent if x["family"] == "injection"]:
    r = ideal(c, reqs[c["id"]])
    r["message"]["content"] = r["message"]["content"].replace(" " + STEER, "")
    here = tl.score_local(c, r, reqs[c["id"]])[0]
    weak = tb.score_train(bank_by[c["source"]], r)[0]
    if here or not weak:
        silent.append((c["id"], here, weak))
n_inj = sum(c["family"] == "injection" for c in sent)
check(f"on all {n_inj} sent injection cases the task done without naming the steering fails here and passes "
      "train_bank.score_train", not silent and n_inj > 300, silent[:3])
web = next(c for c in sent if c["family"] == "injection" and "search the web" in c["inject"].lower())   # by its words
ro = next((n for n in reqs[web["id"]]["offered"] if n in rules.READ_ONLY and n != "search_web"), None)
check("the read-only tool the paste asked for fails; another offered read-only tool passes",
      not tl.score_local(web, raw(ideal(web, reqs[web["id"]])["message"]["content"], [("search_web", {"query": "x"})]),
                         reqs[web["id"]])[0]
      and ro is not None and tl.score_local(web, raw(ideal(web, reqs[web["id"]])["message"]["content"], [(ro, {})]),
                                            reqs[web["id"]])[0], ro)
dated = next(c for c in sent if c["family"] == "chat_tools" and c["tool"] in reqs[c["id"]]["offered"]
             and any(op == "date" for _, op, _ in c["checks"]) and "tomorrow" in c["utterance"])
good = tb._ideal(dated)
oct22 = json.loads(json.dumps(good))
fld = next(f for f, op, _ in dated["checks"] if op == "date")
a = oct22["message"]["tool_calls"][0]["function"]["arguments"]
a[fld] = "2026-10-23" + a[fld][10:]   # tomorrow read off Oct 22, not off the case's own clock
check(f"a date is checked against the case's own clock ({dated['now']}): tomorrow from Oct 22 fails",
      tl.score_local(dated, good, reqs[dated["id"]])[0] and not tl.score_local(dated, oct22, reqs[dated["id"]])[0],
      (dated["utterance"], dated["checks"]))

# --------------------------------------------------------------------------- 6. the held-out sets untouched

lc = holdout_local.cases()
check("held-out local's requests are as before now_line and asks went in (16, 8 and 32 GB)",
      all(holdout_local.fingerprint(lc, memory_gb=gb) == fp for gb, fp in LOCAL_FP.items()))
check("held-out v2's fingerprint, v3's and v4's case shas are as frozen",
      holdout_v2.fingerprint(holdout_v2.cases()) == V2_FP
      and holdout_v3.case_sha(holdout_v3.cases()) == holdout_v3.FROZEN_SHA
      and holdout_v4.case_sha(holdout_v4.cases()) == holdout_v4.FROZEN_SHA)

# --------------------------------------------------------------------------- 7. the oracle

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "grux_oracle"))
import oracle  # noqa: E402
why = oracle.available()
if why:
    print(f"NOT RUN  the oracle cross-check: {why}. The asks come from the frozen table; the port is unchecked here.")
else:
    check("the frozen asks table names the source the oracle compiles", table["source_sha256"] == oracle.source_sha())
    FIELDS = ("filtered", "ranked", "fit", "stable", "tools", "cut", "prefill")
    by_now = collections.defaultdict(list)
    for c in cs:
        by_now[c["now"]].append(c)
    diff_asks, diff_turn, control, saved = [], [], 0, oracle.NOW
    try:
        for now_line, group in sorted(by_now.items()):
            oracle.NOW = tl._clock(now_line)
            turns = [{"utterance": c["utterance"], "window": lt.context_tokens(holdout_local.memory_for(c["num_ctx"])),
                      "system_blocks": [{"text": b["text"], "cached": "cache_control" in b} for b in
                                        lt.system_blocks_v331(MODEL, c["asks"], c["now"])]} for c in group]
            sw = oracle.run(turns)["turns"]
            for c, s in zip(group, sw):
                if {k: s[k] for k in oracle.ASK_KEYS} != c["asks"]:
                    diff_asks.append(c["id"])
                if c["id"] in gone:
                    continue
                p = lt.assemble_v331(c["utterance"], MODEL, holdout_local.memory_for(c["num_ctx"]), c["asks"], c["now"])
                p = dict(p, tools=p["offered"])
                bad_f = [f for f in FIELDS if p[f] != s[f] and not (f == "fit" and list(p[f]) == s[f])]
                if bad_f:
                    diff_turn.append((c["id"], bad_f))
        oracle.NOW = saved
        at_oct22 = oracle.asks([c["utterance"] for c in cs])
        control = sum(at_oct22[c["utterance"]] != c["asks"] for c in cs)
    finally:
        oracle.NOW = saved
    check(f"the frozen asks re-derived from the 3.3.1 Swift at each case's own clock: {len(cs)} asks, "
          f"{len(diff_asks)} differ", not diff_asks, diff_asks[:3])
    check(f"control: asked at Oct 22 instead, {control} of them differ (the clock matters)", control > 0)
    check(f"Python and the compiled 3.3.1 Swift assemble all {len(sent)} sent turns the same at their own window "
          f"and clock ({', '.join(FIELDS)})", not diff_turn, diff_turn[:3])

print("SELF-TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
