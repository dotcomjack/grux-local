#!/usr/bin/env python3
"""The Grux 3.3.1 local turn (local_turn.chat_request_v331) against the app's own code, and held-out v2 untouched.

  1. held-out v2 is byte for byte what it was: every one of its 466 requests hashed, against the hash recorded before
     the 3.3.1 port went in, and its fingerprint against check_prompts.EXPECTED_V2 (needs GRUX_SRC at v3.1.0);
  2. the Swift tests' own cases, ported: LocalTurnStableToolsTests, LocalTurnPrefixTests, LocalChipToolsTests,
     FirstRunWalkW2dTests and LocalTurnSteppedHistoryTests at v3.3.1;
  3. THE HEAD RULE: everyday asks get one tool list led by LocalTurn.stableHead and one cut line, and most held-out
     turns take it. This is the check proven red by disabling stable_local_tools;
  4. TrainedTools: every hash in TrainedToolsV4.swift reproduced from Python's bytes (the sorted-key JSON a -v4 tag's
     request and its signatures are built on);
  5. the oracle (recipe/grux_oracle, the 3.3.1 Swift compiled as is): every held-out turn and every test ask, at
     16 GB and at 8 GB, assembled by Python and by Swift, compared field by field; the frozen asks table re-derived
     and compared. A control first: the 3.1.0 rank, swapped in, must disagree with the oracle, or the compare sees
     nothing. Needs swiftc; without it this part says NOT RUN and why;
  6. the counts: how many held-out chat turns take the stable list and how many the ranked one, and why.
No model runs. Prints PASS or FAIL per check; exit 1 on any FAIL.

  GRUX_SRC=<v3.1.0 tree> GRUX_SRC_V3=<v3.3.1 tree> python3 recipe/test_local_turn_v331.py
"""
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "s2"))
sys.path.insert(0, os.path.join(HERE, "grux_oracle"))
import local_turn as lt  # noqa: E402
import holdout_local  # noqa: E402

# sha256 over [id, request_v2(c, "m")] for every held-out v2 case, measured on origin/main (f2c69e8) before the
# 3.3.1 port, GRUX_SRC at public Grux v3.1.0.
V2_REQUEST_SHA = "89475d1c63a498a2393aebd231438269dbf9ae6a0dc69cff70c882688619023f"
MODEL = "gruxai.com/grux/grux-local:4b"
fails = 0


def check(label, ok, detail=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + label + ("" if ok else f"  [{detail}]"))
    fails += 0 if ok else 1


def tool(n, d):
    return {"name": n, "description": d, "schema": {}}


# --------------------------------------------------------------------------- 1. held-out v2 untouched

if not os.environ.get("GRUX_SRC"):
    check("held-out v2 untouched", False, "GRUX_SRC is not set: point it at Grux v3.1.0 or nothing is proven")
else:
    import re
    import holdout_v2
    # check_prompts.py runs on import, so its recorded value is read, not imported
    EXPECTED_V2 = re.search(r'EXPECTED_V2 = "([0-9a-f]+)"', open(os.path.join(HERE, "check_prompts.py")).read()).group(1)
    v2 = holdout_v2.cases()
    h = hashlib.sha256()
    for c in v2:
        h.update(json.dumps([c["id"], holdout_v2.request_v2(c, "m")], sort_keys=True, ensure_ascii=False).encode())
    check(f"held-out v2: all {len(v2)} requests hash as before the port ({h.hexdigest()[:16]})",
          len(v2) == 466 and h.hexdigest() == V2_REQUEST_SHA, h.hexdigest())
    fp = holdout_v2.fingerprint(v2)[:16]
    check(f"held-out v2 fingerprint {fp} is check_prompts.EXPECTED_V2", fp == EXPECTED_V2, fp)

# --------------------------------------------------------------------------- 2. the Swift tests' own cases

every = lt.all_tools_v331()
check(f"catalogue at {lt.V331_TAG}: {len(every)} tools", len(every) == lt.V331_TOOLS)
check("stableHead is the eight everyday tools less the two that open another app, then four core tools",
      lt.STABLE_HEAD == ["list_tasks", "add_task", "complete_task", "list_events", "create_event", "capture_memory",
                         "set_mode", "create_note", "remember_fact", "remove_task"], lt.STABLE_HEAD)
# LocalTurnStableToolsTests.test_stableOrderLeadsWithTheHeadThenKeepsTheUsualOrder
check("stableOrder leads with the head, then the usual order",
      [t["name"] for t in lt.stable_order([tool("zeta", "z"), tool("add_task", "a"), tool("alpha", "a"),
                                           tool("list_tasks", "l")])] == ["list_tasks", "add_task", "zeta", "alpha"])
# test_pointsPastNeedsAStrongMatchOutsideTheList
inside = [tool("list_tasks", "List the tasks"), tool("add_task", "Add a task")]
allt = inside + [tool("transcript_fetcher", "Fetch the transcript of a youtube video")]
check("pointsPast: three words outside the list point past it",
      lt.points_past(inside, allt, "fetch the transcript of this youtube video"))
check("pointsPast: two words do not", not lt.points_past(inside, allt, "add a task to watch a youtube video"))
check("pointsPast: an everyday ask does not", not lt.points_past(inside, allt, "what is on my plate"))
# test_aShortAskForAGroupToolPointsPast
inside2 = [tool("list_tasks", "List the tasks"), tool("get_current_activity", "Read what is on the screen now")]
check("pointsPast: two words for a group tool point past",
      lt.points_past(inside2, inside2 + [tool("start_meeting_capture", "Start capturing a meeting recording")],
                     "record a meeting"))
check("pointsPast: a tie with a listed tool goes to the group's tool",
      lt.points_past(inside2, inside2 + [tool("read_screen", "Read the screen")], "read my screen"))
check("pointsPast: a core tool needs three words",
      not lt.points_past(inside2, inside2 + [tool("create_note", "Write a note")], "write a note"))
# test_fillerWordsAndATieWithAnEverydayToolStayOnTheList
inside3 = [tool("list_tasks", "List the tasks"), tool("complete_task", "Mark a task complete")]
check("pointsPast: a filler word or a tie with complete_task keeps the list",
      not lt.points_past(inside3, inside3 + [tool("list_research_reports", "List research reports, complete with sources")],
                         "done with the expense report, mark it complete"))
check("pointsPast: \"about\" does not count",
      not lt.points_past(inside3, inside3 + [tool("read_workday_log", "Read the workday log, about each task, complete or not")],
                         "complete the task about the watercolor class"))
# test_intentVerbsStillPointPast
check("pointsPast: \"make\" is an intent word",
      lt.points_past(inside, inside + [tool("create_folder", "Make a folder for meeting captures")],
                     "make a folder for client notes")
      and not {"make", "know", "tell", "want", "need"} & lt.FILLER_WORDS)
# FirstRunWalkW2dTests.test_theToolsAskedAboutComeFirst, LocalTurnPrefixTests.test_aPluralFindsItsSingularButNewsIsNotNew
toy = [tool("play_music_track", "Play a song"), tool("create_note", "Write a note"), tool("list_tasks", "List the tasks"),
       tool("search_web", "Search the web")]
check("rank: the note tool leads \"write a note about lunch\"", lt.rank_v331(toy, "write a note about lunch")[0]["name"] == "create_note")
check("rank: the everyday tools lead an unclear message", lt.rank_v331(toy, "hi")[0]["name"] == "list_tasks")
check("rank: \"news\" is not \"new\"",
      lt.rank_v331([tool("add_task", "Add a new task"), tool("read_feed", "Read the latest news headlines")],
                   "what's the news")[0]["name"] == "read_feed")
check("rank: \"tasks\" finds \"task\"",
      lt.rank_v331([tool("other", "Something else"), tool("add_task", "Add a task")], "my tasks")[0]["name"] == "add_task")
# ToDoListAskTests: "to-do" is a task word
ranked = [t["name"] for t in lt.rank_v331(every, "Make a to-do list: milk, eggs")]
check("rank: a to-do list ranks add_task above list_meetings", ranked.index("add_task") < ranked.index("list_meetings"))
# test_fitKeepsTheNewestMessageAndPrefersToolsToOldHistory
m, n = lt.fit_v331(7_000, [2_000] * 20 + [100], [700] * 50, 8_192)
check(f"fit keeps the newest message and prefers tools to old history ({m} messages, {n} tools)", 1 <= m < 21 and n > 0)
# LocalTurnSteppedHistoryTests, LocalTurnPrefixTests.test_historyStartMovesInWholeSteps
hs = [(0, 0, 0), (30, 30, 0), (31, 31, 10), (40, 40, 10), (41, 41, 20), (5, 1, 4), (5, 0, 4), (12, 3, 9), (25, 13, 12),
      (26, 13, 13), (30, 10, 20), (45, 40, 20), (25, 1, 24), (1, 1, 0)]
check("historyStart moves in whole steps and keeps what fits", all(lt.history_start(a, b) == w for a, b, w in hs),
      [(a, b, w, lt.history_start(a, b)) for a, b, w in hs if lt.history_start(a, b) != w])
# LocalChipToolsTests
for ask, need in [("Undo `rm notes.txt` from 5:00 PM", [lt.LIST_TOOL, lt.RESTORE_TOOL]),
                  ("Why did you refuse `afplay /tmp/x.aiff`?", [lt.AUDIT_TOOL]),
                  ("What did you run today? (3 commands)", [lt.AUDIT_TOOL])]:
    got = lt.assemble_v331(ask, MODEL)["offered"]
    check(f"chip ask {ask!r} carries {need}", all(x in got for x in need), got)
ordinary = lt.assemble_v331("write a list", MODEL)["offered"]
check("an ordinary turn sends no recovery tool, a chip turn no more tools than it",
      not {lt.AUDIT_TOOL, lt.LIST_TOOL, lt.RESTORE_TOOL} & set(ordinary)
      and all(len(lt.assemble_v331(a, MODEL)["offered"]) <= len(ordinary) for a in
              ["Undo `rm notes.txt` from 5:00 PM", "Why did you refuse `afplay /tmp/x.aiff`?",
               "What did you run today? (3 commands)"]), ordinary)
check("only the chip shapes take chip tools", all(lt.chip_tools(a) == [] for a in
                                                  ["undo that paste", "what did I do today", "why did you say that"]))
# LocalTurnPrefixTests.test_theToolsGoOutInOneOrderWhateverWasAsked, test_theToolsThatOpenAnotherAppGoLast,
# test_theCutLineNamesToolsInOneOrder
order = lt.canonical_order(every)
for ask in ["hi", "what is on my calendar tomorrow", "open github.com"]:
    names = lt.assemble_v331(ask, MODEL)["offered"]
    pos = [order[x] for x in names]
    check(f"{ask!r}: tools go out in the one canonical order", pos == sorted(pos), names)
check("\"what is on my calendar tomorrow\" offers list_events",
      "list_events" in lt.assemble_v331("what is on my calendar tomorrow", MODEL)["offered"])
opens = [order[n] for n in ("open_url", "open_app", "play_on_youtube", "search_web")]
check("the tools that open another app go last in the canonical order",
      min(opens) > max(v for k, v in order.items() if lt.group_v331(k) != "otherApps"))
op = lt.assemble_v331("open github.com", MODEL)["offered"]
first = next((i for i, x in enumerate(op) if lt.group_v331(x) == "otherApps"), None)
check("\"open github.com\" offers a tool that opens a page, at the end", first is not None
      and all(lt.group_v331(x) == "otherApps" for x in op[first:]), op)
cut = lt.assemble_v331("what is on my calendar tomorrow", MODEL, memory_gb=8)["cut"] or ""
cut_names = cut.replace("TOOLS NOT OFFERED IN THIS LOCAL TURN: ", "").split(". ")[0].split(", ")
cpos = [order[x] for x in cut_names if x in order]
check(f"the cut line names {len(cpos)} tools in the one order", len(cpos) > 3 and cpos == sorted(cpos))
# test_messageLengthDoesNotChangeTheList, test_aLocalTurnStillFitsAnEightGigabyteWindow
a = lt.assemble_v331("add a task to buy milk", MODEL)
b = lt.assemble_v331("add a task to buy milk" + " and bread" * 260, MODEL)
check("a 2,600 character longer message does not change the list", a["offered"] == b["offered"] and a["stable"])
p = lt.assemble_v331("add a task to buy milk", MODEL, memory_gb=8)
size = lt.tokens(sum(lt.swift_count(x["text"]) for x in p["system_blocks"])) \
    + sum(lt.tokens(lt.message_chars(x)) for x in p["messages"]) + sum(lt.tokens(lt.tool_chars(t)) for t in p["tools"])
check(f"either path fits an 8 GB window ({size} + {lt.REPLY_TOKENS} tokens)", size + lt.REPLY_TOKENS <= 8_192)
# test_aTurnThatWouldNotFitItsWindowKeepsTheRankedList
huge = lt.assemble_v331("word " * 8_000, MODEL, memory_gb=8)
check("a turn that would not fit its window takes the ranked list",
      not huge["stable"] and huge["why_ranked"] == "does_not_fit", huge["why_ranked"])
# test_aLargeTurnBlockKeepsTheRankedListOnEightGigabytesOnly (the Swift test's own blocks)
cached = [{"type": "text", "text": "a" * 6_000, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "NOW: Monday"}]
big = [cached[0], {"type": "text", "text": "m" * 12_000}]
kept = lt.relevance_filter_v331(every, "add a task to buy milk")
msgs = [{"role": "user", "content": "add a task to buy milk"}]
s8 = lt.stable_local_tools(big, msgs, kept, "add a task to buy milk", 8_192, 8_192)
s16 = lt.stable_local_tools(big, msgs, kept, "add a task to buy milk", 8_192, 16_384)
sref = lt.stable_local_tools(cached, msgs, kept, "add a task to buy milk", 8_192, 16_384)
check("a large turn block takes the ranked list on 8 GB only, and does not change the list on 16 GB",
      s8 is None and s16 is not None and [t["name"] for t in s16] == [t["name"] for t in sref])

# DraftAsk / ListAsk (D159, D167, D175): no tools, the note in the turn block, the reply's start prefilled
for ask, start in [("write a text to Priya saying I will be ten minutes late", "Hi Priya,"),
                   ("make a packing list for a beach weekend", "Here is a packing list for a beach weekend:")]:
    r = lt.chat_request_v331(ask, MODEL)
    msgs = r["body"]["messages"]
    check(f"{ask!r}: no tools, no cut line, a note, and the reply starts {start!r}",
          "tools" not in r["body"] and r["cut"] is None and r["why_ranked"] == "no_tools"
          and msgs[-1] == {"content": start, "role": "assistant"} and msgs[-2]["role"] == "user"
          and ("DRAFT: " in msgs[-2]["content"] or "LIST: " in msgs[-2]["content"]), [m["role"] for m in msgs])
raw = {"message": {"role": "assistant", "content": " running ten minutes late, sorry!"}}
check("the scorer reads the prefilled start and the model's words after it",
      holdout_local.with_prefill(raw, "Hi Priya,")["message"]["content"] == "Hi Priya, running ten minutes late, sorry!"
      and raw["message"]["content"] == " running ten minutes late, sorry!")
# OpenAICompatBackend: think false, keep_alive 30m, keys sorted at every level
body = lt.chat_request_v331("what is on my plate today?", MODEL)["body"]


def sorted_everywhere(o):
    if isinstance(o, dict):
        return list(o) == sorted(o) and all(sorted_everywhere(v) for v in o.values())
    return all(sorted_everywhere(v) for v in o) if isinstance(o, list) else True


check("the body says think false and keep_alive 30m, num_predict 1024, keys sorted at every level",
      body["think"] is False and body["keep_alive"] == "30m" and body["options"]["num_predict"] == 1024
      and body["options"]["num_ctx"] == 16_384 and sorted_everywhere(body))
check("on the wire the system message holds no per-turn line, the user's words come last (LocalTurnPrefixTests)",
      body["messages"][0]["role"] == "system" and "NOW: " not in body["messages"][0]["content"]
      and body["messages"][1]["content"].startswith(lt.suite.TURN_LABEL)
      and body["messages"][1]["content"].endswith("THE USER'S MESSAGE:\nwhat is on my plate today?"))

# --------------------------------------------------------------------------- 3. THE HEAD RULE

e1, e2, e3 = (lt.assemble_v331(x, MODEL) for x in
              ["what is on my plate today?", "add a task to call the dentist tomorrow", "what's the tallest mountain in Africa"])
check("HEAD: three everyday asks get the same tools, led by stableHead, with one cut line",
      e1["offered"] == e2["offered"] == e3["offered"] and e1["offered"][:len(lt.STABLE_HEAD)] == lt.STABLE_HEAD
      and e1["cut"] is not None and e1["cut"] == e2["cut"] == e3["cut"],
      f"{e1['offered'][:4]} {e2['offered'][:4]} {e3['offered'][:4]}")
check("HEAD: those turns took the stable list (a control: a ranked turn would rank them differently)",
      e1["stable"] and e2["stable"] and e3["stable"]
      and [t["name"] for t in lt.rank_v331(lt.relevance_filter_v331(every, "what's the tallest mountain in Africa"),
                                         "what's the tallest mountain in Africa")][:2]
      != [t["name"] for t in lt.rank_v331(lt.relevance_filter_v331(every, "add a task to call the dentist tomorrow"),
                                        "add a task to call the dentist tomorrow")][:2])

cs = holdout_local.cases()
reqs = {c["id"]: holdout_local.request_local(c, MODEL) for c in cs}
sent = [c for c in cs if "not_sent" not in reqs[c["id"]]]
heads = [c for c in sent if reqs[c["id"]]["stable"]]
check(f"HEAD: {len(heads)} of {len(sent)} sent held-out turns take the one stable list, all with the same tools",
      len(heads) > len(sent) // 2 and len({tuple(reqs[c['id']]['offered']) for c in heads}) == 1)

# --------------------------------------------------------------------------- 4. TrainedTools

v4 = lt.trained_v4(lt.src_v331())
same = [t["name"] for t in every if v4.get(t["name"]) == lt.schema_hash(t)]
check(f"TrainedToolsV4: {len(same)} of {len(v4)} trained hashes reproduced from Python's sorted-key bytes",
      len(v4) > 0 and len(same) == len(v4), sorted(set(v4) - set(same)))
check("only a grux-local tag ending -v4 is a trained model",
      lt.is_trained_model("gruxai.com/grux/grux-local:4b-v4") and lt.is_trained_model("grux-local:4b-v4")
      and not lt.is_trained_model("grux-local:4b") and not lt.is_trained_model(MODEL)
      and not lt.is_trained_model("qwen3.5:4b-v4"))
r4 = lt.chat_request_v331("what's the tallest mountain in Africa", "gruxai.com/grux/grux-local:4b-v4")
check("a -v4 tag sends signatures in place of descriptions, parameters unchanged",
      any(t["function"]["description"].startswith(t["function"]["name"] + "(") for t in r4["body"]["tools"])
      and r4["offered"] == lt.chat_request_v331("what's the tallest mountain in Africa", MODEL)["offered"])

# --------------------------------------------------------------------------- 5. the oracle

import oracle  # noqa: E402
why = oracle.available()
if why:
    print(f"NOT RUN  the oracle cross-check: {why}. The asks come from the frozen table; the port is unchecked here.")
else:
    table = json.load(open(lt.ASKS_PATH, encoding="utf-8"))
    check("the frozen asks table names the source the oracle compiles",
          table["source_sha256"] == oracle.source_sha() and table["tag"] == lt.V331_TAG)
    utts = list(table["asks"])
    fresh = oracle.asks(utts)
    diff = [u for u in utts if fresh[u] != table["asks"][u]]
    check(f"the frozen asks table re-derived from Swift: {len(utts)} asks, {len(diff)} differ", not diff, diff[:3])
    held = oracle.held_out_utterances()
    check(f"the table holds every held-out chat turn ({len(held)})", len(held) == 265 and set(held) <= set(utts))
    FIELDS = ("filtered", "ranked", "fit", "stable", "tools", "cut", "prefill")

    def compare(memory_gb, rank=None):
        saved = lt.rank_v331
        if rank:
            lt.rank_v331 = rank
        try:
            py = {u: lt.assemble_v331(u, MODEL, memory_gb) for u in utts}
            turns = [{"utterance": u, "window": lt.context_tokens(memory_gb),
                      "system_blocks": [{"text": b["text"], "cached": "cache_control" in b}
                                        for b in lt.system_blocks_v331(MODEL, lt.asks_v331(u))]} for u in utts]
        finally:
            lt.rank_v331 = saved
        sw = oracle.run(turns)
        bad = []
        for u, s in zip(utts, sw["turns"]):
            p = dict(py[u], tools=py[u]["offered"])
            for f in FIELDS:
                if p[f] != s[f] and not (f == "fit" and list(p[f]) == s[f]):
                    bad.append((f, u[:50], p[f], s[f]))
        return bad, sw

    control, _ = compare(16, rank=lt.rank)
    check(f"control: the 3.1.0 rank swapped in disagrees with Swift on {len({b[1] for b in control})} turns",
          len(control) > 0)
    for gb in (16, 8):
        bad, sw = compare(gb)
        check(f"{gb} GB: Python and the compiled 3.3.1 Swift assemble all {len(utts)} turns the same "
              f"({', '.join(FIELDS)})", not bad, bad[:3])
    g = sw["global"]
    check("Swift's stableHead, canonical order and tool sizes are Python's",
          g["stable_head"] == lt.STABLE_HEAD and g["canonical"] == [t["name"] for t in lt.stable_order(every)]
          and g["chars"] == [lt.tool_chars(t) for t in every])
    check("Swift's signatures and schema hashes for all 119 tools are Python's",
          all(g["signatures"][t["name"]] == lt.signature(t) and g["hashes"][t["name"]] == lt.schema_hash(t)
              for t in every) and sorted(g["trained"]) == sorted(same),
          [t["name"] for t in every if g["signatures"][t["name"]] != lt.signature(t)][:3])

# --------------------------------------------------------------------------- 6. the counts

ranked_by = collections.Counter(reqs[c["id"]]["why_ranked"] for c in sent if not reqs[c["id"]]["stable"])
not_sent = collections.Counter(reqs[c["id"]]["not_sent"] for c in cs if "not_sent" in reqs[c["id"]])
print(f"COUNT  held-out v2's {len(cs)} chat turns under 3.3.1: {len(cs) - len(sent)} answered with no model "
      f"{dict(not_sent)}; {len(sent)} sent: {len(heads)} stable list, {len(sent) - len(heads)} ranked {dict(ranked_by)}")
# The Swift comment (LocalTurn.swift:247) counts all 265 turns through the list rule, fast paths aside: here too.
every_turn = [lt.assemble_v331(c["utterance"], MODEL) for c in cs]
allr = collections.Counter(t["why_ranked"] for t in every_turn if not t["stable"])
print(f"COUNT  all {len(cs)} turns assembled, fast paths ignored: {sum(t['stable'] for t in every_turn)} stable, "
      f"{sum(allr.values())} ranked {dict(allr)}; the Swift comment says 35 of 265 ranked")
check("every held-out turn is either not sent, stable or ranked with a reason",
      all("not_sent" in reqs[c["id"]] or reqs[c["id"]]["stable"] or reqs[c["id"]]["why_ranked"] for c in cs))

print("SELF-TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
