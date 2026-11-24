"""Held-out "local": held-out v2's chat turns, sent the way the current Grux Mac app (tag v3.3.1) sends a local turn.

Held-out v2 (s2/holdout_v2.py) stays frozen: it sends each chat case as a Grux 3.1.0 local turn and its request bytes
do not change. This set takes the same 265 chat-shaped cases (chat_tools, chat_plain, chat_honesty, injection), with
the same ids, labels and scorer (holdout_v2.score_v2), and sends each one as Grux 3.3.1 does
(local_turn.chat_request_v331): the stable tool list most turns take or the ranked one, the cut line, ABOUT_YOU and the
new tool note, the per-turn notes, a prefilled reply start on a draft or list ask, think false, keep_alive 30m.

What a case can be, every one counted and named by the self-test, never dropped silently:
  - not sent: 3.3.1 answers the message itself, with no model (a PIM card, the calendar, the math or date answer, a
    clarifying question from ConfidenceGate, ...). eval_local writes it as status "not_sent" with the reason, outside
    the pass rate. Held-out v2 sent these as model turns; 3.1.0 already answered the PIM and clarify ones itself.
  - sent, scored against the tools that turn offered (holdout_v2.score_v2). For a chat_tools case the expected tool
    is offered, or named in the cut line (v2's rule: pass by saying plainly it cannot be done in this local turn), or
    neither (filtered out by ToolRelevance, or a no-tools draft turn): the last bucket is scored by the same rule, but
    the prompt does not name the tool, so it is reported apart.
  - a prefilled turn's reply is the prefill plus what the model wrote (OpenAICompatBackend.swift:885-890), so the
    scorer reads both.

The other ten v2 families (dictation, email, ...) are not in this set: their prompts are not the chat turn.

  GRUX_SRC_V3=... python3 recipe/s2/holdout_local.py        # counts, fingerprint, buckets, the not-sent list
"""
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import holdout_v2  # noqa: E402
import local_turn  # noqa: E402

FINGERPRINT_MODEL = "gruxai.com/grux/grux-local:4b"   # ABOUT_YOU names the model, so the fingerprint fixes one
MEMORY_BY_CTX = {8_192: 8, 16_384: 16, 32_768: 32}    # LocalTurn.contextTokens(memoryGB:), LocalTurn.swift:23-29


def cases():
    return [dict(c) for c in holdout_v2.cases() if c["family"] in local_turn.CHAT_FAMILIES]


def memory_for(num_ctx):
    if num_ctx not in MEMORY_BY_CTX:
        raise SystemExit(f"--set local sends this Mac's own window: --num-ctx is one of {sorted(MEMORY_BY_CTX)}")
    return MEMORY_BY_CTX[num_ctx]


def request_local(c, model, memory_gb=16):
    """The 3.3.1 request for case c, or {"not_sent": reason} when 3.3.1 answers it with no model."""
    return local_turn.chat_request_v331(c["utterance"], model, memory_gb)


def with_prefill(raw, prefill):
    """The reply the app shows: the prefilled start, then what the model wrote after it."""
    if not prefill:
        return raw
    out = json.loads(json.dumps(raw))
    msg = out.setdefault("message", {})
    msg["content"] = prefill + (msg.get("content") or "")
    return out


def score_local(c, raw, req):
    """(pass, why, what we read), by held-out v2's scorer, against the tools this turn offered."""
    return holdout_v2.score_v2(c, with_prefill(raw, req.get("prefill")), req["offered"])


def bucket(c, req):
    """Where a case lands: not_sent:<reason>, or for chat_tools offered / cut_line / not_named, else sent."""
    if "not_sent" in req:
        return "not_sent:" + req["not_sent"]
    if c["family"] != "chat_tools":
        return "sent"
    if c["tool"] in req["offered"]:
        return "offered"
    if req["cut"] and c["tool"] in req["cut"].split(": ", 1)[1].split(". ", 1)[0].split(", "):
        return "cut_line"
    return "not_named"


def fingerprint(cs, model=FINGERPRINT_MODEL, memory_gb=16):
    """Every case with the request it sends (or why it sends none)."""
    return hashlib.sha256(json.dumps([[c, request_local(c, model, memory_gb)] for c in cs], sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def report(cs, model=FINGERPRINT_MODEL):
    reqs = {c["id"]: request_local(c, model) for c in cs}
    reqs8 = {c["id"]: request_local(c, model, 8) for c in cs}
    b = collections.Counter(bucket(c, reqs[c["id"]]) for c in cs)
    sent = [c for c in cs if "not_sent" not in reqs[c["id"]]]
    paths = collections.Counter("stable" if reqs[c["id"]]["stable"] else "ranked: " + reqs[c["id"]]["why_ranked"]
                                for c in sent)
    flips = [c["id"] for c in sent if reqs8[c["id"]].get("offered") != reqs[c["id"]]["offered"]]
    print(f"{len(cs)} chat cases: {len(sent)} sent, {len(cs) - len(sent)} answered by 3.3.1 with no model")
    for k, n in sorted(b.items()):
        print(f"  {k:24s} {n}")
    print("sent turns by tool list:", dict(sorted(paths.items())))
    print(f"8 GB Mac against 16 GB: {len(flips)} sent turns offer different tools", flips[:10])
    for c in cs:
        k = bucket(c, reqs[c["id"]])
        if k.startswith("not_sent") or k in ("cut_line", "not_named"):
            print(f"  {k:24s} {c['id']}  {c.get('tool', '')}  {c['utterance'][:70]!r}")
    return b, paths, flips


if __name__ == "__main__":
    cs = cases()
    print(f"held-out local: {len(cs)} cases at Grux {local_turn.V331_TAG}, fingerprint {fingerprint(cs)[:16]} "
          f"({FINGERPRINT_MODEL}, 16 GB)")
    report(cs)
