#!/usr/bin/env python3
"""The oracle: Grux v3.3.1's own Swift, compiled (build.sh), asked about chat messages.

Two uses:
  - the asks a local turn depends on and local_turn.py does not re-port (which fast path answers a message with no
    model, the draft, list and pasted-text asks with their notes and prefilled starts, the weather note, the
    volatile sections): `asks()`, frozen for the held-out turns in recipe/local_turn_v331_asks.json;
  - a check of local_turn.py's 3.3.1 port: `assemble()` runs the lean branch of assemblePendingContext (main.swift,
    a verbatim copy of ChatService.swift:1119-1192 and 1226-1253 over the compiled LocalTurn and ToolRelevance) on
    the system blocks Python built, so the two can be compared turn for turn.

  GRUX_SRC_V3=... python3 recipe/grux_oracle/oracle.py --write-asks   # rewrite the frozen table

Needs swiftc (Xcode or the command line tools) and GRUX_SRC_V3 at v3.3.1. No model, no network.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.dirname(HERE)
sys.path.insert(0, RECIPE)
import local_turn  # noqa: E402

# suite.NOW, "Thursday, Oct 22 2026 · 2:00 PM EDT": the date asks are read on, on a Mac in New York.
NOW = datetime.datetime(2026, 10, 22, 18, 0, tzinfo=datetime.timezone.utc).timestamp()
TZ = "America/New_York"
# Messages the tests use beyond the held-out turns (the Swift tests' own asks, LocalTurnStableToolsTests.swift,
# LocalTurnPrefixTests.swift, LocalChipToolsTests.swift, FirstRunWalkW2dTests.swift).
TEST_UTTERANCES = [
    "what is on my plate today?", "add a task to call the dentist tomorrow", "what's the tallest mountain in Africa",
    "add a task to buy milk", "add a task to buy milk" + " and bread" * 40, "add a task to buy milk" + " and bread" * 260,
    "word " * 8_000, "hi", "what is on my calendar tomorrow", "open github.com", "write a list",
    "Undo `rm notes.txt` from 5:00 PM", "Why did you refuse `afplay /tmp/x.aiff`?", "What did you run today? (3 commands)",
    "what is two plus two",
    # draft and list asks (DraftAskTests, ListAskTests): no held-out turn is one, so the prefill path is checked here
    "write a text to Priya saying I will be ten minutes late", "draft an email to my landlord asking to fix the heater",
    "make a packing list for a beach weekend", "write a list of ten camping essentials",
]


def available():
    """Why the oracle cannot run here, or None."""
    if not os.environ.get("GRUX_SRC_V3"):
        return "GRUX_SRC_V3 is not set"
    if subprocess.run(["sh", "-c", "command -v swiftc"], capture_output=True).returncode:
        return "no swiftc on this machine"
    return None


def _out_dir():
    h = hashlib.sha256()
    for f in sorted(os.listdir(HERE)):
        if f.endswith((".swift", ".py", ".sh")):
            h.update(f.encode() + open(os.path.join(HERE, f), "rb").read())
    h.update(os.path.realpath(os.environ["GRUX_SRC_V3"]).encode())
    return os.path.join(os.path.expanduser("~/.cache/grux-oracle"), h.hexdigest()[:16])


def binary():
    out = _out_dir()
    if not os.path.exists(os.path.join(out, "oracle")):
        local_turn.all_tools_v331()   # refuses a tree that is not v3.3.1
        subprocess.run(["sh", os.path.join(HERE, "build.sh"), out], check=True)
    return os.path.join(out, "oracle")


def source_sha():
    """sha256 of the Swift the oracle compiled (after prep.py), so a table names the source it came from."""
    binary()
    src = os.path.join(_out_dir(), "src")
    h = hashlib.sha256()
    for f in sorted(os.listdir(src)):
        h.update(f.encode() + open(os.path.join(src, f), "rb").read())
    return h.hexdigest()


def run(turns):
    """turns: [{"utterance", "system_blocks": [{"text", "cached"}], "window"}]. Returns the oracle's answer."""
    tools = [{"name": t["name"], "description": t["description"], "schema": t["schema"]}
             for t in local_turn.all_tools_v331()]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump({"now": NOW, "tools": tools, "turns": turns}, f, ensure_ascii=False)
        path = f.name
    try:
        r = subprocess.run([binary(), path], capture_output=True, env={**os.environ, "TZ": TZ}, check=True)
    finally:
        os.remove(path)
    return json.loads(r.stdout.decode("utf-8"))


ASK_KEYS = ("fast", "pim", "greeting", "confused", "no_tools", "notes", "prefill", "sections")


def asks(utterances):
    """{utterance: what 3.3.1's own classifiers say about it}. None of these depend on the system text."""
    blocks = [{"text": "x", "cached": True}, {"text": "y", "cached": False}]
    out = run([{"utterance": u, "system_blocks": blocks, "window": 16_384} for u in utterances])
    return {u: {k: t[k] for k in ASK_KEYS} for u, t in zip(utterances, out["turns"])}


def held_out_utterances():
    sys.path.insert(0, os.path.join(RECIPE, "s2"))
    import holdout_v2
    return [c["utterance"] for c in holdout_v2.cases() if c["family"] in local_turn.CHAT_FAMILIES]


def write_asks():
    us = list(dict.fromkeys(held_out_utterances() + TEST_UTTERANCES))
    table = {"tag": local_turn.V331_TAG, "commit": local_turn.V331_COMMIT, "now": NOW, "tz": TZ,
             "source_sha256": source_sha(), "asks": asks(us)}
    with open(local_turn.ASKS_PATH, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {len(table['asks'])} asks to {local_turn.ASKS_PATH}")


if __name__ == "__main__":
    why = available()
    if why:
        sys.exit("oracle unavailable: " + why)
    if "--write-asks" in sys.argv:
        write_asks()
    else:
        print(__doc__)
