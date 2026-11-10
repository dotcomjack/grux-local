#!/usr/bin/env python3
"""The lane's rules without a box or the network: the training data scan, the ready item's created_at, UTC stamps
read as local dates, the dead-man's verdicts and the heartbeat alerts. Prints PASS or FAIL per check; exit 1 on any
FAIL. Every planted secret below is built at run time from pieces, never written whole in this file.

  python3 recipe/autopilot/tests/test_rules.py
"""
import datetime as dt
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp()
os.environ.update({"GRUX_AUTOPILOT_READY": os.path.join(TMP, "ready"),
                   "GRUX_AUTOPILOT_HEARTBEAT": os.path.join(TMP, "hb.json"),
                   "GRUX_LOCAL_AUTOPILOT_RESULTS": os.path.join(TMP, "results")})
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import data_scan  # noqa: E402
import deadman  # noqa: E402
import run  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def jsonl(rows):
    p = os.path.join(TMP, f"d{len(os.listdir(TMP))}.jsonl")
    with open(p, "w") as f:
        for r in rows:
            f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
    return p


# --- data scan
vocab = os.path.join(TMP, "vocab.txt")
open(vocab, "w").write("# planted terms\nplanted person\nplantedco\n")
clean = [{"id": "c1", "messages": [{"role": "user", "content": "add a task to call the plumber"}]},
         {"id": "c2", "messages": [{"role": "user", "content": "what is 15 percent of 80"}]}]
check("clean data passes", data_scan.check(jsonl(clean), vocab)[0])
secrets = {"aws": "AKIA" + "Q" * 16, "github": "ghp" + "_" + "a1B2" * 9, "slack": "xox" + "b-" + "1234567890-abc",
           "openai": "sk" + "-" + "proj" + "x" * 30, "stripe": "sk" + "_live_" + "a" * 24,
           "key block": "-----BEGIN " + "RSA PRIVATE KEY-----", "hex": "a1" * 24}
for name, value in secrets.items():
    ok, msg = data_scan.check(jsonl(clean + [{"id": "bad", "messages": [{"role": "user", "content": "x " + value}]}]), vocab)
    check(f"secret shape refused: {name}", not ok and "bad" in msg and value not in msg, msg)
for name, row in {"identity term": {"id": "bad", "messages": [{"role": "user", "content": "email PlantedCo today"}]},
                  "two-word term, double space": {"id": "bad", "content": "ask planted   person"},
                  "term in a dict key": {"id": "bad", "plantedco": 1},
                  "row that is a list": ["planted person"]}.items():
    ok, msg = data_scan.check(jsonl(clean + [row]), vocab)
    check(f"identity refused: {name}", not ok and "planted person" not in msg.lower(), msg)
ok, msg = data_scan.check(jsonl(clean + ["{not json"]), vocab)
check("unreadable row refused", not ok, msg)
check("blank line is not a crash", data_scan.check(jsonl(clean + [""]), vocab)[0])
check("whole word only (plantedcompany passes)",
      data_scan.check(jsonl([{"id": "w", "content": "plantedcompany"}]), vocab)[0])
check("no vocabulary: refused", not data_scan.check(jsonl(clean), "")[0])
check("missing required vocabulary: refused", not data_scan.check(jsonl(clean), os.path.join(TMP, "none.txt"))[0])
check("missing optional vocabulary is skipped",
      data_scan.check(jsonl(clean), vocab + ":?" + os.path.join(TMP, "none.txt"))[0])

# --- ready item keeps its first created_at; stamps are UTC Z
p = common.ready_item("t", "k", "T", "w", ["/a"], None, [])
first = json.load(open(p))["created_at"]
check("created_at is UTC with Z", first.endswith("Z") and "T" in first, first)
os.utime(p)
json.dump({**json.load(open(p)), "created_at": "2026-01-01T00:00:00Z"}, open(p, "w"))
common.ready_item("t", "k", "T", "w", ["/a"], None, [])
check("unchanged item keeps created_at", json.load(open(p))["created_at"] == "2026-01-01T00:00:00Z")
common.ready_item("t", "k", "T", "changed", ["/a"], None, [])
check("changed item gets a new created_at", json.load(open(p))["created_at"] != "2026-01-01T00:00:00Z")

# --- the dead-man
today = dt.date.today().isoformat()
three_am = dt.datetime.now().replace(hour=3, minute=5, second=0, microsecond=0).astimezone(dt.timezone.utc)
stamp = three_am.strftime("%Y-%m-%dT%H:%M:%SZ")
check("a UTC stamp reads as its local date", common.local_date(stamp) == today, common.local_date(stamp))
os.makedirs(common.RESULTS, exist_ok=True)


def beat(o):
    json.dump(o, open(common.HEARTBEAT, "w"))


beat({"last_run_at": stamp, "ok": True, "summary": "x"})
check("no watch report: red", any("no watch report" in b for b in deadman.problems(today)))
open(os.path.join(common.RESULTS, f"watch-{today}.json"), "w").write("{}")
check("ran ok today: green", deadman.problems(today) == [])
beat({"last_run_at": "2026-01-05T08:00:00Z", "ok": True})
check("last ran last week: red", deadman.problems(today) != [])
beat({"last_run_at": stamp, "ok": False, "summary": "planted"})
check("ran but not ok: red", any("not ok" in b for b in deadman.problems(today)))
beat({"last_run_at": stamp, "ok": True, "running": {"pid": 1, "started_at": stamp}})
check("killed mid-run after an earlier ok: red", any("never finished" in b for b in deadman.problems(today)))
check("hung past the wait: not reported as killed", deadman.problems(today, hung=True) == [])
check("a reused pid is not a run", not deadman.alive(os.getpid()))

# --- heartbeat alerts
run.week_run = lambda: {"steps": [{"score_exit": 3}, {"train": "x:4b", "exit": 2}], "alerts": ["BASE ROW STUCK: t"],
                        "watch": {"candidates": 1}, "scored": [], "decisions": []}
run.standing_verdicts = lambda: {"x:4b": "tie"}
run.main()
hb = json.load(open(common.HEARTBEAT))
check("heartbeat red on a meter refusal, a failed train and a stuck base",
      hb["ok"] is False and "spend meter" in hb["summary"] and "train.py" in hb["summary"]
      and "BASE ROW STUCK" in hb["summary"] and "x:4b tie" in hb["summary"] and "running" not in hb, hb["summary"])
run.week_run = lambda: {"steps": [], "watch": {"candidates": 1}, "scored": [], "decisions": []}
run.main()
hb = json.load(open(common.HEARTBEAT))
check("heartbeat green on a quiet week", hb["ok"] is True and hb["last_run_at"].endswith("Z"), hb)

print(f"{'FAIL' if FAILS else 'PASS'}: {len(FAILS)} failed")
sys.exit(1 if FAILS else 0)
