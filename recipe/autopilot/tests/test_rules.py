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

check("empty file: refused", not data_scan.check(jsonl([]), vocab)[0])
check("only blank lines: refused", not data_scan.check(jsonl(["", ""]), vocab)[0])
bad_utf8 = os.path.join(TMP, "bad-utf8.jsonl")
open(bad_utf8, "wb").write(json.dumps(clean[0]).encode() + b"\n" + b'{"id": "x", "c": "\xff\xfe"}\n')
ok, msg = data_scan.check(bad_utf8, vocab)
check("invalid UTF-8: a clean refusal", not ok and "unreadable" in msg, msg)
bad_vocab = os.path.join(TMP, "bad-vocab.json")
json.dump({"banned": ["ok", 7, None]}, open(bad_vocab, "w"))
ok, msg = data_scan.check(jsonl(clean), bad_vocab)
check("vocabulary with non-strings: a clean refusal", not ok and "vocabulary" in msg, msg)

# --- the installer's filter: the thing that keeps personal data out of a box
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("installer", os.path.join(os.path.dirname(HERE), "install_launchd.py"))
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
installer.TRAIN = os.path.join(TMP, "pinned", "train.jsonl")
src = jsonl(clean + [{"id": "drop-me", "content": "call PlantedCo"},
                     {"content": "no id, still names planted person"},     # a row with no id
                     ["plantedco in a list row"],
                     {"id": "keep", "content": "water the plants"}])
kept, dropped = installer.install_train_data(src, vocab)
pinned = [json.loads(l) for l in open(installer.TRAIN)]
check("installer drops identity rows by line, ids or not", (kept, dropped) == (3, 3)
      and [r["id"] for r in pinned] == ["c1", "c2", "keep"], f"{kept} kept, {dropped} dropped, {pinned}")
check("installed file scans clean", data_scan.check(installer.TRAIN, vocab)[0])
before = open(installer.TRAIN).read()
for name, rows in {"a secret row": clean + [{"id": "s", "content": "AKIA" + "Z" * 16}],
                   "an unreadable row": clean + ["{broken"],
                   "every row dropped": [{"id": "d", "content": "plantedco"}]}.items():
    try:
        installer.install_train_data(jsonl(rows), vocab)
        check(f"installer refuses {name}", False, "it installed")
    except SystemExit as e:
        check(f"installer refuses {name}, pinned file untouched", open(installer.TRAIN).read() == before, str(e))

# --- callers that make boxes refuse with no meter, and an empty reservation id
import subprocess  # noqa: E402
RECIPE = os.path.dirname(os.path.dirname(HERE))
try:
    r = subprocess.run([sys.executable, os.path.join(RECIPE, "s2", "sweep.py"), "--sweep", os.path.join(TMP, "sw"),
                        "--data", jsonl(clean)], capture_output=True, text=True, timeout=20)
    check("sweep.py without --meter: refuses, launches nothing",
          r.returncode != 0 and "meter" in r.stderr and not os.path.exists(os.path.join(TMP, "sw", "4b-r16-e1")), r.stderr)
except subprocess.TimeoutExpired:
    check("sweep.py without --meter: refuses, launches nothing", False, "it kept launching runs for 20 s")
fake = os.path.join(TMP, "empty-meter.py")
open(fake, "w").write("import sys\nsys.exit(0)\n")                    # exits 0 and prints no id
import meter  # noqa: E402
try:
    meter.reserve(fake, "grux-local", 1, "H200")
    check("an empty reservation id is refused", False, "it returned an id")
except meter.Refused as e:
    check("an empty reservation id is refused", "no id" in str(e), str(e))

# every box made anywhere in the recipe is made with its own reservation (a sweep over the source)
import re as _re  # noqa: E402
calls = []
for root, _, files in os.walk(RECIPE):
    for f in files:
        if f.endswith(".py") and "tests" not in root:
            src_text = open(os.path.join(root, f)).read()
            for m in _re.finditer(r"\b(OllamaBox|TrainBox)\(", src_text):
                if src_text[max(0, m.start() - 6):m.start()] == "class ":
                    continue
                depth, j = 1, m.end()
                while depth and j < len(src_text):
                    depth += {"(": 1, ")": -1}.get(src_text[j], 0)
                    j += 1
                calls.append((f, "reservation=" in src_text[m.end():j]))
check(f"every box constructor call passes a reservation ({len(calls)} calls)",
      len(calls) >= 5 and all(ok for _, ok in calls), [c for c in calls if not c[1]])
log = os.path.join(TMP, "meter-log.txt")
open(fake, "w").write("import sys\nopen(%r, 'a').write(' '.join(sys.argv[1:3]) + '\\n')\n"
                      "print('rid-%%d' %% len(open(%r).read().splitlines()))\n" % (log, log))
for attempt in range(3):           # three boxes, the middle one dies: three reservations, three settles
    try:
        with meter.Reserved(fake, "grux-local", 1, "H200") as rid:
            if attempt == 1:
                raise RuntimeError("planted eviction")
    except RuntimeError:
        pass
lines = open(log).read().split("\n")
check("each box its own reservation, settled even when the box dies",
      sum(l.startswith("reserve") for l in lines) == 3 and sum(l.startswith("settle") for l in lines) == 3, lines)

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
