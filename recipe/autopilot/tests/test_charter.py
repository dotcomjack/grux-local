"""No swing charter, no box: with GRUX_SWING_CHARTER unset, score.py queues every model on the swing backlog and
exits 6 without reserving or creating anything (owner's rule, 2026-11-14: never a GPU swing on one or two jobs).
The meter and the backlog are fakes written to a temp folder. Pass another recipe folder as argv[1] to run it
against an older copy (it fails there: the old score.py reserves)."""
import json, os, subprocess, sys, tempfile
R = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with tempfile.TemporaryDirectory() as d:
    meter_log, backlog = os.path.join(d, "meter.log"), os.path.join(d, "backlog.jsonl")
    open(os.path.join(d, "meter.py"), "w").write(
        "import sys\nopen(%r,'a').write(' '.join(sys.argv[1:])+'\\n')\nprint('rid0')\n" % meter_log)
    open(os.path.join(d, "charter.py"), "w").write(
        "import sys,json\nopen(%r,'a').write(json.dumps(sys.argv[1:])+'\\n')\nprint('b-test')\n" % backlog)
    env = {**os.environ, "DAYTONA_SPEND_TOOL": os.path.join(d, "meter.py"),
           "SWING_CHARTER_TOOL": os.path.join(d, "charter.py"), "GRUX_LOCAL_AUTOPILOT_STATE": d,
           "GRUX_LOCAL_AUTOPILOT_RESULTS": d, "DAYTONA_API_KEY": "unused"}
    env.pop("GRUX_SWING_CHARTER", None)
    r = subprocess.run([sys.executable, os.path.join(R, "autopilot", "score.py"), "--models", "a:9b,b:4b"],
                       env=env, capture_output=True, text=True, timeout=120)
    queued = [json.loads(l) for l in open(backlog)] if os.path.exists(backlog) else []
    reserved = open(meter_log).read() if os.path.exists(meter_log) else ""
    print("exit", r.returncode, "| queued", len(queued), "| meter calls", len(reserved.splitlines()))
    ok = r.returncode == 6 and len(queued) == 2 and all(q[3] == "measure" for q in queued) and not reserved
    if not ok:
        print(r.stdout[-600:], r.stderr[-600:])
sys.exit(0 if ok else 1)
