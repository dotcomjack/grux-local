#!/usr/bin/env python3
"""Dead-man check for the weekly run: Mondays at 6:00 AM, three hours after the 3:00 AM job.

Red (Telegram, alarm prefix) unless today's watch report exists in the results home AND the heartbeat says the run
finished today with ok true. A run still going at 6:00 AM (a training week runs for hours) is waited for, up to
WAIT_HOURS, and judged when it ends; a page that cannot be sent is retried for about five minutes. It runs from
the installed lane, separately from run.py, so a run.py that never started (a missing file, a launchd error, a dead
Mac at 3:00 AM that woke later) still pages.

  python3 recipe/autopilot/deadman.py [--dry-run] [--date YYYY-MM-DD]
  python3 recipe/autopilot/deadman.py --test-send     one recovery-prefixed line, to prove the page path
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

ENV_FILE = os.path.expanduser(os.environ.get("DCJ_TELEGRAM_ENV", "~/.claude/telegram/.env"))
RED, OK = "\U0001F6A8", "\u2705"
WAIT_HOURS = 8


def page(text, tries=10, gap=30):
    for i in range(tries):
        try:
            return send(text)
        except (OSError, ValueError) as e:  # asleep at 3:00 AM, woken before the network: try again
            print(f"page attempt {i + 1} failed: {e}", flush=True)
            time.sleep(gap)
    raise SystemExit(f"could not page after {tries} attempts: {text}")


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def wait_for_run(date, poll=300):
    """If today's run is still going, wait for it to end (at most WAIT_HOURS). Returns what was waited for."""
    t0 = time.time()
    while time.time() - t0 < WAIT_HOURS * 3600:
        try:
            run = json.load(open(common.HEARTBEAT)).get("running") or {}
        except (OSError, ValueError):
            return None
        if (run.get("started_at") or "")[:10] != date or not alive(run.get("pid")):
            return None if time.time() - t0 < 1 else f"waited {round((time.time() - t0) / 60)} minutes for the run"
        print(f"run pid {run.get('pid')} started {run.get('started_at')} is still going; waiting", flush=True)
        time.sleep(poll)
    return f"still running after {WAIT_HOURS} hours"


def send(text):
    if not os.path.exists(ENV_FILE):
        raise SystemExit(f"cannot page: {ENV_FILE} is missing")
    env = dict(l.strip().split("=", 1) for l in open(ENV_FILE) if "=" in l)
    tok, chat = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        raise SystemExit("cannot page: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing")
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    with urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=20) as r:
        return json.loads(r.read()).get("ok")


def problems(date):
    out = []
    watch = os.path.join(common.RESULTS, f"watch-{date}.json")
    if not os.path.exists(watch):
        out.append(f"no watch report for {date} in {common.RESULTS.replace(os.path.expanduser('~'), '~', 1)}")
    try:
        beat = json.load(open(common.HEARTBEAT))
    except (OSError, ValueError) as e:
        return out + [f"no readable heartbeat ({type(e).__name__})"]
    when = (beat.get("last_run_at") or "")[:10]
    if when != date:
        out.append(f"heartbeat last ran {when or 'never'}, not {date}")
    elif not beat.get("ok"):
        out.append(f"the run finished but not ok: {beat.get('summary', '')[:300]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-send", action="store_true")
    ap.add_argument("--date", default="")
    a = ap.parse_args()
    if a.test_send:
        print("sent" if page(f"{OK} Grux Local autopilot: dead-man check installed; it pages here if the Monday "
                             "3:00 AM run has not reported by 6:00 AM. (test of the page path)") else "send failed")
        return
    date = a.date or dt.date.today().isoformat()
    waited = None if a.dry_run else wait_for_run(date)
    bad = problems(date)
    if waited and waited.startswith("still running"):
        bad.append(f"the run is {waited}")
    if not bad:
        print(f"OK: the Grux Local weekly run reported for {date}")
        return
    text = f"{RED} Grux Local autopilot: the Monday run did not report. " + "; ".join(bad)
    print(("WOULD PAGE: " if a.dry_run else "PAGING: ") + text)
    if not a.dry_run:
        page(text)
    sys.exit(1)


if __name__ == "__main__":
    main()
