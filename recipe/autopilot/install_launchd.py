#!/usr/bin/env python3
"""Write (and with --load, load) the weekly LaunchAgent: Mondays at 3:00 AM, label com.dcj.grux-local-autopilot.

Paths are this machine's own, filled in when it runs, so the repo carries no home directory. The Daytona key is read
at run time from the Keychain by name (dcj-secret); no key is written into the plist.

  python3 recipe/autopilot/install_launchd.py [--python <venv python>] [--load]
"""
import argparse
import os
import plistlib
import subprocess

LABEL = "com.dcj.grux-local-autopilot"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=os.path.expanduser("~/.grux/autopilot/grux-local/venv/bin/python"))
    ap.add_argument("--secret-tool", default=os.path.expanduser("~/.local/bin/dcj-secret"))
    ap.add_argument("--load", action="store_true")
    a = ap.parse_args()
    log = os.path.expanduser("~/Library/Logs/grux-local-autopilot.log")
    plist = {
        "Label": LABEL,
        "ProgramArguments": [a.secret_tool, "run", "dcj/daytona/grux-api-key", "DAYTONA_API_KEY", "--",
                             a.python, os.path.join(HERE, "run.py")],
        "StartCalendarInterval": {"Weekday": 1, "Hour": 3, "Minute": 0},
        "WorkingDirectory": os.path.dirname(os.path.dirname(HERE)),
        "StandardOutPath": log, "StandardErrorPath": log,
        "EnvironmentVariables": {"PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"},
    }
    path = os.path.expanduser(f"~/Library/LaunchAgents/{LABEL}.plist")
    with open(path, "wb") as f:
        plistlib.dump(plist, f)
    print(f"wrote {path}")
    if a.load:
        uid = os.getuid()
        subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LABEL}"], capture_output=True)
        r = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", path], capture_output=True, text=True)
        print(f"bootstrap exit {r.returncode} {r.stderr.strip()}")


if __name__ == "__main__":
    main()
