#!/usr/bin/env python3
"""The back catalog (gruxai round 4, road B, pressed 2026-11-16): with base.json's backfill on, a library build last
updated before `since` is scored like a new one, after every new one, and a back-catalog win is reported, never
trained by the lane. Without the network: planted listing entries and candidates. Prints PASS or FAIL per check;
exit 1 on any FAIL.

  python3 recipe/autopilot/tests/test_backfill.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp()
os.environ.update({"GRUX_AUTOPILOT_HEARTBEAT": os.path.join(TMP, "hb.json"),
                   "GRUX_LOCAL_AUTOPILOT_RESULTS": os.path.join(TMP, "results")})
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import run  # noqa: E402
import watch  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


cfg = common.base_config()
since = cfg["since"]
on = {**cfg, "backfill": {"enabled": True}}
off = {**cfg, "backfill": {"enabled": False}}
old = {"name": "planted-old", "caps": ["tools"], "sizes": ["4b"], "cloud": False, "updated": "2025-01-01",
       "pulls": "1M", "description": ""}
new = {**old, "name": "planted-new", "updated": "2099-01-01"}
no_tools = {**old, "name": "planted-old-notools", "caps": []}

check("the shipped base.json turns the back catalog on", bool((cfg.get("backfill") or {}).get("enabled")),
      str(cfg.get("backfill")))
v, why, sizes = watch.judge_ollama_entry(old, off, since, cfg["family"])
check("backfill off: an old build stays baseline", v == "baseline", f"{v} {why}")
v, why, sizes = watch.judge_ollama_entry(old, on, since, cfg["family"])
check("backfill on: an old build is checked, marked backfill", v == "check" and why.startswith("backfill")
      and sizes == ["4b"], f"{v} {why!r} {sizes}")
v, why, _ = watch.judge_ollama_entry(new, on, since, cfg["family"])
check("backfill on: a new build is checked, not marked backfill", v == "check" and not why.startswith("backfill"),
      f"{v} {why!r}")
v, why, _ = watch.judge_ollama_entry(no_tools, on, since, cfg["family"])
check("backfill on: the filters still apply to an old build (no tools badge)", v == "filtered", f"{v} {why}")
v, why, _ = watch.judge_ollama_entry({**old, "name": cfg["family"]}, on, since, cfg["family"])
check("backfill on: the base family itself stays baseline", v == "baseline", f"{v} {why}")

head = cfg["headline_tier"]
cands = [{"id": "back-4b", "tier": head, "updated": "2025-06-01", "backfill": True},
         {"id": "new-9b", "tier": "9b", "updated": "2026-11-01"},
         {"id": "back-9b", "tier": "9b", "updated": "2025-12-01", "backfill": True},
         {"id": "new-4b", "tier": head, "updated": "2026-10-30"}]
order = [c["id"] for c in run.order_to_score(cands, head)]
check("every new build is scored before any back-catalog build", order == ["new-4b", "new-9b", "back-4b", "back-9b"],
      str(order))

check("a new build's win trains", run.may_train({"id": "n"}, "win", no_train=False))
check("a back-catalog win is never trained by the lane", not run.may_train({"id": "b", "backfill": True}, "win",
                                                                          no_train=False))
check("--no-train still stops a new build's win", not run.may_train({"id": "n"}, "win", no_train=True))
check("a lose never trains", not run.may_train({"id": "n"}, "lose", no_train=False))

print("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL")
sys.exit(1 if FAILS else 0)
