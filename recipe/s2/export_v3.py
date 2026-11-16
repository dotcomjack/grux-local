#!/usr/bin/env python3
"""train.jsonl from a teacher's answers to the v3 training bank (eval_run.py --set train_v3).

A job is kept only when holdout_v3's scorer passed the teacher's answer to every step. Each step becomes one row in
box_train.py's shape: the exact request the teacher saw at that step (system prompt, earlier turns, the scripted calls
and results of the steps before it, every tool), and the teacher's raw answer as the target. The request is rebuilt
from the bank, which is deterministic, so nothing but the answer comes from the model.

  python3 recipe/s2/export_v3.py --run results/teacher-v3-<stamp> [--out train.jsonl]
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import holdout_v3  # noqa: E402
import train_bank_v3  # noqa: E402
import train_prompt  # noqa: E402


def kept_answers(run_dir):
    """case id -> the first passing row that carries every step's raw answer, across all pass files."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*-train_v3*.jsonl"))):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("pass") and r["case"] not in out and r.get("raw"):
                out[r["case"]] = r
    return out


HOME = re.compile(r"(?<![A-Za-z0-9._~/-])/(Users|home)/([A-Za-z0-9._-]+)")   # a path that starts there, not ~/x/home/y
REAL_HOMES = {("Users", "Shared")}   # /Users/Shared is on every Mac


def invented_home(rows):
    """True when a target names a home folder (/Users/<name>, /home/<name>) that no input of its step names: the
    teacher guessed the user's account name (measured 2026-11-16: 158 of 6,735 kept jobs, /Users/username,
    /Users/jax, /Users/jack ...). The scorer accepts the path, but on a real Mac it points at nobody's folder."""
    def homes(text):   # whole names (/Users/dan is not /Users/dana); a sentence's closing period is not the name's
        return {(m.group(1), m.group(2).rstrip(".-_")) for m in HOME.finditer(text)}
    for r in rows:
        seen = homes(json.dumps(r["messages"]) + json.dumps(r["tools"]))
        if homes(json.dumps(r["target"])) - seen - REAL_HOMES:
            return True
    return False


def rows_for(case, answer, model="teacher"):
    """One training row per step, or [] when the answer does not cover every step."""
    raws = answer["raw"]
    if len(raws) != len(case["steps"]):
        return []
    out = []
    for k, raw in enumerate(raws):
        body = train_prompt.student_body(case, k)
        out.append({"id": f"{case['id']}#{k}", "family": case["family"], "messages": body["messages"],
                    "tools": body["tools"], "target": {"content": raw.get("content") or "",
                                                       "tool_calls": raw.get("tool_calls") or []}})
    return out


def export(run_dir, out_path):
    """An answer is paired only with the exact job it answered and the exact tools it was offered: its row's job_sha
    and tools_sha must equal the job's now. A row without them (taught on another prompt) is never exported. A job
    whose answer invents a home folder (invented_home) is never exported either."""
    bank = train_bank_v3.kept()
    by_id = {c["id"]: c for c in bank}
    answers = kept_answers(run_dir)
    fams, rows_n, short, changed, gone, hinted, invented = {}, 0, 0, 0, 0, 0, 0
    with open(out_path, "w") as f:
        for cid, ans in answers.items():
            case = by_id.get(cid)
            if case is None:
                gone += 1
                continue
            if (ans.get("job_sha") != holdout_v3.case_sha([case])
                    or ans.get("tools_sha") != train_prompt.tools_sha(case)):
                changed += 1
                continue
            rows = rows_for(case, ans)
            if not rows:
                short += 1
                continue
            if invented_home(rows):
                invented += 1
                continue
            for r in rows:
                f.write(json.dumps(r) + "\n")
            rows_n += len(rows)
            hinted += bool(ans.get("hint"))
            fams[case["family"]] = fams.get(case["family"], 0) + 1
    total = {}
    for c in by_id.values():
        total[c["family"]] = total.get(c["family"], 0) + 1
    summary = {"jobs_kept": sum(fams.values()), "jobs_in_bank": len(by_id), "rows": rows_n,
               "dropped_short_answers": short, "dropped_job_changed": changed, "dropped_job_gone": gone,
               "dropped_invented_home": invented,
               "hinted_jobs": hinted,
               "families": {k: {"kept": fams.get(k, 0), "bank": n, "yield": round(fams.get(k, 0) / n, 3)}
                            for k, n in sorted(total.items())}}
    json.dump(summary, open(os.path.join(os.path.dirname(os.path.abspath(out_path)), "export.json"), "w"), indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    s = export(a.run, a.out or os.path.join(a.run, "train.jsonl"))
    print(json.dumps(s, indent=1))
    print(f"EXPORT: {s['jobs_kept']} of {s['jobs_in_bank']} jobs kept, {s['rows']} rows")


if __name__ == "__main__":
    main()
