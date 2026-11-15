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
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import holdout_v3  # noqa: E402
import train_bank_v3  # noqa: E402


def run_bank_sha(run_dir):
    """The bank sha the teacher run recorded (manifest.json suite_sha), or None."""
    try:
        return json.load(open(os.path.join(run_dir, "manifest.json"))).get("suite_sha")
    except (OSError, ValueError):
        return None


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


def rows_for(case, answer, model="teacher"):
    """One training row per step, or [] when the answer does not cover every step."""
    raws = answer["raw"]
    if len(raws) != len(case["steps"]):
        return []
    out = []
    for k, raw in enumerate(raws):
        body = holdout_v3.request_v3(case, model, k)["body"]
        out.append({"id": f"{case['id']}#{k}", "family": case["family"], "messages": body["messages"],
                    "tools": body["tools"], "target": {"content": raw.get("content") or "",
                                                       "tool_calls": raw.get("tool_calls") or []}})
    return out


def export(run_dir, out_path):
    """An answer is paired only with the exact job it answered: its row's job_sha must equal the job's sha now, or,
    for a row written before job_sha existed, the run's whole-bank sha must equal the bank's sha now."""
    bank = train_bank_v3.kept()
    by_id = {c["id"]: c for c in bank}
    same_bank = run_bank_sha(run_dir) == holdout_v3.case_sha(bank)
    answers = kept_answers(run_dir)
    fams, rows_n, short, changed, gone = {}, 0, 0, 0, 0
    with open(out_path, "w") as f:
        for cid, ans in answers.items():
            case = by_id.get(cid)
            if case is None:
                gone += 1
                continue
            if (ans["job_sha"] != holdout_v3.case_sha([case])) if "job_sha" in ans else not same_bank:
                changed += 1
                continue
            rows = rows_for(case, ans)
            if not rows:
                short += 1
                continue
            for r in rows:
                f.write(json.dumps(r) + "\n")
            rows_n += len(rows)
            fams[case["family"]] = fams.get(case["family"], 0) + 1
    total = {}
    for c in by_id.values():
        total[c["family"]] = total.get(c["family"], 0) + 1
    summary = {"jobs_kept": sum(fams.values()), "jobs_in_bank": len(by_id), "rows": rows_n,
               "dropped_short_answers": short, "dropped_job_changed": changed, "dropped_job_gone": gone,
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
