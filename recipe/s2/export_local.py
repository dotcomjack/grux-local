#!/usr/bin/env python3
"""Training rows from a model's own samples of the train_local jobs (eval_run.py --set train_local --passes K).

Rejection sampling. A sample is kept only when held-out local's scorer passed it (holdout_local.score_local:
holdout_v2.score_v2 against the tools the turn offered) and the model finished it (done_reason "stop": a reply cut off
at num_predict would be trained as a finished one), and a job keeps at most --per-case distinct replies (the
same words, normalized, and the same tool calls with the same arguments are one reply). Each kept reply is one row in
box_train.py's shape, the shape export_v3.py and gen_data.py write: the request the model answered, rebuilt from
s2/train_local.py (deterministic) and compared with the request the sample recorded, and the model's reply as the
target. A sample is paired only with the job it answered (job_sha) and the exact messages and tools it was sent.

Jobs with no passing sample are where the model is weakest: they are listed by family and not trained on. Every row
is checked before it is written, the reply included: contamination.TrainGate (a held-out v4 text, a v4 clone, a v3
or v4 frame) and data_scan's key shapes and identity terms; a flagged row is dropped and counted by kind. The file then
goes through the two gates a launcher reads: contamination.gate_file writes the gated copy and its .gate.json (it
must drop nothing, the rows were gated already) and data_scan.check must call the gated copy clean; either failing
is exit 1.

The rows of one model's samples only: rejection sampling trains a model on its own replies, so samples from two
models in one run are refused.

  GRUX_DATA_SCAN_VOCAB=<file>[:<file>] python3 recipe/s2/export_local.py --run <eval_run dir> [--per-case 2]
      [--out <dir>]   writes <out>/train_local.jsonl, train_local.gated.jsonl (+ .gate.json), export_local.json
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import contamination as ct  # noqa: E402
import data_scan  # noqa: E402
import suite  # noqa: E402
import train_local  # noqa: E402

PER_CASE = 2


def samples(run_dir):
    """Every scored or skipped sample row in the run's train_local files, in file then line order."""
    out = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*-train_local*.jsonl"))):
        for n, line in enumerate(open(path, encoding="utf-8"), 1):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") in ("scored", "skipped"):
                out.append((os.path.basename(path), n, r))
    return out


def calls(reply):
    """The reply's tool calls as box_train.render_target reads them: [{"name", "arguments"}], the arguments read as the
    scorer reads them (suite._args: a JSON string becomes its object), so a row never carries a string render_target
    cannot walk."""
    out = []
    for x in reply.get("tool_calls") or []:
        f = x.get("function") or x
        out.append({"name": f.get("name"), "arguments": suite._args({"function": f})})
    return out


def reply_key(reply):
    """One key per distinct reply: its words normalized (or, when no Latin letter or digit is left, its stripped text),
    and each tool call with its arguments."""
    text = reply.get("content") or ""
    return suite._norm(text) or text.strip(), json.dumps(calls(reply), sort_keys=True, ensure_ascii=False)


def export(run_dir, out_dir, per_case=PER_CASE, vocab=""):
    terms = data_scan.load_terms(vocab)
    if not terms:
        raise SystemExit("no identity vocabulary (GRUX_DATA_SCAN_VOCAB or --vocab): refusing to export rows no scan read")
    word = data_scan.matcher(terms)
    paths = {k: os.path.join(out_dir, k) for k in ("train_local.jsonl", "train_local.gated.jsonl",
                                                    "train_local.gated.jsonl.gate.json", "export_local.json")}
    if any(os.path.exists(p) for p in paths.values()):
        raise SystemExit(f"refused: an export already exists in {out_dir}; export writes new files only")
    every = train_local.cases()
    gone = train_local.not_sent(every)
    by_id = {c["id"]: c for c in every if c["id"] not in gone}
    rows_in = samples(run_dir)
    models = sorted({r.get("model") for _, _, r in rows_in if r.get("status") == "scored"})
    if len(models) > 1:
        raise SystemExit(f"refused: samples from {len(models)} models ({models}); rejection sampling trains a model on "
                         "its own replies, so export one model's run")
    gate = ct.TrainGate()
    drop = {"job_gone": 0, "truncated": 0, "job_changed": 0, "request_changed": 0, "prefilled": 0, "duplicate": 0,
            "over_cap": 0, "gate": {}, "data_scan": {}}
    seen, kept, sampled, passing, rebuilt = {}, {}, {}, {}, {}
    out_rows = []
    for _, _, r in sorted(rows_in, key=lambda x: (x[2].get("seed") or 0, x[0], x[1])):
        case = by_id.get(r["case"])
        if case is None:
            drop["job_gone"] += 1
            continue
        if r.get("status") != "scored":   # the server never answered it: no reply, so not a sample of the model
            continue
        sampled[case["id"]] = sampled.get(case["id"], 0) + 1
        if not r.get("pass"):
            continue
        passing[case["id"]] = passing.get(case["id"], 0) + 1
        if r.get("done_reason") != "stop":   # cut off at num_predict: render_target would train it as finished
            drop["truncated"] += 1
            continue
        if r.get("job_sha") != train_local.case_sha([case]):
            drop["job_changed"] += 1
            continue
        key = (case["id"], r["model"])
        if key not in rebuilt:
            rebuilt[key] = train_local.request_local(case, r["model"])
        req = rebuilt[key]
        sent = r.get("request") or {}
        if "body" not in req or sent.get("messages") != req["body"]["messages"] or sent.get("tools") != req["body"].get(
                "tools"):
            drop["request_changed"] += 1
            continue
        if req.get("prefill") is not None or req["body"]["messages"][-1]["role"] != "user":
            drop["prefilled"] += 1   # box_train opens a new assistant turn after the last message: no continued turn
            continue
        k = reply_key(r["reply"])
        if k in seen.setdefault(case["id"], set()):
            drop["duplicate"] += 1
            continue
        seen[case["id"]].add(k)
        if kept.get(case["id"], 0) >= per_case:
            drop["over_cap"] += 1
            continue
        row = {"id": f"{case['id']}#{kept.get(case['id'], 0)}", "family": case["family"],
               "messages": req["body"]["messages"], "tools": req["body"].get("tools"),
               "target": {"content": r["reply"].get("content") or "", "tool_calls": calls(r["reply"])}}
        hit = gate.why(*ct.row_items(row))
        if hit:
            kind = hit[0].split(":")[0].split(" (")[0]
            drop["gate"][kind] = drop["gate"].get(kind, 0) + 1
            continue
        kind = data_scan.row_hit(row, word)
        if kind:
            drop["data_scan"][kind] = drop["data_scan"].get(kind, 0) + 1
            continue
        kept[case["id"]] = kept.get(case["id"], 0) + 1
        out_rows.append(row)
    os.makedirs(out_dir, exist_ok=True)
    with open(paths["train_local.jsonl"], "x", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    gate_rep = ct.gate_file(paths["train_local.jsonl"], paths["train_local.gated.jsonl"], gate=gate)
    scan_ok, scan_msg = data_scan.check(paths["train_local.gated.jsonl"], vocab)
    fams = {}
    for fam in sorted({c["family"] for c in by_id.values()}):
        jobs = [i for i, c in by_id.items() if c["family"] == fam]
        tried = [i for i in jobs if sampled.get(i)]
        hit = [i for i in tried if kept.get(i)]
        fams[fam] = {"jobs": len(jobs), "sampled": len(tried), "with_a_pass": sum(bool(passing.get(i)) for i in tried),
                     "trained": len(hit), "rows": sum(kept.get(i, 0) for i in jobs),
                     "share_with_a_pass": round(sum(bool(passing.get(i)) for i in tried) / len(tried), 3) if tried else None,
                     "samples": sum(sampled.get(i, 0) for i in jobs), "passing_samples": sum(passing.get(i, 0) for i in jobs),
                     "zero_pass": sorted(i for i in tried if not passing.get(i)),
                     "not_sampled": sorted(i for i in jobs if not sampled.get(i))}
    tried = [i for i in by_id if sampled.get(i)]
    summary = {"run": os.path.abspath(run_dir), "model": models[0] if models else None, "per_case": per_case,
               "jobs_sent": len(by_id), "jobs_sampled": len(tried),
               "jobs_with_a_pass": sum(bool(passing.get(i)) for i in tried),
               "share_with_a_pass": round(sum(bool(passing.get(i)) for i in tried) / len(tried), 3) if tried else None,
               "rows": len(out_rows), "samples": sum(sampled.values()), "passing_samples": sum(passing.values()),
               "temperatures": sorted({r.get("temperature") for _, _, r in rows_in if r.get("status") == "scored"},
                                      key=str),
               "dropped": drop, "families": fams, "gate_file": {k: gate_rep[k] for k in ("rows", "kept", "dropped")},
               "data_scan": scan_msg, "files": {k: os.path.abspath(p) for k, p in paths.items()}}
    with open(paths["export_local.json"], "x", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    why = (f"contamination --gate dropped {gate_rep['dropped']} rows the export had passed" if gate_rep["dropped"]
           else None if scan_ok else scan_msg)
    if why:   # nothing a launcher could train on is left behind: the gated copy and its report go
        for k in ("train_local.gated.jsonl.gate.json", "train_local.gated.jsonl"):
            os.unlink(paths[k])
        raise SystemExit("REFUSED: " + why)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the eval_run.py --set train_local run directory")
    ap.add_argument("--out", default="", help="where to write (default: the run directory)")
    ap.add_argument("--per-case", type=int, default=PER_CASE, help="at most this many distinct passing replies a job")
    ap.add_argument("--vocab", default="", help="identity vocabulary files for data_scan (default GRUX_DATA_SCAN_VOCAB)")
    a = ap.parse_args()
    s = export(a.run, a.out or a.run, a.per_case, a.vocab or os.environ.get("GRUX_DATA_SCAN_VOCAB", ""))
    for fam, f in s["families"].items():
        print(f"{fam:13s} jobs {f['jobs']:5d}  sampled {f['sampled']:5d}  with a pass {f['with_a_pass']:5d} "
              f"({f['share_with_a_pass']})  rows {f['rows']:5d}  zero-pass {len(f['zero_pass'])}")
    print(json.dumps({k: s[k] for k in ("dropped", "gate_file", "data_scan")}, indent=1))
    print(f"EXPORT: {s['rows']} rows from {s['jobs_with_a_pass']} of {s['jobs_sampled']} sampled jobs "
          f"({s['share_with_a_pass']}), at most {s['per_case']} a job; {s['files']['train_local.gated.jsonl']}")


if __name__ == "__main__":
    main()
