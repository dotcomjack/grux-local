#!/usr/bin/env python3
"""On a win only: train the winning base with the existing recipe, score it, and hand it to the Mini gate.

Order: the decision must say win; the base's Hugging Face weights must be public (never gated: no token goes in a
box) and allowlisted; the recipe must know the base's Ollama renderer; the contamination check runs FIRST (no
training row may share an 8-word run with either held-out set); then s2/train_run.py trains, merges, packs the
model for Ollama as nvfp4 and scores it on held-out v1 on the box; the Ollama model tar comes out to this machine.
Last, it writes a Mini gate request (results/autopilot/gate-requests/<id>.json, carrying the exact eval_local.py
command) and a ready item. It never pushes a tag, never touches gruxai.com and never runs anything on the Mini.

  DAYTONA_API_KEY=<your key> <venv>/bin/python recipe/autopilot/train.py \\
      results/autopilot/<date>-decide-<model>.json [--dry-run]
--dry-run runs every check and writes nothing that ships; it creates no box.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

TRAIN_CAP_USD = 50.0   # train_run's own cap: training $40 plus scoring $10


def hf_id_for(model):
    for p in sorted(glob.glob(os.path.join(common.RESULTS, "watch-*.json")), reverse=True):
        for c in json.load(open(p)).get("candidates", []):
            if c["id"] == model and c.get("hf_id"):
                return c["hf_id"]
    return None


def hf_public(repo):
    import urllib.request
    url = f"https://huggingface.co/api/models/{repo}?expand[]=gated&expand[]=cardData"
    m = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "grux-local-autopilot/1"}),
                                          timeout=30).read())
    return m.get("gated"), (m.get("cardData") or {}).get("license")


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def contamination(data_path):
    """s2/contamination.py's rule, applied to both held-out sets and to the rendered training rows: a row is
    contaminated when its ask equals a held-out case's text, or its user turn shares any 8-word run with one.
    Returns (contaminated ids, ids that only CONTAIN a short held-out ask inside a longer one: reported, not blocked)."""
    import contamination as c8
    import holdout
    import holdout_v2
    runs, wholes = set(), set()
    for case in holdout.cases() + holdout_v2.cases():
        ws = c8.texts(case)
        runs |= c8.grams(ws)
        wholes.add(" ".join(ws))
    hits, near = [], []
    for line in open(data_path):
        row = json.loads(line)
        content = " ".join(m.get("content") or "" for m in row.get("messages", []) if m.get("role") == "user")
        ask = " ".join(words(content.split("THE USER'S MESSAGE:")[-1]))
        user = words(content)
        if ask in wholes or c8.grams(user) & runs:
            hits.append(row.get("id"))
        elif any(len(w.split()) >= 3 and f" {w} " in f" {ask} " for w in wholes):
            near.append(row.get("id"))
    return hits, near


def home(path):
    """An absolute path: evidence the board opens must resolve from anywhere, never relative to the lane copy."""
    return os.path.abspath(os.path.expanduser(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decision")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpu", default="H200,H100")
    a = ap.parse_args()
    cfg = common.base_config()
    d = json.load(open(a.decision))
    model, tier = d["candidate"], d["tier"]

    def refuse(why):
        print(f"TRAIN REFUSED for {model}: {why}")
        sys.exit(2)

    if d.get("verdict") != "win":
        refuse(f"the verdict is {d.get('verdict')}, not win ({d.get('line')})")
    hf = hf_id_for(model)
    if not hf:
        refuse("no Hugging Face weights are linked to this Ollama build; the recipe trains from safetensors")
    gated, lic = hf_public(hf)
    if gated:
        refuse(f"{hf} is HF-gated (gated={gated}); no token goes in a box")
    ok, why = __import__("watch").license_ok("hf:" + hf, lic, cfg)
    if not ok:
        refuse(f"{hf}: {why}")
    cand_path = os.path.normpath(os.path.join(common.REPO, d["candidate_result"]))
    cand = json.load(open(cand_path))
    fam = (cand.get("details") or {}).get("family", "")
    if fam.replace(".", "") != cfg["train"]["modelfile_family"].replace(".", ""):
        refuse(f"the stock build's family is {fam!r}; s2/train_run.py's Modelfile renders {cfg['train']['modelfile_family']}"
               " only, so a person extends MODELFILE for this family first")
    data = os.path.expanduser(os.environ.get(cfg["train"]["data_env"]) or cfg["train"]["data_path"])
    if not os.path.exists(data):
        refuse(f"no training data at {data} (install_launchd.py --train-data pins it there)")
    vocab = os.environ.get("GRUX_DATA_SCAN_VOCAB") or cfg["train"]["data_scan_vocab"]
    import data_scan
    ok, msg = data_scan.check(data, vocab, require_vocab=True)
    print(msg)
    if not ok:
        refuse(msg)
    hits, near = contamination(data)
    print(f"contamination: {len(hits)} training rows equal a held-out ask or share an 8-word run with held-out v1 or v2"
          f"; {len(near)} more contain a short held-out ask inside a longer one (reported, not blocked)")
    if hits:
        refuse(f"contaminated training data, first ids {hits[:5]}")
    est = TRAIN_CAP_USD
    try:
        common.swing_guard(est, f"training {hf}")
    except common.Refused as e:
        refuse(str(e))
    run_dir = os.path.join(common.STATE, "train", f"{common.today()}-{common.stem(model)}")
    tag = f"grux-{common.stem(model).replace('_', '-')}"
    cmd = [sys.executable, os.path.join(common.RECIPE, "s2", "train_run.py"), "--run", run_dir, "--model", hf,
           "--data", data, "--rank", str(cfg["train"]["rank"]), "--epochs", str(cfg["train"]["epochs"]),
           "--cap", str(TRAIN_CAP_USD), "--tag", tag, "--skip-base", "--gpu", a.gpu,
           "--meter", common.SPEND_TOOL, "--meter-lane", common.LANE, "--owner", common.OWNER, "--data-scan-vocab", vocab]
    print("train command:", " ".join(cmd))
    if a.dry_run:
        print("DRY RUN: every check passed; no box created, nothing written")
        return
    r = subprocess.run(cmd)
    status = json.load(open(os.path.join(run_dir, "status.json"))).get("status") if os.path.exists(
        os.path.join(run_dir, "status.json")) else "no status"
    if r.returncode != 0 or status != "done":
        refuse(f"train_run.py ended {status} (exit {r.returncode}); log {run_dir}/run.log")
    rows = [json.loads(l) for l in open(os.path.join(run_dir, f"eval-{tag}-holdout.jsonl"))]
    if len({x["case"] for x in rows if x.get("status") == "scored"}) != 464:
        refuse(f"the trained model scored {len(rows)} of 464 held-out v1 tasks on the box; no gate request for a partial score")
    box_score = f"{sum(bool(x.get('pass')) for x in rows)}/{len(rows)}"
    tar = os.path.join(run_dir, f"ollama-{tag}.tar")
    gid = f"{common.today()}-{common.stem(model)}"
    mac_run = f"/tmp/grux-local-gate-{gid}"
    gate = {"id": gid, "candidate": model, "hf_id": hf, "tier": tier, "made_for": d["made_for"],
            "trained_tag": tag, "model_tar": tar, "box_score_holdout_v1": box_score, "decision": home(a.decision),
            "recipe": cfg["train"]["recipe"],
            "steps": [f"tar xf {os.path.basename(tar)} -C ~/.ollama/models",
                      "export GRUX_SRC=\"$(sh recipe/get_grux_source.sh)\" && python3 recipe/check_prompts.py",
                      f"python3 recipe/eval_local.py --model {tag} --set holdout --run {mac_run}",
                      f"python3 results/gate.py {mac_run}/{tag}-holdout.jsonl --tag {tier}"],
            "eval_command": f"python3 recipe/eval_local.py --model {tag} --set holdout --run {mac_run}",
            "never": "this request publishes nothing: no tag push, no gruxai.com change, latest stays where it is"}
    gpath = os.path.join(common.GATE_REQUESTS, gid + ".json")
    common.write_json(gpath, gate)
    item = common.ready_item(
        f"grux-local-{gid}", "mini-gate", f"Grux Local: gate {tag} on the Mini",
        f"A new open base ({model}) beat today's {tier} base and was trained; the Mini checks it before anyone ships it.",
        [home(a.decision), home(cand_path), home(gpath)], gate["eval_command"], [])
    print(f"wrote {gpath} and {item}")


if __name__ == "__main__":
    main()
