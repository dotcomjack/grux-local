# Grux Local autopilot

Keeps Grux Local on the best open base a Mac can run, on its own, once a week.

| Step | File | What it does |
|---|---|---|
| Watch | `watch.py` | Reads the Hugging Face API (the orgs in `base.json`) and the Ollama library. Keeps a base only when its license is on the allowlist (Apache-2.0, MIT, or an override with its reason), its weights are not HF-gated, it has 1B to 10.5B parameters and fits its Mac tier, and its Ollama build has the `tools` badge. Every model it drops is logged with the reason. Seen state is keyed by (source, id, digest); a build stays new until it has a verdict, so a re-published base is new again and a failed, refused or killed week is picked up the next. `--self-test` runs planted cases: a gated model and non-allowlisted licenses must be filtered, their clean twins must pass. |
| Score | `score.py` | One spot GPU box per model, in parallel. Each model is scored as a Mac runs it: the default Ollama library build, thinking off, num_ctx 16,384, one request at a time, on held-out v1 (464) and held-out v2 (466) with Grux's own prompts and parsers (`check_prompts.py` must print two MATCH lines first). Each finished task leaves the box at once as one line in this machine's outbox; a rerun or a replacement box resumes there. Writes `results/autopilot/<date>-<model>.json` with N and a 95% interval per cell, the build's digest and the Ollama version. |
| Decide | `decide.py` | Win, tie or lose against the stock row of the base for the same tier (`base.json`), on the 95% interval of the difference. `--reproduce` checks a box score of a published stock row against that row's interval. |
| Train | `train.py` | On a win only: the weights must be public and allowlisted, the contamination check runs first, then `s2/train_run.py` (the 4b-v2 recipe) trains, packs and scores, and the lane writes a Mini gate request (`results/autopilot/gate-requests/<id>.json`, with the exact `eval_local.py --set holdout` command) and a ready item. It never pushes a tag, never touches gruxai.com and never runs anything on the Mini. |
| Week | `run.py`, `install_launchd.py` | The weekly run (Mondays 3:00 AM), and the LaunchAgent that starts it. |
| Boxes | `boxes.py` | Lists the lane's boxes (`dcj.owner=grux-local`), or deletes one model's box (the kill-resume proof). |

Spend: every box create is preceded by a reservation on a shared weekly meter (`DAYTONA_SPEND_TOOL`); a refusal
creates no box. A swing estimated over $150 refuses in code before any reservation. Spot boxes only; each is
ephemeral, labelled `dcj.owner=grux-local` and held by a `Lease` (`../lease.py`).

Raw outboxes and logs stay on the machine that ran them (`~/.grux/autopilot/grux-local/`); the repo keeps only the
summary JSON. No personal data goes into a box: every task is synthetic, and the box only pulls a public tag.
