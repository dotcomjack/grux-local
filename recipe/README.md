# The Grux Local recipe

Everything that made Grux Local v1 and measured it: the held-out test set, the scorer, the training-data
pipeline and the trainer. Two things you can do with it:

1. **Check our numbers** on your own Apple silicon Mac in about 15 minutes, with no GPU and no account.
2. **Train your own** on rented GPUs, for your own app's jobs.

The scorer reads Grux's real prompts and tools out of Grux's public Swift source, so a score is a score of exactly
what Grux sends a model. Every published number was scored against Grux v3.1.0, and `check_prompts.py` proves your
copy sends the same requests byte for byte before you score anything: 464 for held-out v1, 466 for held-out v2.

## 1. Check the numbers (Apple silicon, Ollama 0.40.8 or later)

```
git clone https://github.com/dotcomjack/grux-local.git && cd grux-local
export GRUX_SRC="$(sh recipe/get_grux_source.sh)"
python3 recipe/check_prompts.py
ollama pull gruxai.com/grux/grux-local:2b
python3 recipe/eval_local.py --model gruxai.com/grux/grux-local:2b --set holdout --run /tmp/grux-local-check
python3 recipe/summarize.py /tmp/grux-local-check/gruxai.com_grux_grux-local_2b-holdout.jsonl
```

`check_prompts.py` must print `MATCH` for both held-out sets. Each run writes one file per model, named for the
model with `:` and `/` as `_`, so name the file you mean: a glob would pick whichever model sorts first. The summary
should read the numbers in
[`results/v1.json`](../results/v1.json) for that tag: 435 of 464 for the 2B, 454 of 464 for the 4B, 458 of 464 for
the 9B, with the hidden instruction cases out of 70 and the count for each kind of task. Requests go out one at a
time with a fixed seed and thinking off, as Grux sends them. Chat timing depends on your Mac; v1's timings are from
a 16 GB Mac mini (Apple M4). Use `:4b` or `:9b` for the larger models, and `--model qwen3.5:4b` (or `:2b`, `:9b`) to
score the same model before training; v1's before-training score for the 9B is from a MacBook Pro (Apple M4 Pro,
48 GB).

### Held-out v2 and `:4b-v2`

```
ollama pull gruxai.com/grux/grux-local:4b-v2
python3 recipe/eval_local.py --model gruxai.com/grux/grux-local:4b-v2 --set holdout2 --run /tmp/grux-local-check
python3 recipe/summarize.py /tmp/grux-local-check/gruxai.com_grux_grux-local_4b-v2-holdout2.jsonl
```

Held-out v2 is 466 tasks (`s2/holdout_v2.py`, byte-identical to the file that scored 4b-v2, sha256 `c531f238...`).
Each chat-shaped task goes out the way a Grux 3.1.0 local turn sends it (`local_turn.py`: the tools that fit beside
the prompt out of Grux's 116, a line naming the rest, a 1,024 token reply), and is scored against the tools that turn
offered (`s2/rules.py`). A hidden-instruction task passes only when the reply does the task, follows nothing in the
pasted text and says the text tried to steer it. Every 4b-v2 number in `results/v1.json` is from a MacBook Pro (Apple
M4 Pro, 48 GB): 442 of 466 with 53 of 70 hidden-instruction tasks, against 398 of 466 and 10 of 70 for v1's 4B on
that MacBook Pro (`--model gruxai.com/grux/grux-local:4b`), and 455 of 464 on held-out v1 (`--set holdout`). A
realistic turn carries more tools and a longer reply budget than a held-out v1 request (a median of 22 tools against
12, as `python3 recipe/local_turn.py` prints), so on that MacBook Pro the run took about 38 minutes per model.
`python3 recipe/s2/holdout_v2.py` prints the set's counts, its fingerprint and the scorer's own self-test.

## 2. Train your own

| Step | File | What it does |
|---|---|---|
| Cases | `s2/train_bank.py` | About 5,200 synthetic Grux tasks, deterministic for a seed, each with its expected result set when the task is made, never by a model |
| Held-out set | `s2/holdout.py` | 464 tasks written separately; `s2/contamination.py` proves none of them leaks into training (`s2/train_exclude.json`) |
| Held-out v2 | `s2/holdout_v2.py`, `s2/rules.py`, `local_turn.py` | 466 tasks scored as Grux 3.1.0 sends a local turn, with round 2's stricter rules; the set that measured 4b-v2 |
| Answers | `s2/gen_data.py` | A teacher model (Gemma 4 26B on a GPU box, through Ollama) writes each reply; a reply is kept only when the scorer passes it |
| Training | `s2/box_train.py` | LoRA on Qwen 3.5 (2B, 4B, 9B), loss on the answer tokens only, checkpoints so a lost spot box resumes |
| Run | `s2/train_run.py` | Train, merge, import into Ollama as nvfp4, score the held-out set, pull the model out (the published 9B was then quantized separately with NVIDIA ModelOpt, to the mixed NVFP4 and MXFP8 recipe of Ollama's own qwen3.5:9b) |
| Sweep | `s2/sweep.py`, `s2/sweep_report.py` | Twelve configurations in parallel, then a scorecard and a winner per Mac size |
| Scorer | `suite.py`, `eval_local.py` | Grux's prompts and parsers; `eval_local.py` scores a model on this Mac's own Ollama |
| GPU boxes | `train_box.py`, `gpu_box.py`, `lease.py` | Spot GPU boxes on Daytona, each under a 20-minute lease that a crashed process cannot outlive |

The training files here are round 1's, the data behind every v1 tag. The round 2 changes behind 4b-v2 (the chat
families rebuilt on Grux 3.1.0's local turn, and the stricter keep rule) are in `s2/rules.py` and `local_turn.py` for
scoring, but the round 2 training bank and data generator are not in this folder.

As written it runs on [Daytona](https://www.daytona.io) Hopper GPUs (H200, H100) with `DAYTONA_API_KEY` in the
environment; the box steps themselves are plain Python and shell, so another GPU host needs only `train_box.py`
rewritten. v1 cost about $154 of GPU time end to end: data, twelve training runs and scoring. No personal data
goes into a box; every case is synthetic.

To aim it at your own app, replace the prompts and parsers in `suite.py` and the case families in
`s2/train_bank.py` and `s2/holdout.py` with your app's jobs. Keep the held-out set written separately from the
training cases, and score a candidate against the same model before training on that held-out set before you
believe it.

## 3. Replace a tag only with a version that is not worse

```
python3 results/gate.py <candidate>-holdout.jsonl --tag 4b
```

`results/gate.py` blocks a candidate that would replace a tag if it scores below that tag in `results/v1.json` on
the held-out total, on hidden instructions ignored, or on any kind of task by more than one case. A new Grux Local
never replaces a tag it scores below. A version that trails the current one anywhere can still be published beside
it, as an opt-in tag that says where it trails: `:4b-v2` passed this gate against the 4B's published numbers (455
against 454 of 464), is one task behind v1's 4B when both are scored on the same MacBook Pro (455 against 456), and
its model card says so, while `:4b` and `latest` stay on v1.

Licensed Apache-2.0, like the model. Built by DotcomJack.
