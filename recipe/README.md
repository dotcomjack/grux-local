# The Grux Local recipe

Everything that made Grux Local v1 and measured it: the held-out test set, the scorer, the training-data
pipeline and the trainer. Two things you can do with it:

1. **Check our numbers** on your own Apple silicon Mac in about 15 minutes, with no GPU and no account.
2. **Train your own** on rented GPUs, for your own app's jobs.

The scorer reads Grux's real prompts out of Grux's public Swift source, so a score is a score of exactly what
Grux sends a model. v1 was scored against Grux v3.1.0, and `check_prompts.py` proves your copy sends the same
464 requests byte for byte before you score anything.

## 1. Check the numbers (Apple silicon, Ollama 0.40.8 or later)

```
git clone https://github.com/dotcomjack/grux-local.git && cd grux-local
export GRUX_SRC="$(sh recipe/get_grux_source.sh)"
python3 recipe/check_prompts.py
ollama pull gruxai.com/grux/grux-local:2b
python3 recipe/eval_local.py --model gruxai.com/grux/grux-local:2b --set holdout --run /tmp/grux-local-check
python3 recipe/summarize.py /tmp/grux-local-check/*-holdout.jsonl
```

`check_prompts.py` must print `MATCH`. The summary should read the numbers in
[`results/v1.json`](../results/v1.json) for that tag: 435 of 464 for the 2B, 454 of 464 for the 4B, with the hidden
instruction cases out of 70 and the count for each kind of task. Requests go out one at a time with a fixed seed
and thinking off, as Grux sends them. Chat timing depends on your Mac; v1's timings are from a 16 GB Mac mini
(Apple M4). Use `:4b` for the larger model, and `--model qwen3.5:4b` (or `:2b`) to score the same model before
training.

## 2. Train your own

| Step | File | What it does |
|---|---|---|
| Cases | `s2/train_bank.py` | About 5,200 synthetic Grux tasks, deterministic for a seed, each with its expected result set when the task is made, never by a model |
| Held-out set | `s2/holdout.py` | 464 tasks written separately; `s2/contamination.py` proves none of them leaks into training (`s2/train_exclude.json`) |
| Answers | `s2/gen_data.py` | A teacher model (Gemma 4 26B on a GPU box, through Ollama) writes each reply; a reply is kept only when the scorer passes it |
| Training | `s2/box_train.py` | LoRA on Qwen 3.5 (2B, 4B, 9B), loss on the answer tokens only, checkpoints so a lost spot box resumes |
| Run | `s2/train_run.py` | Train, merge, import into Ollama as nvfp4, score the held-out set, pull the model out |
| Sweep | `s2/sweep.py`, `s2/sweep_report.py` | Twelve configurations in parallel, then a scorecard and a winner per Mac size |
| Scorer | `suite.py`, `eval_local.py` | Grux's prompts and parsers; `eval_local.py` scores a model on this Mac's own Ollama |
| GPU boxes | `train_box.py`, `gpu_box.py`, `lease.py` | Spot GPU boxes on Daytona, each under a 20-minute lease that a crashed process cannot outlive |

As written it runs on [Daytona](https://www.daytona.io) Hopper GPUs (H200, H100) with `DAYTONA_API_KEY` in the
environment; the box steps themselves are plain Python and shell, so another GPU host needs only `train_box.py`
rewritten. v1 cost about $154 of GPU time end to end: data, twelve training runs and scoring. No personal data
goes into a box; every case is synthetic.

To aim it at your own app, replace the prompts and parsers in `suite.py` and the case families in
`s2/train_bank.py` and `s2/holdout.py` with your app's jobs. Keep the held-out set written separately from the
training cases, and score a candidate against the same model before training on that held-out set before you
believe it.

Licensed Apache-2.0, like the model. Built by DotcomJack.
