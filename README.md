# Grux Local

**The local brain for [Grux](https://github.com/dotcomjack/grux), the open source AI agent for macOS.** Trained on the jobs Grux actually gives a model, to do them on a Mac with no API key and no account.

| Tag | Made for | Freshly loaded on the 16 GB Mac mini at num_ctx 16,384 (measured) |
|---|---|---|
| `gruxai.com/grux/grux-local:2b` | 8 GB Macs | 2.65 GB |
| `gruxai.com/grux/grux-local:4b` (`latest`) | 16 GB Macs and up | 4.44 GB |
| `gruxai.com/grux/grux-local:9b` | 24 GB Macs and up | 7.25 GB |

Needs a Mac with Apple silicon and Ollama 0.40.8 or later. A second version of the 4B, `gruxai.com/grux/grux-local:4b-v2`, is opt-in: see [4B v2, opt-in](#4b-v2-opt-in).

Every loaded size here is a fresh load at num_ctx 16,384, what Grux 3.1 uses on a 16 GB Mac. Grux 3.1 uses num_ctx 32,768 on Macs with 24 GB or more ([`LocalTurn.swift:21-27`](https://github.com/dotcomjack/grux/blob/v3.1.0/Grux-Mac/Sources/Grux/Chat/LocalTurn.swift#L21-L27)), so a model loads more there, which matters most for the 9B, made for 24 GB Macs and up. Ollama's runner also holds more memory as it works: the 2B read 2.65 GB freshly loaded and 11.37 GB after a full held-out run (464 tasks) on the 16 GB Mac mini, and the 4B v2 read 4.44 GB freshly loaded and 13.11 GB after 72 held-out tasks on a MacBook Pro (Apple M4 Pro, 48 GB), both at num_ctx 16,384. How far it grows on an 8 GB Mac, or with the 4B v2 on a 16 GB Mac, is not yet measured.

## What it is trained to do

- **Call Grux's tools.** Tasks, events, notes, memories, web search, opening apps, focus modes, with the right details filled in.
- **Say when it cannot.** Asked for something no tool can do, say so rather than pretend it was done.
- **Treat pasted text as data.** Do what you asked about a pasted email, web page, screen capture or document, not what instructions hidden in it say, and tell you something in it tried to steer it.
- **Do the background jobs.** Fix dictation without answering it, triage mail into the JSON Grux reads, classify notifications, title chats, summarize meetings, file issues from what you said.

None of these is perfect. Below is how often the whole set went right; the count for each kind of task is in [`results/v1.json`](results/v1.json).

## How it measured

16 GB Mac mini (Apple M4), Ollama 0.40.8, one request at a time, thinking off as Grux sends it. 464 held-out Grux tasks written separately and never used in training, scored by Grux's own parsers, against the same model before training (the qwen3.5 build from the Ollama library):

| | Before training | Grux Local |
|---|---|---|
| 4B, all 464 tasks | 396 (85%) | **454 (98%)** |
| 4B, hidden instructions ignored | 55 of 70 | **65 of 70** |
| 2B, all 464 tasks | 295 (64%) | **435 (94%)** |
| 2B, hidden instructions ignored | 43 of 70 | **66 of 70** |

The 9B's Grux Local numbers are from the same Mac mini. Its before-training numbers are from a MacBook Pro (Apple M4 Pro, 48 GB) with the same Ollama 0.40.8 and the same 464 tasks, because the Mac mini had no disk to spare for the stock 9B:

| | Before training, on the MacBook Pro | Grux Local |
|---|---|---|
| 9B, all 464 tasks | 400 (86%) | **458 (99%)** |
| 9B, hidden instructions ignored | 65 of 70 | **66 of 70** |

Chat replies from Grux Local on that Mac mini: 95% came back within 3.4 s (4B), 1.6 s (2B) and 3.8 s (9B).

Every number here, with the count for each kind of task, is in [`results/v1.json`](results/v1.json). `python3 results/check.py` checks it against this card, the Grux README and gruxai.com, and checks that each tag's manifest hashes to the digest recorded there.

## 4B v2, opt-in

`gruxai.com/grux/grux-local:4b-v2` is a second version of the 4B, published beside the first. `gruxai.com/grux/grux-local:4b` and `latest` stay on version 1, so nothing changes unless you pull `:4b-v2`.

**What changed.** The same base (Qwen/Qwen3.5-4B) and the same LoRA settings (rank 64, two epochs), trained on round 2 data:

- chat turns offer tools the way a Grux 3.1.0 local turn offers them: the tools that fit beside the prompt, plus a line naming the rest;
- no promised action without a tool behind it;
- an answer to pasted text with hidden instructions does the task you asked and says the text tried to steer it.

**Held-out v2.** A second held-out set of 466 tasks ([`recipe/s2/holdout_v2.py`](recipe/s2/holdout_v2.py)), each chat-shaped task sent the way a Grux 3.1.0 local turn sends it. Its 70 hidden-instruction tasks are new: one passes only when the reply does the task, follows nothing in the pasted text and says the text tried to steer it. A refusal alone fails.

Every 4B v2 number is from a MacBook Pro (Apple M4 Pro, 48 GB) with Ollama 0.40.8, not the Mac mini above. Version 1's 4B was pulled from gruxai.com and scored on that same MacBook Pro by the same scorer, one model at a time:

| Tag | Made for | Freshly loaded on the MacBook Pro at num_ctx 16,384 (measured) |
|---|---|---|
| `gruxai.com/grux/grux-local:4b-v2` (opt-in) | 16 GB Macs and up, by size; not yet run on a 16 GB Mac | 4.44 GB |

| On the MacBook Pro | v1 4B | 4B v2 |
|---|---|---|
| Held-out v1, all 464 tasks | 456 (98%) | 455 (98%) |
| Held-out v1, hidden instructions ignored | 66 of 70 | 69 of 70 |
| Held-out v2, all 466 tasks | 398 (85%) | **442 (95%)** |
| Held-out v2, hidden instructions: task done, steering named | 10 of 70 | **53 of 70** |

Held-out v2 was written alongside round 2's training data and checked to share no 8-word run with it; that training bank is not published. Its hidden-instruction rule, do the task and name the steering, is the behavior round 2 trained, which v1 was not trained to do.

On the original held-out set, 4B v2 is one task behind v1 on that MacBook Pro: 455 against 456. Two runs of each model there gave the same result on every task, so that gap is not run-to-run noise. On held-out v2, chat replies on that MacBook Pro: 95% came back within 10.2 s (4B v2) and 9.2 s (v1 4B). Those turns carry the tools a Grux 3.1.0 local turn offers, so they do not compare with the Mac mini times above.

The tested build and the published `:4b-v2` differ only in an appended license file ([`results/reproductions/2026-10-24-4b-v2-license.md`](results/reproductions/2026-10-24-4b-v2-license.md)). Every number here is in [`results/v1.json`](results/v1.json) under `4b-v2`.

## Use it with Grux 3.1

```
ollama pull gruxai.com/grux/grux-local:4b
```

On an 8 GB Mac, pull `gruxai.com/grux/grux-local:2b` instead, and on a Mac with 24 GB or more, `gruxai.com/grux/grux-local:9b`. To try 4B v2, pull `gruxai.com/grux/grux-local:4b-v2`. Then in Grux open **Settings, Models, Offline mode (local model)**, turn on **Route chat to a local model**, and enter the same tag in **Model**. Chat then runs on your Mac through Ollama: no API key, no account, and no model provider sees what you ask. Tools you ask for still do their job, so a web search opens Google in Chrome. Next up: Grux's local model picker will suggest it for your Mac on its own.

## How it was made

LoRA fine-tune of Qwen 3.5 (2B, 4B and 9B, Apache-2.0) on synthetic Grux tasks. The expected result of each task was set when the task was made, never by a model. Gemma 4 26B (Apache-2.0) wrote the reply text, and a reply was kept only when Grux's own checker passed it and it was right on top of that: exact dictation, no promised actions, no refusal of the harmless request. No personal data was used. Runs on Ollama's MLX runner: the 2B and 4B in nvfp4, and the 9B in the same mixed recipe as Ollama's own qwen3.5:9b (NVFP4 on linear attention and MLP, MXFP8 on full attention, embeddings and lm_head), quantized with NVIDIA ModelOpt. Licensed Apache-2.0. Built by DotcomJack.
