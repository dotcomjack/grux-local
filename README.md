# Grux Local

**The local brain for [Grux](https://github.com/dotcomjack/grux), the open source AI agent for macOS.** Trained on the jobs Grux actually gives a model, to do them on a Mac with no API key and no account.

| Tag | Made for | Freshly loaded on the 16 GB Mac mini at num_ctx 16,384 (measured) |
|---|---|---|
| `gruxai.com/grux/grux-local:2b` (text only) | 8 GB Macs | 1.54 GB |
| `gruxai.com/grux/grux-local:4b` (`latest`) | 16 GB Macs and up | 4.44 GB |
| `gruxai.com/grux/grux-local:9b` | 24 GB Macs and up | 7.25 GB |

Needs a Mac with Apple silicon and Ollama 0.40.8 or later. The 2B takes text only: it was rebuilt in November 2026 and no longer accepts images, see [The 2B, rebuilt](#the-2b-rebuilt). A second version of the 4B, `gruxai.com/grux/grux-local:4b-v2`, is opt-in: see [4B v2, opt-in](#4b-v2-opt-in).

Every loaded size here is a fresh load at num_ctx 16,384, what Grux 3.1 uses on a 16 GB Mac. Grux 3.1 uses num_ctx 32,768 on Macs with 24 GB or more ([`LocalTurn.swift:21-27`](https://github.com/dotcomjack/grux/blob/v3.1.0/Grux-Mac/Sources/Grux/Chat/LocalTurn.swift#L21-L27)), so a model loads more there, which matters most for the 9B, made for 24 GB Macs and up. Ollama can also report more memory for the model as it works: on the 16 GB Mac mini the 4B (`latest`) read 4.44 GB freshly loaded and 13.11 GB after a full held-out run (464 tasks), while the 2B read 1.54 GB both freshly loaded and after the same 464 tasks, though macOS put its runner process at 2,637 MB after them; the 4B v2 read 4.44 GB freshly loaded and 13.11 GB after 72 held-out tasks on a MacBook Pro (Apple M4 Pro, 48 GB). All at num_ctx 16,384. How far it grows on an 8 GB Mac is not yet measured.

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
| 2B, all 464 tasks | 295 (64%) | **439 (95%)** |
| 2B, hidden instructions ignored | 43 of 70 | **64 of 70** |

The 9B's Grux Local numbers are from the same Mac mini. Its before-training numbers are from a MacBook Pro (Apple M4 Pro, 48 GB) with the same Ollama 0.40.8 and the same 464 tasks, because the Mac mini had no disk to spare for the stock 9B:

| | Before training, on the MacBook Pro | Grux Local |
|---|---|---|
| 9B, all 464 tasks | 400 (86%) | **458 (99%)** |
| 9B, hidden instructions ignored | 65 of 70 | **66 of 70** |

Chat replies from Grux Local on that Mac mini: 95% came back within 3.4 s (4B), 2.4 s (2B) and 3.8 s (9B).

Every number here, with the count for each kind of task, is in [`results/v1.json`](results/v1.json). `python3 results/check.py` checks it against this card, the Grux README and gruxai.com, and checks that each tag's manifest hashes to the digest recorded there.

## The 2B, rebuilt

`gruxai.com/grux/grux-local:2b` now points at a new build, manifest `sha256:1622ac1b80afb8841fc588fbc57eff60284f318d021311322a84df4aa097251e`. It replaced the first 2B (manifest `sha256:601cd23c66ef4c834dec0c3f518c3663f2abe2298f62249b7337eb0845fb3ee4`), so `ollama pull gruxai.com/grux/grux-local:2b` now gets the new one.

**What changed.** The same base (Qwen/Qwen3.5-2B), trained with LoRA on round 5 data, which is not published here, then converted to a single GGUF file and quantized to Q4_K_M with llama.cpp. Ollama runs it on its llama.cpp runner (`llama-server`), not on MLX like the other tags.

**Text only.** Ollama lists its capabilities as completion, tools and thinking, with no vision, and a chat request that carries an image gets HTTP 400 "Multimodal data provided, but model does not support multimodal requests." The first 2B accepted images. So in Grux, a chat with an image attached fails when it is routed to the local 2B: send the text instead, or use another model for pictures.

Both 2Bs on the same 16 GB Mac mini, Ollama 0.40.8, num_ctx 16,384 unless named: the first 2B and the new one's scored build on the evening of November 23, 2026, and the new one's published bytes on November 24:

| On the 16 GB Mac mini | First 2B | New 2B |
|---|---|---|
| Held-out v1, all 464 tasks | 435 (94%) | **439 (95%)** |
| Held-out v1, hidden instructions ignored | **65 of 70** | 64 of 70 |
| Held-out v2, all 466 tasks | 371 (80%) | **414 (89%)** |
| Held-out v2, hidden instructions: task done, steering named | 4 of 70 | **46 of 70** |
| Local turns as Grux 3.3.1 sends them, 237 tasks, four runs | 163 to 166 | **196 to 197** |
| Of those, saying plainly what it cannot do | **31 to 32 of 34** | 27 of 34 |
| File naming at num_ctx 8,192, 92 items | **65 of 92** | 62 of 92 |
| Chat replies on held-out v1, 95% within | **2.0 s** | 2.4 s |

Where the new 2B is worse: it takes text only; it ignored hidden instructions in one held-out v1 case fewer (64 against 65 that evening, and 66 when the first 2B was published); on the local turns it failed 7 of the 34 that ask for something no tool can do, in every run (1 event created that nobody asked for, 1 call to a tool the turn did not offer, 2 offers of actions no offered tool does, 3 empty replies); it named fewer files well (62 against 65 of 92); and its chat replies were slower, on held-out v1 (see the table) and slightly on held-out v2 (95% within 8.8 s against 8.6 s). Its first word came later on held-out v1 and on the local turns (median 1.09 s against 0.09 s and 0.12 s) and sooner on held-out v2 (0.63 s against 5.68 s).

Held-out v1 and the loaded size were measured on the published bytes. Held-out v2, the local turns and file naming were measured on the scored build, which differs from the published `:2b` only in an appended license file ([`results/reproductions/2026-11-24-2b-v2-license.md`](results/reproductions/2026-11-24-2b-v2-license.md)); on held-out v1 the two gave the same pass or fail on every task. The local turns are held-out v2's 265 chat turns sent the way Grux 3.3.1 sends a local turn, less the 28 that Grux answers without a model; the fourth run was at num_ctx 8,192, what Grux uses on an 8 GB Mac. Every number here and its raw rows are in [`results/v1.json`](results/v1.json) under `2b` and [`results/mini-r5-20261123/`](results/mini-r5-20261123/); the first 2B's record is under `replaced`.

Round 5's 15,842 training rows went through the recipe's training gate (`recipe/s2/contamination.py`), which checks them against held-out v3 and v4 and found none touched ([`train-mix-r5.gate.json`](results/mini-r5-20261123/train-mix-r5.gate.json), its source path shortened to the file name). This repository holds no check of round 5's rows against held-out v1 or v2, so the held-out v2 gain above is not shown to be free of overlap with training.

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

## Held-out v3, for a bigger tier

We built a third held-out set for the next, larger Grux Local tiers: 321 long, multi-step jobs on the full Grux turn,
with all 119 tools Grux v3.3.1 defines offered on every request. It measures chains of dependent tool calls, shell
work with snapshots and undo, saying no when no tool can do the ask, ignoring instructions planted in pasted or
fetched text, using a fact said far back in a long conversation, and picking the right tool out of the whole
catalogue. Every label was written by hand and no model labelled or judges any case. No model has been scored on
it yet. How it works: [`recipe/README.md`](recipe/README.md#held-out-v3-the-full-grux-turn).

## Use it with Grux 3.1

```
ollama pull gruxai.com/grux/grux-local:4b
```

On an 8 GB Mac, pull `gruxai.com/grux/grux-local:2b` instead (text only), and on a Mac with 24 GB or more, `gruxai.com/grux/grux-local:9b`. To try 4B v2, pull `gruxai.com/grux/grux-local:4b-v2`. Then in Grux open **Settings, Models, Offline mode (local model)**, turn on **Route chat to a local model**, and enter the same tag in **Model**. Chat then runs on your Mac through Ollama: no API key, no account, and no model provider sees what you ask. Tools you ask for still do their job, so a web search opens Google in Chrome. Next up: Grux's local model picker will suggest it for your Mac on its own.

## How it was made

LoRA fine-tune of Qwen 3.5 (2B, 4B and 9B, Apache-2.0) on synthetic Grux tasks. The expected result of each task was set when the task was made, never by a model. Gemma 4 26B (Apache-2.0) wrote the reply text, and a reply was kept only when Grux's own checker passed it and it was right on top of that: exact dictation, no promised actions, no refusal of the harmless request. No personal data was used. That is round 1, behind the 4B and 9B; the 4B v2 was trained on round 2 and the current 2B on round 5, neither of which is published. The 4B runs on Ollama's MLX runner in nvfp4, and the 9B in the same mixed recipe as Ollama's own qwen3.5:9b (NVFP4 on linear attention and MLP, MXFP8 on full attention, embeddings and lm_head), quantized with NVIDIA ModelOpt. The 2B is a GGUF file in Q4_K_M, converted and quantized with llama.cpp, and runs on Ollama's llama.cpp runner. Licensed Apache-2.0. Built by DotcomJack.
