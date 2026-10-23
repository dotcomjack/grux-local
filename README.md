# Grux Local

**The local brain for [Grux](https://github.com/dotcomjack/grux), the open source AI agent for macOS.** Trained on the jobs Grux actually gives a model, to do them on a Mac with no API key and no account.

| Tag | Made for | Loaded on the 16 GB Mac mini (measured) |
|---|---|---|
| `gruxai.com/grux/grux-local:2b` | 8 GB Macs | 2.65 GB |
| `gruxai.com/grux/grux-local:4b` (`latest`) | 16 GB Macs and up | 4.44 GB |
| `gruxai.com/grux/grux-local:9b` | 24 GB Macs and up | 7.25 GB |

Needs a Mac with Apple silicon and Ollama 0.40.8 or later.

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

## Use it with Grux 3.1

```
ollama pull gruxai.com/grux/grux-local:4b
```

On an 8 GB Mac, pull `gruxai.com/grux/grux-local:2b` instead, and on a Mac with 24 GB or more, `gruxai.com/grux/grux-local:9b`. Then in Grux open **Settings, Models, Offline mode (local model)**, turn on **Route chat to a local model**, and enter the same tag in **Model**. Chat then runs on your Mac through Ollama: no API key, no account, and no model provider sees what you ask. Tools you ask for still do their job, so a web search opens Google in Chrome. Next up: Grux's local model picker will suggest it for your Mac on its own.

## How it was made

LoRA fine-tune of Qwen 3.5 (2B, 4B and 9B, Apache-2.0) on synthetic Grux tasks. The expected result of each task was set when the task was made, never by a model. Gemma 4 26B (Apache-2.0) wrote the reply text, and a reply was kept only when Grux's own checker passed it and it was right on top of that: exact dictation, no promised actions, no refusal of the harmless request. No personal data was used. Runs on Ollama's MLX runner: the 2B and 4B in nvfp4, and the 9B in the same mixed recipe as Ollama's own qwen3.5:9b (NVFP4 on linear attention and MLP, MXFP8 on full attention, embeddings and lm_head), quantized with NVIDIA ModelOpt. Licensed Apache-2.0. Built by DotcomJack.
