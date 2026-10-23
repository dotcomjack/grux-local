# Grux Local

**The local brain for [Grux](https://github.com/dotcomjack/grux), the open source AI agent for macOS.** Trained on the jobs Grux actually gives a model, so a Mac with no API key and no account gets them done, privately, on device.

| Tag | Made for | Loaded on a Mac (measured) |
|---|---|---|
| `gruxai.com/grux/grux-local:2b` | 8 GB Macs | 2.65 GB |
| `gruxai.com/grux/grux-local:4b` (`latest`) | 16 GB Macs and up | 4.44 GB |

## What it does

- **Calls the right Grux tool with the right details.** Tasks, events, notes, memories, web search, opening apps, focus modes.
- **Says no honestly.** Asked for something no tool can do, it says so instead of pretending it did.
- **Treats pasted text as data.** Instructions hidden in a pasted email, web page, screen capture or document are not followed. It does what you asked about the text and tells you something in it tried to steer it.
- **Does the background jobs.** Fixes dictation without answering it, triages mail into the JSON Grux reads, classifies notifications, titles chats, summarizes meetings, files issues from what you said.

## Measured on a Mac mini

16 GB Mac mini, Ollama 0.40.8, one request at a time, thinking off as Grux sends it. 464 held-out Grux tasks written separately and never used in training, scored by Grux's own parsers, against the same model before training:

| | Before training | Grux Local |
|---|---|---|
| 4B, all 464 tasks | 396 (85%) | **456 (98%)** |
| 4B, hidden instructions ignored | 55 of 70 | **66 of 70** |
| 2B, all 464 tasks | 295 (64%) | **435 (94%)** |
| 2B, hidden instructions ignored | 43 of 70 | **66 of 70** |

Chat replies: 95% within 3.0 s (4B) and 2.2 s (2B) on that Mac mini.

## Use it

```
ollama pull gruxai.com/grux/grux-local:4b
```

Then in Grux: **Settings, Model**, enter `gruxai.com/grux/grux-local:4b`. On an 8 GB Mac use `:2b`. Works with Grux 3.1 today. Next up: the local model picker suggests it for your Mac on its own.

## How it was made

LoRA fine-tune of Qwen 3.5 (2B and 4B, Apache-2.0) on synthetic Grux tasks. Training answers were written by Gemma 4 26B (Apache-2.0) and kept only when Grux's own checker passed them and they were right on top of that: exact dictation, no promised actions, no refusal of the harmless request. No personal data was used. Apple silicon, Ollama 0.40.8 or later (MLX runner, nvfp4). Licensed Apache-2.0. Built by DotcomJack.
