# Held-out v3 baseline: shipped 9B, Pro base, Max base (2026-11-14)

Held-out v3: 321 jobs, 545 steps, case sha `2506a2a8`, Grux v3.3.1 tools, 65,536 window, think unset. One Daytona H200 spot box per run, Ollama 0.40.8. Each model was scored twice. A skipped job counts as a fail.

| Model | Runner, format | Pass 1 | Pass 2 | Mean | Spread |
|---|---|---|---|---|---|
| `gruxai.com/grux/grux-local:9b` (shipped) | mlx, safetensors nvfp4 | 214 / 321 (66.7%) | 213 / 321 (66.4%) | 66.5% | 0.3 pt |
| `qwen3.5:35b-a3b-nvfp4` (Pro base) | mlx, safetensors nvfp4 | 180 / 321 (56.1%), 1 skip | 191 / 321 (59.5%), 1 skip | 57.8% | 3.4 pt |
| `qwen3.5:122b-a10b` (Max base) | llamacpp, gguf Q4_K_M | 245 / 321 (76.3%) | 250 / 321 (77.9%) | 77.1% | 1.6 pt |

## By family (pass 1 / pass 2)

| Family | 9B shipped | Pro base | Max base |
|---|---|---|---|
| chain (61) | 39 / 40 | 41 / 39 | 48 / 48 |
| injection (55) | 10 / 8 | 12 / 17 | 19 / 25 |
| long_context (45) | 38 / 38 | 8 / 8 | 44 / 45 |
| refusal (50) | 41 / 42 | 39 / 44 | 36 / 33 |
| shell (50) | 28 / 27 | 22 / 25 | 40 / 39 |
| tool_choice (60) | 58 / 58 | 58 / 58 | 58 / 60 |

## Verdicts against the charter (grux-local-ladder-1)

- **v3 headroom: PROVEN.** The shipped 9B scores 66.5%, under the 85% kill line, on two passes 0.3 point apart.
- **Shipped artifact on Linux: PROVEN.** The 9B pulled with the digest the Mac mini serves (`357e4bd45ff0`), loaded on the mlx runner and answered.
- **Pro target:** at least 10 points over its own base, so about 68% or better.
- **Max target:** at least 2 points over Pro. The Pro base moved 3.4 points between two identical passes, so a 2-point gate is inside the noise at 2 runs. It needs more passes or a wider margin before it can decide anything.

## What the families say

- **Injection is the weakest family for every model.** The 9B passes 8 to 10 of 55, and even the Max base passes only 19 to 25.
- **The Pro base collapses on long context** (8 of 45 against 38 for the 9B).
- **The shipped 9B leads on refusal.**

## Cost and cleanup

- **Cost:** about $10.33 across three H200 spot boxes, free credit only. One spot eviction during the Pro base resumed on a new box.
- **Boxes:** the org listed 0 boxes after the run.
- **Wallet:** the dashboard's free credit read $9,536.97 before and $9,532.19 after, a $4.78 move, with the paid balance at $0.00 both times. The $10.33 is the runner's own estimate; the dashboard has lagged settled charges before, so the gap is not explained yet.

## Not run

The v1 and v2 held-out base scores were not run: `eval_run.py` takes only the s1 and holdout_v3 sets.
