#!/usr/bin/env python3
"""Runs INSIDE the training box. LoRA SFT of a Qwen 3.5 instruct model on Grux's jobs, rendered with
the model's own chat template and thinking off, so training text is byte for byte what Ollama's
`qwen3.5` renderer sends at inference. Loss only on the assistant turn. Writes the merged model.

  python box_train.py --model Qwen/Qwen3.5-2B --data train.jsonl --out out --steps 50

Spot boxes are evicted without warning (measured 2026-10-22, mid-training). `--save-every N` writes
out/ckpt (adapter, optimizer, scheduler, step) atomically for the caller to pull out; `--resume DIR`
continues from such a checkpoint, replaying the data draws before it (the rows match an unbroken run; dropout's
torch random state is not saved, so the numbers are close, not identical); `--merge-only DIR` merges a finished adapter without training.
"""
import argparse
import hashlib
import sys
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, PeftModel, get_peft_model


def drop_verdict(dropped, total, max_pct):
    """None when few enough rows were too long to train on, else why the run must stop. A silent drop trained the
    full-turn rows (about 30,000 tokens) out of existence at the old 8,192 limit."""
    if total and 100.0 * dropped / total > max_pct:
        return f"{dropped} of {total} rows are longer than --max-len ({100.0 * dropped / total:.1f}% > {max_pct}%)"
    return None


def render_target(t):
    """The assistant turn exactly as Qwen 3.5's template renders it (tokenizer_config.json)."""
    content = t.get("content") or ""
    s = content
    for i, call in enumerate(t.get("tool_calls") or []):
        if i == 0:
            s += "\n\n<tool_call>\n<function=" if content.strip() else "<tool_call>\n<function="
        else:
            s += "\n<tool_call>\n<function="
        s += call["name"] + ">\n"
        for k, v in (call.get("arguments") or {}).items():
            v = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            s += f"<parameter={k}>\n{v}\n</parameter>\n"
        s += "</function>\n</tool_call>"
    return s + "<|im_end|>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=0, help="0 = epochs")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--max-drop-pct", type=float, default=1.0, help="refuse to train when more rows are too long")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--resume", default="", help="checkpoint dir written by --save-every")
    ap.add_argument("--merge-only", default="", help="adapter dir: merge and save, no training")
    ap.add_argument("--micro", type=int, default=1, help="examples per forward pass (length-bucketed)")
    ap.add_argument("--micro-tokens", type=int, default=24000, help="token ceiling per forward pass")
    a = ap.parse_args()
    torch.manual_seed(7)
    random.seed(7)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    cfg = transformers.AutoConfig.from_pretrained(a.model)
    cls = getattr(transformers, cfg.architectures[0])
    model = cls.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda")
    mem("after load")
    if a.merge_only:
        return save_merged(PeftModel.from_pretrained(model, a.merge_only), tok, a)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    ckpt = a.resume if a.resume and os.path.exists(os.path.join(a.resume, "step.json")) else ""
    if ckpt:
        model = PeftModel.from_pretrained(model, os.path.join(ckpt, "adapter"), is_trainable=True)
    else:
        lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules="all-linear",
                          exclude_modules=r".*(visual|vision|merger|lm_head).*", task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(a.data)]
    examples, dropped = [], 0
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"], tools=r.get("tools") or None, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(render_target(r["target"]), add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) > a.max_len:
            dropped += 1
            continue
        examples.append((p_ids, c_ids))
    print(f"examples {len(examples)} dropped_too_long {dropped}", flush=True)
    why = drop_verdict(dropped, len(rows), a.max_drop_pct)
    if why:
        sys.exit("REFUSED " + why)
    with open(os.path.join(os.path.dirname(a.out) or ".", "render_sample.txt"), "w") as f:
        r = rows[0]
        f.write(tok.apply_chat_template(r["messages"], tools=r.get("tools") or None, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
                + render_target(r["target"]))

    total = a.steps or math.ceil(len(examples) * a.epochs / (a.accum * a.micro))
    lm_head = model.get_output_embeddings()
    if a.micro > 1:
        check_fast_loss(model, lm_head, examples[:3])
        mem("after the fast-loss check")
        torch.cuda.empty_cache()   # the check's full-vocabulary pass leaves about 70 GB cached on a 35B (measured
        mem("after releasing the check's cache")   # 2026-11-16): Triton's first-backward autotune then had none
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / max(1, total)))
    start = 0
    if ckpt:
        opt.load_state_dict(torch.load(os.path.join(ckpt, "optimizer.pt"), map_location="cuda"))
        sched.load_state_dict(torch.load(os.path.join(ckpt, "scheduler.pt")))
        meta = json.load(open(os.path.join(ckpt, "step.json")))
        start = meta["step"]
        why = resume_mismatch(meta, draw_settings(a))
        if why:
            sys.exit("REFUSED " + why)
    draws = {"order": [], "queue": []}
    if start:   # replay the draws the steps before `start` made, so a resumed run trains on what an unbroken one would
        for _ in range(start * a.accum):
            next_draw(draws, examples, a.micro, a.micro_tokens)
        print(f"resumed at step {start}/{total} from {ckpt}, {start * a.accum} draws replayed", flush=True)
    model.train()
    t0 = time.time()
    for step in range(start, total):
        loss_sum = 0.0
        for _ in range(a.accum):
            idx = next_draw(draws, examples, a.micro, a.micro_tokens)
            if a.micro == 1:
                p_ids, c_ids = examples[idx[0]]
                ids = torch.tensor([p_ids + c_ids], device="cuda")
                labels = torch.tensor([[-100] * len(p_ids) + c_ids], device="cuda")
                loss = model(input_ids=ids, labels=labels).loss
            else:
                batch = [examples[i] for i in idx]
                if step < start + 2:   # the first steps' shapes and memory: what an OOM report needs
                    print(f"batch rows {len(batch)} padded {max(len(p) + len(c) for p, c in batch)} "
                          f"answer tokens {sum(len(c) for _, c in batch)}", flush=True)
                loss = fast_loss(model, lm_head, batch)
            if step < start + 2:
                mem("after forward")
            (loss / a.accum).backward()
            if step < start + 2:
                mem("after backward")
            loss_sum += loss.item() / a.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        print(f"step {step + 1}/{total} loss {loss_sum:.4f} lr {sched.get_last_lr()[0]:.2e} "
              f"elapsed {time.time() - t0:.0f}s", flush=True)
        if a.save_every and (step + 1) % a.save_every == 0 and step + 1 < total:
            save_ckpt(model, opt, sched, step + 1, total, a.out, draw_settings(a))
    model.save_pretrained(os.path.join(a.out, "adapter"))
    print("ADAPTER DONE", flush=True)
    save_merged(model, tok, a)


def mem(where):
    """CUDA memory at a point in the run, in GB (allocated, reserved, peak allocated, total)."""
    g = 1024 ** 3
    print(f"MEM {where}: allocated {torch.cuda.memory_allocated() / g:.1f} reserved {torch.cuda.memory_reserved() / g:.1f} "
          f"peak {torch.cuda.max_memory_allocated() / g:.1f} of {torch.cuda.get_device_properties(0).total_memory / g:.1f}",
          flush=True)


def next_draw(draws, examples, micro, max_tokens):
    """The example indices of the next forward pass: one shuffled epoch order at micro 1, else the epoch's length
    buckets, refilled when spent. Every random draw of training goes through here, so --resume replays the steps
    before its checkpoint by calling this start x accum times. Until 2026-11-23 a resume reseeded (7 + start) and
    began a fresh epoch, so a run resumed mid-epoch trained some rows twice and never saw others (the r5 pilot
    resumed twice)."""
    if micro == 1:
        if not draws["order"]:
            draws["order"] = list(range(len(examples)))
            random.shuffle(draws["order"])
        return [draws["order"].pop()]
    if not draws["queue"]:
        draws["queue"] = buckets(examples, micro, max_tokens)
    return draws["queue"].pop()


def buckets(examples, micro, max_tokens):
    """One epoch of micro-batches: shuffle, sort pools of similar length, cut by count and a token
    ceiling (padded length x rows), shuffle the batches. Right padding only: a pad after the text
    cannot change any earlier position, in attention or in the linear-attention layers alike."""
    order = list(range(len(examples)))
    random.shuffle(order)
    out = []
    for k in range(0, len(order), micro * 64):
        pool = sorted(order[k:k + micro * 64], key=lambda i: len(examples[i][0]) + len(examples[i][1]))
        cur = []
        for i in pool:
            n = len(examples[i][0]) + len(examples[i][1])
            if cur and (len(cur) == micro or n * (len(cur) + 1) > max_tokens):
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
    random.shuffle(out)
    return out


def fast_loss(model, lm_head, batch):
    """Mean over rows of each row's mean token loss on its answer, with the vocabulary projection run
    ONLY on answer tokens (the prompt is about 95% of every chat example's tokens)."""
    L = max(len(p) + len(c) for p, c in batch)
    ids = torch.zeros((len(batch), L), dtype=torch.long, device="cuda")
    mask = torch.zeros((len(batch), L), dtype=torch.long, device="cuda")
    rows, pos, tgt, row_of = [], [], [], []
    for r, (p, c) in enumerate(batch):
        n = len(p) + len(c)
        ids[r, :n] = torch.tensor(p + c, device="cuda")
        mask[r, :n] = 1
        for j in range(len(c)):
            rows.append(r)
            pos.append(len(p) + j - 1)      # the hidden state at t predicts token t + 1
            tgt.append(c[j])
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, logits_to_keep=1)
    h = out.hidden_states[-1][torch.tensor(rows, device="cuda"), torch.tensor(pos, device="cuda")]
    ce = F.cross_entropy(lm_head(h).float(), torch.tensor(tgt, device="cuda"), reduction="none")
    r = torch.tensor(rows, device="cuda")
    per_row = torch.zeros(len(batch), device="cuda").index_add_(0, r, ce) / torch.bincount(r, minlength=len(batch))
    return per_row.mean()


def check_fast_loss(model, lm_head, few):
    """Refuse to train if the fast path disagrees with the model's own loss (it rests on the last
    hidden state being the normed one, which a transformers release could change)."""
    model.eval()
    with torch.no_grad():
        for p, c in few:
            ids = torch.tensor([p + c], device="cuda")
            ref = model(input_ids=ids, labels=torch.tensor([[-100] * len(p) + c], device="cuda")).loss.item()
            got = fast_loss(model, lm_head, [(p, c)]).item()
            print(f"fast-loss check: model {ref:.5f} fast {got:.5f}", flush=True)
            if abs(ref - got) > 1e-2 * max(1.0, abs(ref)):
                raise SystemExit("fast loss disagrees with the model's own loss; train with --micro 1")
    model.train()


def draw_settings(a):
    """What decides the data draws: a resume with any of these changed would replay a different stream."""
    if not hasattr(a, "data_sha256"):   # once per run: the file is the same for every checkpoint
        h = hashlib.sha256()
        with open(a.data, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        a.data_sha256 = h.hexdigest()
    return {"accum": a.accum, "micro": a.micro, "micro_tokens": a.micro_tokens, "max_len": a.max_len,
            "data_sha256": a.data_sha256}


def resume_mismatch(meta, now):
    """None when a checkpoint's draw settings match this run's, else why the resume must stop. A checkpoint from
    before 2026-11-23 has none recorded; it resumes (its own run reseeded, so nothing it trained matches anyway)."""
    had = meta.get("draws")
    if had is None:
        return None
    bad = [f"{k} {had.get(k)!r} then, {v!r} now" for k, v in now.items() if had.get(k) != v]   # the data by its sha
    return ("the checkpoint's data draws differ: " + "; ".join(bad)) if bad else None


def save_ckpt(model, opt, sched, step, total, out, draws=None):
    tmp, final = os.path.join(out, "ckpt.tmp"), os.path.join(out, "ckpt")
    os.makedirs(tmp, exist_ok=True)
    model.save_pretrained(os.path.join(tmp, "adapter"))
    torch.save(opt.state_dict(), os.path.join(tmp, "optimizer.pt"))
    torch.save(sched.state_dict(), os.path.join(tmp, "scheduler.pt"))
    json.dump({"step": step, "total": total, "draws": draws}, open(os.path.join(tmp, "step.json"), "w"))
    if os.path.exists(final):
        os.rename(final, final + ".old")
    os.rename(tmp, final)
    os.system(f"rm -rf {final}.old")
    print(f"CKPT {step}/{total}", flush=True)


def save_merged(model, tok, a):
    merged = model.merge_and_unload()
    out = os.path.join(a.out, "merged")
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
    finalize(a.model, out)
    print("TRAIN DONE", flush=True)


def finalize(model_id, merged):
    """Make the merged folder the base's folder with new weights, nothing else changed. Measured
    2026-10-22: a re-saved tokenizer and a new generation_config.json (eos <|endoftext|>) made Ollama
    print a literal <|im_end|> after every reply, and loading dropped the 15 tensors of the MTP head."""
    import glob
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file
    base = model_id if os.path.isdir(model_id) else __import__("huggingface_hub").snapshot_download(model_id)
    for n in os.listdir(merged):
        if not (n.endswith(".safetensors") or n == "model.safetensors.index.json"):
            os.remove(os.path.join(merged, n))
    carried = []
    for p in glob.glob(os.path.join(base, "*")):
        n = os.path.basename(p)
        if os.path.isfile(p) and not (n.endswith(".safetensors") or n == "model.safetensors.index.json"):
            shutil.copy(p, os.path.join(merged, n))
            carried.append(n)
    index_path = os.path.join(merged, "model.safetensors.index.json")
    index = json.load(open(index_path))
    # The MTP head (multi-token prediction) is left out, as Ollama's own qwen3.5 builds leave it out:
    # measured 2026-10-22 on the Mini, shipped qwen3.5:4b has 0 mtp tensors (4.5B, 4.07 GB loaded) and a
    # build that kept the 15 loaded at 4.52 GB, 11% over base. Ollama does not run it.
    for k in [k for k in index["weight_map"] if k.startswith("mtp.")]:
        del index["weight_map"][k]
    missing = {}
    for shard in glob.glob(os.path.join(base, "*.safetensors")):
        with safe_open(shard, "pt") as f:
            for k in f.keys():
                if k not in index["weight_map"] and not k.startswith("mtp."):
                    missing[k] = f.get_tensor(k)
    if missing:
        save_file(missing, os.path.join(merged, "model-base-extra.safetensors"), metadata={"format": "pt"})
        index["weight_map"].update({k: "model-base-extra.safetensors" for k in missing})
    json.dump(index, open(index_path, "w"), indent=2)
    print(f"FINALIZED: carried {sorted(carried)}; restored {len(missing)} base tensors", flush=True)


if __name__ == "__main__":
    main()
