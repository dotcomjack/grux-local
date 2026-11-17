#!/usr/bin/env python3
"""Runs INSIDE a GGUF box (s2/gguf_run.py): merge a trained LoRA adapter into its public base, on the GPU when the
weights fit there with room to spare and on the host otherwise (a 35B is about 70 GB of bf16; an RTX 5090 has 32 GB),
then leave the folder exactly as box_train leaves it (box_train.save_merged: the base's own config and tokenizer, no
MTP head), so the GGUF is made from the same merged weights the nvfp4 build was.

  python gguf_box.py --base /workspace/base --adapter /workspace/out/adapter --out /workspace/out [--device auto]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def weights_bytes(folder):
    return sum(os.path.getsize(os.path.join(folder, n)) for n in os.listdir(folder) if n.endswith(".safetensors"))


def pick_device(weights, vram):
    """cuda when the bf16 weights take at most two thirds of the card, less 8 GB for the merge's own buffers."""
    return "cuda" if vram >= 1.5 * weights + 8e9 else "cpu"


def adapter_problems(saved_keys, loaded_keys, b_nonzero, differ):
    """Why the loaded adapter is not the trained one, or [] when it is. PEFT only warns on an adapter weight that
    matches no module, so a transformers or peft release that renamed one would merge a partial or no-op adapter
    without a word; here it stops the merge."""
    lora = lambda ks: {k for k in ks if ".lora_A." in k or ".lora_B." in k}  # noqa: E731
    s, l = lora(saved_keys), lora(loaded_keys)
    bad = []
    if not s:
        bad.append("the adapter file has no LoRA weights")
    if s - l:
        bad.append(f"{len(s - l)} adapter weights matched no module (first {sorted(s - l)[0]})")
    if l - s:
        bad.append(f"{len(l - s)} LoRA weights in the model are not in the adapter file (first {sorted(l - s)[0]})")
    if differ:
        bad.append(f"{differ} LoRA weights loaded with values other than the file's")
    if s and not b_nonzero:
        bad.append("every lora_B is zero: the merge would change nothing")
    return bad


def base_commit(folder):
    """The Hugging Face commit snapshot_download wrote into local_dir (its .cache metadata), or None."""
    p = os.path.join(folder, ".cache", "huggingface", "download", "config.json.metadata")
    try:
        return open(p).readline().strip() or None
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()
    import peft
    import torch
    import transformers
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    import box_train
    vram = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    w = weights_bytes(a.base)
    dev = pick_device(w, vram) if a.device == "auto" else a.device
    print(f"MERGE on {dev}: weights {w / 1e9:.1f} GB, GPU {vram / 1e9:.1f} GB", flush=True)
    cfg = transformers.AutoConfig.from_pretrained(a.base)
    model = getattr(transformers, cfg.architectures[0]).from_pretrained(a.base, torch_dtype=torch.bfloat16,
                                                                        device_map=dev)
    model = PeftModel.from_pretrained(model, a.adapter)
    saved = load_file(os.path.join(a.adapter, "adapter_model.safetensors"))
    loaded = get_peft_model_state_dict(model)
    common = [k for k in saved if k in loaded]
    differ = sum(not torch.equal(saved[k].to(loaded[k].device, loaded[k].dtype), loaded[k]) for k in common)
    b_nonzero = any(bool(loaded[k].abs().max() > 0) for k in common if ".lora_B." in k)
    bad = adapter_problems(saved.keys(), loaded.keys(), b_nonzero, differ)
    if bad:
        sys.exit("ADAPTER REFUSED: " + "; ".join(bad))
    facts = {"device": dev, "weights_gb": round(w / 1e9, 2), "gpu_gb": round(vram / 1e9, 1),
             "lora_weights": len(common), "transformers": transformers.__version__, "peft": peft.__version__,
             "torch": torch.__version__, "base_commit": base_commit(a.base)}
    box_train.save_merged(model, None, argparse.Namespace(model=a.base, out=a.out))   # prints TRAIN DONE
    print("MERGE FACTS " + json.dumps(facts), flush=True)   # last, so the caller's log tail always has it


if __name__ == "__main__":
    main()
