#!/usr/bin/env python3
"""Runs INSIDE a GGUF box (s2/gguf_run.py): merge a trained LoRA adapter into its public base, on the GPU when the
weights fit there with room to spare and on the host otherwise (a 35B is about 70 GB of bf16; an RTX 5090 has 32 GB),
then leave the folder exactly as box_train leaves it (box_train.save_merged: the base's own config and tokenizer, no
MTP head), so the GGUF is made from the same merged weights the nvfp4 build was.

  python gguf_box.py --base /workspace/base --adapter /workspace/out/adapter --out /workspace/out [--device auto]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def weights_bytes(folder):
    return sum(os.path.getsize(os.path.join(folder, n)) for n in os.listdir(folder) if n.endswith(".safetensors"))


def pick_device(weights, vram):
    """cuda when the bf16 weights take at most two thirds of the card, less 8 GB for the merge's own buffers."""
    return "cuda" if vram >= 1.5 * weights + 8e9 else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()
    import torch
    import transformers
    from peft import PeftModel
    import box_train
    vram = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    w = weights_bytes(a.base)
    dev = pick_device(w, vram) if a.device == "auto" else a.device
    print(f"MERGE on {dev}: weights {w / 1e9:.1f} GB, GPU {vram / 1e9:.1f} GB", flush=True)
    cfg = transformers.AutoConfig.from_pretrained(a.base)
    model = getattr(transformers, cfg.architectures[0]).from_pretrained(a.base, torch_dtype=torch.bfloat16,
                                                                        device_map=dev)
    box_train.save_merged(PeftModel.from_pretrained(model, a.adapter), None,
                          argparse.Namespace(model=a.base, out=a.out))   # prints TRAIN DONE when the folder is whole


if __name__ == "__main__":
    main()
