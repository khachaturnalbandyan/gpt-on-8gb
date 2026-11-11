"""Measure one model size on the GPU: memory per training stage, peak memory, speed, or the OOM.

Runs a few training steps on random tokens (memory and speed do not depend on the text).
Appends one JSON line to --out, so a crash in one size never loses the others.

Usage:
    python3 probe.py --name gpt2-small --n-layer 12 --n-head 12 --n-embd 768
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig

SEED = 1337
VOCAB_SIZE = 50257  # GPT-2 BPE, the tokenizer used for FineWeb-Edu
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.1
LEARNING_RATE = 3e-4
WARMUP_STEPS = 2
GIB = 1024**3


def gib(value):
    return round(value / GIB, 3)


def allocated():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def make_optimizer(name, model):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True)
    if name == "adamw8bit":
        import bitsandbytes

        return bitsandbytes.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    raise SystemExit(f"unknown optimizer {name}")


def train_step(model, optimizer, batch_size, block_size, accumulation, stages=None):
    """One optimizer step. If stages is a dict, fill it with memory after each stage of the first micro-batch."""
    for micro in range(accumulation):
        tokens = torch.randint(0, VOCAB_SIZE, (batch_size, block_size + 1), device="cuda")
        inputs, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(inputs, targets)
        if stages is not None and micro == 0:
            stages["after_forward"] = allocated()
        (loss / accumulation).backward()
        if stages is not None and micro == 0:
            stages["after_backward"] = allocated()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if stages is not None:
        stages["after_optimizer_step"] = allocated()


def probe(args):
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        gradient_checkpointing=args.checkpointing,
    )
    result = {"name": args.name, "config": config.to_dict(), "batch_size": args.batch_size,
              "accumulation": args.accumulation, "optimizer": args.optimizer, "expandable_segments": args.expandable,
              "device": torch.cuda.get_device_name(), "total_gib": gib(torch.cuda.get_device_properties(0).total_memory)}
    stages = {}
    try:
        model = GPT(config).cuda()
        result["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        stages["weights"] = allocated()
        optimizer = make_optimizer(args.optimizer, model)
        train_step(model, optimizer, args.batch_size, args.block_size, args.accumulation, stages)
        for _ in range(WARMUP_STEPS - 1):
            train_step(model, optimizer, args.batch_size, args.block_size, args.accumulation)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.steps):
            train_step(model, optimizer, args.batch_size, args.block_size, args.accumulation)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tokens = args.steps * args.accumulation * args.batch_size * args.block_size
        result.update(status="ok", tokens_per_sec=round(tokens / elapsed), step_seconds=round(elapsed / args.steps, 3))
    except torch.OutOfMemoryError as error:
        result.update(status="oom", error=str(error).splitlines()[0])
    result["stages_gib"] = {stage: gib(value) for stage, value in stages.items()}
    result["peak_allocated_gib"] = gib(torch.cuda.max_memory_allocated())
    result["peak_reserved_gib"] = gib(torch.cuda.max_memory_reserved())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--n-head", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw")
    parser.add_argument("--expandable", action="store_true", help="PyTorch allocator: expandable_segments (less fragmentation)")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/probe.jsonl"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the memory probe.")
    if args.expandable:
        # read by the CUDA allocator on its first allocation, so it must be set before any tensor hits the GPU
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.manual_seed(SEED)

    result = probe(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
