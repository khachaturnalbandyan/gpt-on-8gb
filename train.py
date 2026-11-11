"""Train one GPT on FineWeb-Edu tokens for a fixed wall-clock budget (the one-night comparison).

The learning rate follows the elapsed fraction of the budget (warmup, then cosine), so every size
gets a complete schedule in the same time. Logs JSON lines to <out-dir>/log.jsonl. After every eval it saves last.pt; --resume continues from it,
with the clock picking up where it stopped (time spent down does not count against the budget).

Usage:
    python3 train.py --name gpt2-small --n-layer 12 --n-head 12 --n-embd 768 --hours 6
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from probe import VOCAB_SIZE, make_optimizer

SEED = 1337
GRAD_CLIP = 1.0
EVAL_ROWS = 160  # sequences per eval, split into micro-batches of the training size
WARMUP_FRACTION = 0.02
MIN_LR_RATIO = 0.1
SAMPLE_TOKENS = 200
SAMPLE_PROMPT = "The most important thing about learning is"


def get_batch(data, batch_size, block_size):
    starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    rows = np.stack([data[start : start + block_size + 1] for start in starts]).astype(np.int64)
    tokens = torch.from_numpy(rows).pin_memory().to("cuda", non_blocking=True)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def learning_rate(fraction, max_lr):
    if fraction < WARMUP_FRACTION:
        return max_lr * max(fraction, 1e-3) / WARMUP_FRACTION
    progress = min(1.0, (fraction - WARMUP_FRACTION) / (1 - WARMUP_FRACTION))
    min_lr = max_lr * MIN_LR_RATIO
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def autocast():
    return torch.amp.autocast("cuda", dtype=torch.bfloat16)


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size):
    model.eval()
    losses = []
    for _ in range(max(1, EVAL_ROWS // batch_size)):
        inputs, targets = get_batch(data, batch_size, block_size)
        with autocast():
            _, loss = model(inputs, targets)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_text(model, encoder):
    model.eval()
    generator_state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(SEED)  # same randomness for every model and every eval
    start = torch.tensor([encoder.encode_ordinary(SAMPLE_PROMPT)], device="cuda")
    with autocast():
        tokens = model.generate(start, SAMPLE_TOKENS)[0].tolist()
    torch.cuda.set_rng_state(generator_state)
    model.train()
    return encoder.decode(tokens)


def load_tokens(path):
    if not path.exists():
        raise SystemExit(f"{path} not found. Run prepare_data.py first.")
    return np.memmap(path, dtype=np.uint16, mode="r")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--n-head", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1, help="micro-batch")
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw8bit")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--eval-minutes", type=float, default=15)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=None, help="default: runs/<name>")
    parser.add_argument("--resume", action="store_true", help="continue from <out-dir>/last.pt")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    import tiktoken

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    encoder = tiktoken.get_encoding("gpt2")
    train_data = load_tokens(args.data_dir / "train.bin")
    val_data = load_tokens(args.data_dir / "val.bin")
    out_dir = args.out_dir or Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=args.block_size, n_layer=args.n_layer,
                       n_head=args.n_head, n_embd=args.n_embd, gradient_checkpointing=args.checkpointing)
    model = GPT(config).cuda()
    optimizer = make_optimizer(args.optimizer, model)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    tokens_per_step = args.batch_size * args.accumulation * args.block_size
    run_info = {"event": "start", "name": args.name, "parameters": parameters, "config": config.to_dict(),
                "batch_size": args.batch_size, "accumulation": args.accumulation, "optimizer": args.optimizer,
                "lr": args.lr, "hours": args.hours, "tokens_per_step": tokens_per_step,
                "device": torch.cuda.get_device_name()}

    def log(record):
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    budget_seconds = args.hours * 3600
    next_eval = 0.0
    step = 0
    best_val_loss = float("inf")
    elapsed_before = 0.0
    if args.resume:
        state = torch.load(out_dir / "last.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, elapsed_before = state["step"], state["elapsed"]
        best_val_loss, next_eval = state["best_val_loss"], state["next_eval"]
        np.random.set_state(state["numpy_rng"])
        run_info.update(event="resume", step=step, minutes=round(elapsed_before / 60, 2))
    log(run_info)
    started = time.perf_counter() - elapsed_before

    def save_last(elapsed):
        temporary = out_dir / "last.pt.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "elapsed": elapsed, "best_val_loss": best_val_loss, "next_eval": next_eval,
                    "numpy_rng": np.random.get_state()}, temporary)
        temporary.replace(out_dir / "last.pt")  # atomic, so a crash mid-save keeps the previous one

    def evaluate(elapsed):
        nonlocal best_val_loss
        val_loss = estimate_loss(model, val_data, args.batch_size, args.block_size)
        record = {"event": "eval", "step": step, "tokens_seen": step * tokens_per_step,
                  "minutes": round(elapsed / 60, 2), "val_loss": round(val_loss, 4),
                  "sample": sample_text(model, encoder)}
        log(record)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "config": config.to_dict(), "step": step,
                        "val_loss": val_loss}, out_dir / "best.pt")

    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= budget_seconds:
            break
        if elapsed >= next_eval:
            evaluate(elapsed)
            next_eval += args.eval_minutes * 60
            save_last(elapsed)
        current_lr = learning_rate(elapsed / budget_seconds, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        for _ in range(args.accumulation):
            inputs, targets = get_batch(train_data, args.batch_size, args.block_size)
            with autocast():
                _, loss = model(inputs, targets)
            (loss / args.accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % 100 == 0:
            log({"event": "train", "step": step, "loss": round(loss.item(), 4), "lr": round(current_lr, 7),
                 "minutes": round(elapsed / 60, 2)})

    elapsed = time.perf_counter() - started
    evaluate(elapsed)
    log({"event": "end", "steps": step, "tokens_seen": step * tokens_per_step, "minutes": round(elapsed / 60, 2),
         "tokens_per_sec": round(step * tokens_per_step / elapsed), "best_val_loss": round(best_val_loss, 4),
         "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)})


if __name__ == "__main__":
    main()
