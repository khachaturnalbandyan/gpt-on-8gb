"""Train a character-level chess GPT on Lichess move text for a fixed wall-clock budget.

Same recipe as ../train.py (warmup + cosine LR over --hours, eval every --eval-minutes,
last.pt for --resume), reusing its model and LR schedule. Chess changes:
- vocab (32 chars) comes from <data-dir>/meta.json, samples start from ";1.e4",
- a checkpoint ckpt_hNN.pt every --checkpoint-minutes (default 60) plus ckpt_h00.pt at step 0,
  so each hour of training can be tested later (legal moves, games vs Stockfish).

Usage:
    python train_chess.py --name chess-25m --n-layer 8 --n-head 8 --n-embd 512 --hours 12
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]          # the repo root: model.py, probe.py, train.py
sys.path.insert(0, str(REPO_ROOT))
from model import GPT, GPTConfig  # noqa: E402
from probe import make_optimizer  # noqa: E402
from train import GRAD_CLIP, SEED, learning_rate  # noqa: E402

EVAL_ROWS = 160
SAMPLE_CHARS = 300
SAMPLE_PROMPT = ";1.e4"


def load_meta(data_dir):
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found. Run prepare_chess.py first.")
    return json.loads(meta_path.read_text())


def load_tokens(path):
    if not path.exists():
        raise SystemExit(f"{path} not found. Run prepare_chess.py first.")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, block_size, device):
    starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    rows = torch.from_numpy(np.stack([data[s : s + block_size + 1] for s in starts]).astype(np.int64))
    if device == "cuda":
        rows = rows.pin_memory().to(device, non_blocking=True)
    return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def autocast(device):
    return torch.amp.autocast(device, dtype=torch.bfloat16)


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, device):
    model.eval()
    losses = []
    for _ in range(max(1, EVAL_ROWS // batch_size)):
        inputs, targets = get_batch(data, batch_size, block_size, device)
        with autocast(device):
            _, loss = model(inputs, targets)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_text(model, vocab, device):
    """Same seed every time, so samples from different hours are comparable."""
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
    torch.manual_seed(SEED)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    start = torch.tensor([[stoi[ch] for ch in SAMPLE_PROMPT]], device=device)
    with autocast(device):
        ids = model.generate(start, SAMPLE_CHARS)[0].tolist()
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    model.train()
    return "".join(vocab[i] for i in ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--n-head", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16, help="micro-batch")
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw")
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--eval-minutes", type=float, default=15)
    parser.add_argument("--checkpoint-minutes", type=float, default=60)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=None, help="default: runs/<name>")
    parser.add_argument("--resume", action="store_true", help="continue from <out-dir>/last.pt")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="cpu only for smoke tests")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required (or pass --device cpu for a smoke test).")
    if args.device == "cpu" and args.optimizer == "adamw8bit":
        raise SystemExit("adamw8bit needs CUDA.")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    meta = load_meta(args.data_dir)
    vocab = meta["vocab"]
    train_data = load_tokens(args.data_dir / "train.bin")
    val_data = load_tokens(args.data_dir / "val.bin")
    out_dir = args.out_dir or Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    config = GPTConfig(vocab_size=meta["vocab_size"], block_size=args.block_size, n_layer=args.n_layer,
                       n_head=args.n_head, n_embd=args.n_embd, gradient_checkpointing=args.checkpointing)
    model = GPT(config).to(args.device)
    optimizer = make_optimizer(args.optimizer, model) if args.device == "cuda" else \
        torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    parameters = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.batch_size * args.accumulation * args.block_size
    run_info = {"event": "start", "name": args.name, "parameters": parameters, "config": config.to_dict(),
                "batch_size": args.batch_size, "accumulation": args.accumulation, "optimizer": args.optimizer,
                "lr": args.lr, "hours": args.hours, "tokens_per_step": tokens_per_step,
                "train_chars": len(train_data), "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu"}

    def log(record):
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    budget_seconds = args.hours * 3600
    checkpoint_seconds = args.checkpoint_minutes * 60
    step, elapsed_before = 0, 0.0
    next_eval, next_checkpoint = 0.0, 0.0
    best_val_loss = float("inf")
    if args.resume:
        state = torch.load(out_dir / "last.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, elapsed_before = state["step"], state["elapsed"]
        best_val_loss, next_eval = state["best_val_loss"], state["next_eval"]
        next_checkpoint = state["next_checkpoint"]
        np.random.set_state(state["numpy_rng"])
        run_info.update(event="resume", step=step, minutes=round(elapsed_before / 60, 2))
    log(run_info)
    started = time.perf_counter() - elapsed_before

    def model_file(val_loss):
        return {"model": model.state_dict(), "config": config.to_dict(), "vocab": vocab, "step": step,
                "tokens_seen": step * tokens_per_step, "val_loss": val_loss}

    def save_last(elapsed):
        temporary = out_dir / "last.pt.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "elapsed": elapsed, "best_val_loss": best_val_loss, "next_eval": next_eval,
                    "next_checkpoint": next_checkpoint, "numpy_rng": np.random.get_state()}, temporary)
        temporary.replace(out_dir / "last.pt")  # atomic

    def evaluate(elapsed):
        nonlocal best_val_loss
        val_loss = estimate_loss(model, val_data, args.batch_size, args.block_size, args.device)
        tokens_seen = step * tokens_per_step
        log({"event": "eval", "step": step, "tokens_seen": tokens_seen, "epochs": round(tokens_seen / len(train_data), 3),
             "minutes": round(elapsed / 60, 2), "val_loss": round(val_loss, 4),
             "sample": sample_text(model, vocab, args.device)})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_file(val_loss), out_dir / "best.pt")
        return val_loss

    def save_checkpoint(elapsed, val_loss):
        # numbered by the schedule, not the clock, so the final save never overwrites an hourly one
        index = round(next_checkpoint / checkpoint_seconds)
        path = out_dir / f"ckpt_h{index:02d}.pt"
        torch.save({**model_file(val_loss), "minutes": round(elapsed / 60, 2)}, path)
        log({"event": "checkpoint", "file": path.name, "step": step, "minutes": round(elapsed / 60, 2)})

    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= budget_seconds:
            break
        due_checkpoint = elapsed >= next_checkpoint
        if elapsed >= next_eval or due_checkpoint:
            val_loss = evaluate(elapsed)
            if elapsed >= next_eval:
                next_eval += args.eval_minutes * 60
            if due_checkpoint:
                save_checkpoint(elapsed, val_loss)
                next_checkpoint += checkpoint_seconds
            save_last(elapsed)
        current_lr = learning_rate(elapsed / budget_seconds, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        for _ in range(args.accumulation):
            inputs, targets = get_batch(train_data, args.batch_size, args.block_size, args.device)
            with autocast(args.device):
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
    val_loss = evaluate(elapsed)
    save_checkpoint(elapsed, val_loss)
    tokens_seen = step * tokens_per_step
    end = {"event": "end", "steps": step, "tokens_seen": tokens_seen, "epochs": round(tokens_seen / len(train_data), 3),
           "minutes": round(elapsed / 60, 2), "tokens_per_sec": round(tokens_seen / elapsed),
           "best_val_loss": round(best_val_loss, 4)}
    if args.device == "cuda":
        end["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    log(end)


if __name__ == "__main__":
    main()
