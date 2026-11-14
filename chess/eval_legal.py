"""Percent legal moves per checkpoint, on positions from held-out (val) games.

For each of --positions val games, pick one random ply, give the model the real game up to that ply
as text, let it write one move (first try, no retries) and check it with python-chess.
Same positions and same sampling seed for every checkpoint, so hours are comparable.

Usage:
    python eval_legal.py runs/chess-50m-12h            # every ckpt_h*.pt in the run
    python eval_legal.py runs/chess-50m-12h/ckpt_h03.pt
Writes results/legal-<run>-p<positions>.csv (one row per checkpoint; rows for already measured checkpoints are kept).
"""
import argparse
import csv
import json
import random
from pathlib import Path

import chess
import numpy as np
import torch

from play import ModelPlayer, legal_move, load_model, move_prompt
from prepare_chess import parse_moves

# game phase by ply, to see where the model breaks
PHASES = {"opening": (0, 20), "middle": (20, 60), "late": (60, 10_000)}
FIELDS = ["checkpoint", "minutes", "step", "val_loss", "positions", "legal_pct",
          *[f"legal_pct_{name}" for name in PHASES], *[f"positions_{name}" for name in PHASES]]


def val_games(data_dir, count, seed):
    """`count` random val games as SAN move lists (games with at least 2 plies)."""
    vocab = json.loads((data_dir / "meta.json").read_text())["vocab"]
    ids = np.fromfile(data_dir / "val.bin", dtype=np.uint16)
    text = "".join(np.array(list(vocab))[ids])
    games = [parse_moves(game) for game in text.split(";") if game.strip()]
    games = [game for game in games if len(game) >= 2]
    return random.Random(seed).sample(games, min(count, len(games)))


def pick_positions(games, seed):
    """One (moves before, board) per game at a random ply."""
    rng = random.Random(seed)
    positions = []
    for game in games:
        ply = rng.randrange(len(game))
        board = chess.Board()
        for san in game[:ply]:
            board.push_san(san)
        positions.append((game[:ply], board))
    return positions


def phase_of(ply):
    return next(name for name, (low, high) in PHASES.items() if low <= ply < high)


def measure(checkpoint, positions, args):
    model, vocab, state = load_model(checkpoint, args.device)
    player = ModelPlayer(model, vocab, args.device, args.temperature, torch.Generator().manual_seed(args.seed))
    legal = {name: 0 for name in PHASES}
    total = {name: 0 for name in PHASES}
    for moves, board in positions:
        phase = phase_of(len(moves))
        total[phase] += 1
        legal[phase] += legal_move(board, player.sample_move(move_prompt(moves))) is not None
    row = {"checkpoint": checkpoint.name, "minutes": state.get("minutes"), "step": state.get("step"),
           "val_loss": round(state["val_loss"], 4) if state.get("val_loss") is not None else None,
           "positions": len(positions),
           "legal_pct": round(100 * sum(legal.values()) / len(positions), 1)}
    for name in PHASES:
        row[f"legal_pct_{name}"] = round(100 * legal[name] / total[name], 1) if total[name] else None
        row[f"positions_{name}"] = total[name]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="a run dir (all ckpt_h*.pt) or one checkpoint")
    parser.add_argument("--positions", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.7, help="same as play.py")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.target.exists():
        parser.error(f"not found: {args.target}")

    checkpoints = sorted(args.target.glob("ckpt_h*.pt")) if args.target.is_dir() else [args.target]
    if not checkpoints:
        parser.error(f"no ckpt_h*.pt in {args.target}")
    run_name = args.target.name if args.target.is_dir() else args.target.parent.name
    out = args.out or Path("results") / f"legal-{run_name}-p{args.positions}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = {}
    if out.exists():
        with out.open() as file:
            rows = {row["checkpoint"]: row for row in csv.DictReader(file)}
    positions = pick_positions(val_games(args.data_dir, args.positions, args.seed), args.seed)
    for checkpoint in checkpoints:
        if checkpoint.name in rows:
            print(f"{checkpoint.name}: already measured, skipped")
            continue
        rows[checkpoint.name] = measure(checkpoint, positions, args)
        print(json.dumps(rows[checkpoint.name]), flush=True)
        with out.open("w", newline="") as file:  # rewrite after each checkpoint, so progress is kept
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows[name] for name in sorted(rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
