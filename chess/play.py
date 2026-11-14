"""Play a chess-GPT checkpoint against Stockfish and log every game.

The model never sees a board: each turn it gets the game so far as text (";1.e4 e5 2.") and writes
the next move character by character until a space. python-chess only checks the move.
An illegal move is retried up to --retries times; after that the model forfeits (counted as a loss).

Usage:
    python play.py runs/chess-50m-12h/ckpt_h12.pt --skills 0 1 2 --games 10     # uncalibrated levels
    python play.py runs/chess-50m-12h/ckpt_h12.pt --elo 1320 1500 1700 --games 100  # Elo estimate
Writes results/games-<run>-<ckpt>-<skill|elo>.jsonl (one line per game) + .summary.json:
win/draw/loss per level, and with --elo a maximum-likelihood Elo estimate with its standard error.
Forfeits (illegal moves) count as losses.
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import torch

from prepare_chess import format_game
from train_chess import GPT, GPTConfig

STOCKFISH = "/usr/games/stockfish"
MIN_SF_ELO, MAX_SF_ELO = 1320, 3190  # UCI_Elo range of Stockfish 17.1
# UCI_Elo is "calibrated at a time control of 120s+1.0s" and anchored to CCRL Blitz to +-100 Elo
# (Stockfish README, commit a08b8d4e9711c2). In --elo mode Stockfish gets that clock; the model has none.
ELO_CLOCK = "120+1"
MIN_CLOCK_S = 0.1
MAX_MOVE_CHARS = 10  # longest SAN is like "exd8=Q+" or "Nbxd7#"
MAX_PLIES = 400  # adjudicated as a draw after this
MOVE_END = {" ", ";"}


def move_prompt(san_moves):
    """Text the model continues: ';1.' at the start, ';1.e4 ' for black, ';1.e4 e5 2.' for white."""
    if not san_moves:
        return ";1."
    text = format_game(san_moves)
    if len(san_moves) % 2 == 0:
        return f"{text} {len(san_moves) // 2 + 1}."
    return text + " "


def load_model(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["vocab"], state


class ModelPlayer:
    def __init__(self, model, vocab, device, temperature, generator):
        self.model, self.vocab, self.device = model, vocab, device
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.temperature = temperature
        self.generator = generator

    @torch.no_grad()
    def sample_move(self, prompt):
        """Characters after the prompt, up to the first space or ';'."""
        ids = [self.stoi[ch] for ch in prompt]
        block = self.model.config.block_size
        move = ""
        for _ in range(MAX_MOVE_CHARS):
            tokens = torch.tensor([ids[-block:]], device=self.device)
            logits = self.model(tokens)[0][0, -1].float()
            if self.temperature == 0:
                next_id = int(logits.argmax())
            else:
                probs = torch.softmax(logits / self.temperature, dim=-1).cpu()
                next_id = int(torch.multinomial(probs, 1, generator=self.generator))
            ch = self.vocab[next_id]
            if ch in MOVE_END:
                break
            move += ch
            ids.append(next_id)
        return move


def legal_move(board, text):
    """The move if `text` is a legal SAN move on `board`, else None."""
    try:
        move = board.parse_san(text)
    except ValueError:  # illegal, ambiguous or not a move at all
        return None
    return move if move and board.is_legal(move) else None  # "--" parses as a null move, which is falsy


def model_turn(player, board, san_moves, retries):
    """Returns (move or None, illegal attempts as text)."""
    prompt = move_prompt(san_moves)
    illegal = []
    for _ in range(retries + 1):
        text = player.sample_move(prompt)
        move = legal_move(board, text)
        if move:
            return move, illegal
        illegal.append(text)
    return None, illegal


def opponents(skills, elos):
    """(label, kind, value, UCI options) per Stockfish setting. Both modes set every option, so nothing leaks."""
    if elos:
        return [(f"elo {e}", "elo", e, {"UCI_LimitStrength": True, "UCI_Elo": e, "Skill Level": 20}) for e in elos]
    return [(f"skill {k}", "skill", k, {"UCI_LimitStrength": False, "Skill Level": k}) for k in skills]


def expected_score(rating, opponent):
    return 1 / (1 + 10 ** ((opponent - rating) / 400))


def score(counts):
    games = counts["win"] + counts["draw"] + counts["loss"]
    return (counts["win"] + 0.5 * counts["draw"]) / games if games else None


def elo_from_score(opponent, s):
    """Rating that scores s against `opponent`. None at 0 or 1 (no finite answer)."""
    if s is None or s <= 0 or s >= 1:
        return None
    return opponent + 400 * math.log10(s / (1 - s))


def combined_elo(levels):
    """Maximum-likelihood rating over several opponents: levels = [(opponent elo, games, score)].
    Returns (rating, standard error) or (None, None) if the model scored 0 or 100 % everywhere."""
    total_games = sum(n for _, n, _ in levels)
    points = sum(n * s for _, n, s in levels)
    if total_games == 0 or points <= 0 or points >= total_games:
        return None, None
    low, high = -2000.0, 5000.0
    for _ in range(100):  # the expected points rise with the rating, so bisection finds the match
        mid = (low + high) / 2
        if sum(n * expected_score(mid, opp) for opp, n, _ in levels) < points:
            low = mid
        else:
            high = mid
    rating = (low + high) / 2
    information = sum(n * expected_score(rating, opp) * (1 - expected_score(rating, opp)) for opp, n, _ in levels)
    return rating, 400 / math.log(10) / math.sqrt(information)


def parse_clock(text):
    """'120+1' -> (120.0, 1.0)."""
    base, sep, inc = text.partition("+")
    try:
        return float(base), float(inc) if sep else 0.0
    except ValueError:
        raise ValueError(f"clock must look like 120+1, got {text!r}") from None


def play_game(player, engine, options, model_color, retries, sf_time=None, sf_clock=None):
    """Stockfish thinks either a fixed sf_time per move, or on its own clock sf_clock=(base, increment)."""
    engine.configure(options)
    remaining = sf_clock[0] if sf_clock else None
    low_clock = False
    board = chess.Board()
    san_moves, illegal_log = [], []
    reason = None
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome:
            reason = outcome.termination.name.lower()
            break
        if len(san_moves) >= MAX_PLIES:
            reason = "max_plies"
            break
        if board.turn == model_color:
            move, illegal = model_turn(player, board, san_moves, retries)
            if illegal:
                illegal_log.append({"ply": len(san_moves), "fen": board.fen(), "attempts": illegal,
                                    "played": move is not None})
            if move is None:
                reason = "forfeit_illegal"
                break
        else:
            if sf_clock:
                limit = chess.engine.Limit(white_clock=remaining, black_clock=remaining,
                                           white_inc=sf_clock[1], black_inc=sf_clock[1])
                started = time.perf_counter()
                move = engine.play(board, limit).move
                remaining = remaining - (time.perf_counter() - started) + sf_clock[1]
                if remaining < MIN_CLOCK_S:  # no loss on time for Stockfish; just note it
                    remaining, low_clock = MIN_CLOCK_S, True
            else:
                move = engine.play(board, chess.engine.Limit(time=sf_time)).move
        san_moves.append(board.san(move))
        board.push(move)

    if reason == "forfeit_illegal":
        result = "loss"
    elif outcome and outcome.winner is not None:
        result = "win" if outcome.winner == model_color else "loss"
    else:
        result = "draw"
    return {"model_color": "white" if model_color == chess.WHITE else "black",
            "result": result, "reason": reason, "plies": len(san_moves),
            "model_moves": sum(1 for ply in range(len(san_moves)) if (ply % 2 == 0) == (model_color == chess.WHITE)),
            "illegal_turns": len(illegal_log), "illegal": illegal_log, "moves": format_game(san_moves),
            "sf_clock_left": round(remaining, 1) if sf_clock else None, "sf_low_clock": low_clock}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skills", type=int, nargs="+", default=[0], help="Stockfish Skill Level 0-20 (not calibrated)")
    mode.add_argument("--elo", type=int, nargs="+", default=None,
                      help=f"Stockfish UCI_Elo levels {MIN_SF_ELO}-{MAX_SF_ELO} (calibrated); prints an Elo estimate")
    parser.add_argument("--games", type=int, default=10, help="per level, colours alternate")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sf-time", type=float, default=None,
                        help="fixed Stockfish seconds per move (default 0.05 for --skills; --elo uses --sf-clock)")
    parser.add_argument("--sf-clock", default=None,
                        help=f"Stockfish clock base+increment in seconds (default {ELO_CLOCK} for --elo, its calibration)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        parser.error(f"not found: {args.checkpoint}")
    if args.elo and any(not MIN_SF_ELO <= e <= MAX_SF_ELO for e in args.elo):
        parser.error(f"--elo must be between {MIN_SF_ELO} and {MAX_SF_ELO}")
    if not args.elo and any(not 0 <= k <= 20 for k in args.skills):
        parser.error("--skills must be between 0 and 20")
    kind = "elo" if args.elo else "skill"
    if args.sf_time is not None and args.sf_clock:
        parser.error("use --sf-time or --sf-clock, not both")
    try:
        sf_clock = parse_clock(args.sf_clock or ELO_CLOCK) if (args.sf_clock or (args.elo and args.sf_time is None)) else None
    except ValueError as error:
        parser.error(str(error))
    sf_time = None if sf_clock else (args.sf_time if args.sf_time is not None else 0.05)

    generator = torch.Generator().manual_seed(args.seed)
    model, vocab, state = load_model(args.checkpoint, args.device)
    player = ModelPlayer(model, vocab, args.device, args.temperature, generator)
    out = args.out or Path("results") / f"games-{args.checkpoint.parent.name}-{args.checkpoint.stem}-{kind}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"checkpoint": str(args.checkpoint), "step": state.get("step"), "minutes": state.get("minutes"),
            "temperature": args.temperature, "retries": args.retries, "sf_time": sf_time,
            "sf_clock": f"{sf_clock[0]:g}+{sf_clock[1]:g}" if sf_clock else None}

    summary = {}
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH) as engine, out.open("w", encoding="utf-8") as file:
        for label, opponent_kind, value, options in opponents(args.skills, args.elo):
            counts = Counter()
            for game in range(args.games):
                model_color = chess.WHITE if game % 2 == 0 else chess.BLACK
                record = {**meta, "opponent": opponent_kind, opponent_kind: value, "game": game,
                          **play_game(player, engine, options, model_color, args.retries, sf_time, sf_clock)}
                file.write(json.dumps(record) + "\n")
                file.flush()
                counts[record["result"]] += 1
                counts["forfeits"] += record["reason"] == "forfeit_illegal"
                print(f"{label} game {game} {record['model_color']}: {record['result']} "
                      f"({record['reason']}, {record['plies']} plies, {record['illegal_turns']} illegal turns)",
                      flush=True)
            level = {k: counts[k] for k in ("win", "draw", "loss", "forfeits")}
            level["score"] = score(counts)
            if opponent_kind == "elo":
                single = elo_from_score(value, level["score"])
                level["elo_estimate"] = round(single) if single is not None else None
            summary[label] = level

    report = {"checkpoint": str(args.checkpoint), "games_file": str(out), "sf_time": sf_time,
              "sf_clock": meta["sf_clock"], "per_level": summary}
    if args.elo:
        rating, error = combined_elo([(e, args.games, summary[f"elo {e}"]["score"]) for e in args.elo])
        report["elo_estimate"] = round(rating) if rating is not None else None
        report["elo_standard_error"] = round(error) if error is not None else None
        if rating is None:
            report["note"] = (f"no finite estimate: scored 0 % (weaker than every level, below {min(args.elo)})"
                              " or 100 % everywhere")
    print(json.dumps(report, indent=2))
    (out.with_suffix(".summary.json")).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
