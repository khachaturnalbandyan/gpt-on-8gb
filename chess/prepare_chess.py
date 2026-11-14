"""Filter a Lichess .pgn.zst month and encode the games as character tokens.

Each kept game becomes one line of plain move text, e.g. ";1.e4 e5 2.Nf3 Nc6 ...".
Output (in --out-dir): train.bin / val.bin (uint16, same format as ../train.py),
meta.json (vocab) and stats.json (counts, Elo histogram).

Usage: python prepare_chess.py data/raw/lichess_db_standard_rated_2017-01.pgn.zst --out-dir data
"""

import argparse
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

VOCAB = " #+-.0123456789;=BKNOQRabcdefghx"
STOI = {ch: i for i, ch in enumerate(VOCAB)}
GAME_START = ";"

MIN_ELO = 1800
MIN_SECONDS = 180  # Lichess estimate base + 40*increment; below this is bullet
MIN_PLIES = 20  # 10 full moves
BAD_TERMINATIONS = {"Abandoned", "Rules infraction", "Unterminated"}
VAL_EVERY = 100  # every 100th kept game goes to val (1%)
FLUSH_CHARS = 50_000_000

HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]$')
COMMENT_RE = re.compile(r"\{[^}]*\}|\([^)]*\)")
MOVE_NUMBER_RE = re.compile(r"^\d+\.+")
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def estimated_seconds(time_control):
    """'300+3' -> 420. None for '-' (correspondence) or malformed values."""
    base, sep, inc = time_control.partition("+")
    if not sep or not base.isdigit() or not inc.isdigit():
        return None
    return int(base) + 40 * int(inc)


def parse_moves(movetext):
    """PGN movetext -> list of SAN moves, without numbers, comments, NAGs, annotations or result."""
    moves = []
    for token in COMMENT_RE.sub(" ", movetext).split():
        if token in RESULTS or token.startswith("$"):
            continue
        token = MOVE_NUMBER_RE.sub("", token).rstrip("?!")  # "1.e4", "1." + "e4", "3...Nf6", "e4!?"
        if token:
            moves.append(token)
    return moves


def format_game(moves):
    """['e4','e5','Nf3'] -> ';1.e4 e5 2.Nf3'"""
    parts = []
    for ply, move in enumerate(moves):
        parts.append(f"{ply // 2 + 1}.{move}" if ply % 2 == 0 else move)
    return GAME_START + " ".join(parts)


def reject_reason(headers, moves):
    """Why a game is dropped, or None if it is kept."""
    try:
        white, black = int(headers["WhiteElo"]), int(headers["BlackElo"])
    except (KeyError, ValueError):
        return "no_elo"
    if min(white, black) < MIN_ELO:
        return "low_elo"
    seconds = estimated_seconds(headers.get("TimeControl", ""))
    if seconds is None or seconds < MIN_SECONDS:
        return "bullet_or_no_clock"
    if headers.get("Termination") in BAD_TERMINATIONS:
        return "bad_termination"
    if len(moves) < MIN_PLIES:
        return "too_short"
    return None


def iter_games(lines):
    """Yield (headers, movetext) per game from PGN lines."""
    headers, movetext = {}, []
    for raw in lines:
        line = raw.strip()
        match = HEADER_RE.match(line)
        if match:
            if movetext:  # a new game starts after a game with moves
                yield headers, " ".join(movetext)
                headers, movetext = {}, []
            headers[match.group(1)] = match.group(2)
        elif line:
            movetext.append(line)
        elif movetext:
            yield headers, " ".join(movetext)
            headers, movetext = {}, []
    if headers or movetext:
        yield headers, " ".join(movetext)


class TokenWriter:
    """Buffers game text and appends it to a uint16 .bin file."""

    def __init__(self, path):
        self.path = path
        self.file = open(path, "wb")
        self.buffer = []
        self.buffered = 0
        self.total = 0

    def add(self, text):
        self.buffer.append(text)
        self.buffered += len(text)
        if self.buffered >= FLUSH_CHARS:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        text = "".join(self.buffer)
        ids = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        np.asarray(LOOKUP[ids], dtype=np.uint16).tofile(self.file)
        self.total += len(text)
        self.buffer, self.buffered = [], 0

    def close(self):
        self.flush()
        self.file.close()


LOOKUP = np.full(256, 0, dtype=np.uint16)
for _ch, _i in STOI.items():
    LOOKUP[ord(_ch)] = _i


def decompressed_lines(path):
    proc = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    yield from proc.stdout
    if proc.wait() != 0:
        raise RuntimeError(f"zstd failed on {path} (exit {proc.returncode})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn_zst", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-games", type=int, default=None, help="stop after reading this many games (test)")
    args = parser.parse_args()
    if not args.pgn_zst.exists():
        parser.error(f"not found: {args.pgn_zst}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = TokenWriter(args.out_dir / "train.bin")
    val = TokenWriter(args.out_dir / "val.bin")
    reasons, elo_hist, plies_kept = Counter(), Counter(), []
    read = kept = 0
    started = time.time()

    for headers, movetext in iter_games(decompressed_lines(args.pgn_zst)):
        read += 1
        moves = parse_moves(movetext)
        reason = reject_reason(headers, moves)
        text = format_game(moves) if reason is None else ""
        if reason is None and any(ch not in STOI for ch in text):
            reason = "bad_chars"
        if reason:
            reasons[reason] += 1
        else:
            (val if kept % VAL_EVERY == 0 else train).add(text + " ")
            kept += 1
            plies_kept.append(len(moves))
            avg_elo = (int(headers["WhiteElo"]) + int(headers["BlackElo"])) // 2
            elo_hist[avg_elo // 100 * 100] += 1
        if read % 1_000_000 == 0:
            print(f"{read:,} read, {kept:,} kept, {time.time() - started:.0f} s", flush=True)
        if args.max_games and read >= args.max_games:
            break

    train.close()
    val.close()
    stats = {
        "source": args.pgn_zst.name,
        "filters": {"min_elo_both": MIN_ELO, "min_estimated_seconds": MIN_SECONDS,
                    "min_plies": MIN_PLIES, "dropped_terminations": sorted(BAD_TERMINATIONS)},
        "games_read": read,
        "games_kept": kept,
        "rejected": dict(reasons),
        "train_chars": train.total,
        "val_chars": val.total,
        "mean_plies_kept": round(float(np.mean(plies_kept)), 1) if plies_kept else 0,
        "avg_elo_histogram": {str(k): v for k, v in sorted(elo_hist.items())},
        "seconds": round(time.time() - started),
    }
    (args.out_dir / "meta.json").write_text(json.dumps({"vocab": VOCAB, "vocab_size": len(VOCAB),
                                                        "dtype": "uint16", "game_start": GAME_START}, indent=2))
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
