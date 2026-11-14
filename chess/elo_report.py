"""Merge play.py game files (e.g. from parallel workers) into one report per Stockfish level.

With --elo games it adds the per-level Elo and a maximum-likelihood Elo over all levels.

Usage:
    python elo_report.py results/elo/chess-50m-12h-ckpt_h12/worker-*.jsonl --out results/elo/chess-50m-12h-ckpt_h12/summary.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from play import combined_elo, elo_from_score, score

RESULTS = ("win", "draw", "loss")


def load_games(paths):
    games = []
    for path in paths:
        with path.open(encoding="utf-8") as file:
            games.extend(json.loads(line) for line in file if line.strip())
    return games


def build_report(games):
    if not games:
        raise ValueError("no games")
    kinds = {game["opponent"] for game in games}
    checkpoints = {game["checkpoint"] for game in games}
    if len(kinds) != 1 or len(checkpoints) != 1:
        raise ValueError(f"mixed files: opponents {sorted(kinds)}, checkpoints {sorted(checkpoints)}")
    kind = kinds.pop()

    by_level = defaultdict(Counter)
    for game in games:
        counts = by_level[game[kind]]
        counts[game["result"]] += 1
        counts[f"{game['result']}_as_{game['model_color']}"] += 1
        counts["forfeits"] += game["reason"] == "forfeit_illegal"
        counts["plies"] += game["plies"]

    per_level = {}
    for level in sorted(by_level):
        counts = by_level[level]
        games_played = sum(counts[r] for r in RESULTS)
        row = {r: counts[r] for r in RESULTS}
        row.update(games=games_played, forfeits=counts["forfeits"], score=score(counts),
                   mean_plies=round(counts["plies"] / games_played, 1),
                   as_white={r: counts[f"{r}_as_white"] for r in RESULTS},
                   as_black={r: counts[f"{r}_as_black"] for r in RESULTS})
        if kind == "elo":
            single = elo_from_score(level, row["score"])
            row["elo_estimate"] = round(single) if single is not None else None
        per_level[f"{kind} {level}"] = row

    report = {"checkpoint": checkpoints.pop(), "opponent": kind, "games": len(games),
              "sf_clock": games[0].get("sf_clock"), "sf_time": games[0].get("sf_time"), "per_level": per_level}
    if kind == "elo":
        rating, error = combined_elo([(level, per_level[f"elo {level}"]["games"], per_level[f"elo {level}"]["score"])
                                      for level in sorted(by_level)])
        report["elo_estimate"] = round(rating) if rating is not None else None
        report["elo_standard_error"] = round(error) if error is not None else None
        if rating is None:
            report["note"] = "no finite estimate: the model scored 0 % at every level, or 100 % at every level"
        report["elo_scale_note"] = "Stockfish UCI_Elo is anchored to CCRL Blitz only to about +-100 Elo"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    missing = [str(path) for path in args.files if not path.exists()]
    if missing:
        parser.error(f"not found: {missing}")
    try:
        report = build_report(load_games(args.files))
    except ValueError as error:
        parser.error(str(error))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
