"""Run probe.py for every size on the ladder, each in its own process (an OOM never poisons the next run).

Usage:
    python3 ladder.py --tag no-tricks
    python3 ladder.py --tag all-tricks -- --batch-size 1 --accumulation 8 --checkpointing --optimizer adamw8bit
"""
import argparse
import subprocess
import sys
from pathlib import Path

# name, n_layer, n_head, n_embd. GPT-2 shapes where they exist.
LADDER = [
    ("v1-shape", 6, 6, 384),
    ("small-50m", 8, 8, 512),
    ("gpt2-small", 12, 12, 768),
    ("gpt2-medium", 24, 16, 1024),
    ("mid-24x1280", 24, 20, 1280),
    ("mid-30x1280", 30, 20, 1280),
    ("gpt2-large", 36, 20, 1280),
    ("gpt2-xl", 48, 25, 1600),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="results go to results/ladder-<tag>.jsonl")
    parser.add_argument("--stop-after-oom", type=int, default=1, help="stop after this many OOMs in a row")
    parser.add_argument("probe_args", nargs=argparse.REMAINDER, help="extra arguments passed to probe.py after --")
    args = parser.parse_args()
    extra = [arg for arg in args.probe_args if arg != "--"]
    out = Path("results") / f"ladder-{args.tag}.jsonl"

    ooms = 0
    for name, n_layer, n_head, n_embd in LADDER:
        command = [sys.executable, "probe.py", "--name", name, "--n-layer", str(n_layer),
                   "--n-head", str(n_head), "--n-embd", str(n_embd), "--out", str(out), *extra]
        print(f"=== {name} ===", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        status = "oom" if '"status": "oom"' in completed.stdout else ("ok" if completed.returncode == 0 else "crash")
        print(status, completed.stdout[-600:] if status != "crash" else completed.stderr[-1500:], flush=True)
        ooms = ooms + 1 if status != "ok" else 0
        if ooms >= args.stop_after_oom:
            print(f"stopping: {ooms} failure(s) in a row", flush=True)
            break


if __name__ == "__main__":
    main()
