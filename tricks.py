"""Turn the memory tricks on one at a time for one model size, each step in its own process.

Every step keeps the tricks before it, and the tokens per optimizer step stay the same (8 x 1024).

Usage:
    python3 tricks.py --name gpt2-small --n-layer 12 --n-head 12 --n-embd 768
"""
import argparse
import subprocess
import sys
from pathlib import Path

TOKENS_PER_STEP_BATCH = 8

# label, extra probe.py arguments (cumulative)
STEPS = [
    ("baseline", []),
    ("+expandable-segments", ["--expandable"]),
    ("+micro-batch-1-x8", ["--expandable", "--batch-size", "1", "--accumulation", str(TOKENS_PER_STEP_BATCH)]),
    ("+checkpointing", ["--expandable", "--batch-size", "1", "--accumulation", str(TOKENS_PER_STEP_BATCH),
                        "--checkpointing"]),
    ("+adamw8bit", ["--expandable", "--batch-size", "1", "--accumulation", str(TOKENS_PER_STEP_BATCH),
                    "--checkpointing", "--optimizer", "adamw8bit"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--n-head", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    args = parser.parse_args()
    out = Path("results") / f"tricks-{args.name}.jsonl"

    for label, extra in STEPS:
        command = [sys.executable, "probe.py", "--name", f"{args.name} {label}", "--n-layer", str(args.n_layer),
                   "--n-head", str(args.n_head), "--n-embd", str(args.n_embd), "--out", str(out), *extra]
        print(f"=== {label} ===", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            print("crash", completed.stderr[-1500:], flush=True)


if __name__ == "__main__":
    main()
