"""Stream FineWeb-Edu, tokenize with the GPT-2 BPE tokenizer, write train/val token files.

Documents are separated by the end-of-text token. Tokens are written in chunks, so memory stays flat.

Usage:
    python3 prepare_data.py --train-tokens 1000000000 --val-tokens 5000000
"""
import argparse
import json
from pathlib import Path

import numpy as np

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
ENCODE_BATCH = 1024
LOG_EVERY_TOKENS = 50_000_000


def token_batches(texts, encoder, batch_size=ENCODE_BATCH):
    """Yield uint16 arrays, one per batch of documents, each document followed by end-of-text."""
    batch = []
    for text in texts:
        batch.append(text)
        if len(batch) == batch_size:
            yield encode_batch(batch, encoder)
            batch = []
    if batch:
        yield encode_batch(batch, encoder)


def encode_batch(texts, encoder):
    documents = encoder.encode_ordinary_batch(texts)
    flat = [token for document in documents for token in (*document, encoder.eot_token)]
    return np.array(flat, dtype=np.uint16)


def write_tokens(batches, path, limit):
    """Write batches to path until limit tokens; return the count written. The rest of the last batch is dropped."""
    written = 0
    next_log = LOG_EVERY_TOKENS
    with path.open("wb") as file:
        for tokens in batches:
            chunk = tokens[: limit - written]
            file.write(chunk.tobytes())
            written += len(chunk)
            if written >= next_log:
                print(f"{path.name}: {written:,} tokens", flush=True)
                next_log += LOG_EVERY_TOKENS
            if written >= limit:
                break
    return written


def load_texts():
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)
    return (row["text"] for row in dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--val-tokens", type=int, default=5_000_000)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    batches = token_batches(load_texts(), encoder)
    # val comes first in the stream, so a longer train run later never overlaps it
    val_count = write_tokens(batches, args.out_dir / "val.bin", args.val_tokens)
    train_count = write_tokens(batches, args.out_dir / "train.bin", args.train_tokens)
    if train_count < args.train_tokens:
        print(f"warning: stream ended early, train has {train_count:,} of {args.train_tokens:,} tokens")

    meta = {"dataset": f"{DATASET_NAME}/{DATASET_CONFIG}", "tokenizer": "gpt2", "vocab_size": encoder.n_vocab,
            "train_tokens": train_count, "val_tokens": val_count}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
