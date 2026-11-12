# gpt-on-8gb

What's the biggest GPT you can train on an 8 GB laptop GPU? This repo has the code and every measurement
from the experiments, including follow-ups comparing a 6-hour and a 12-hour training run, and a small model
trained longer against a bigger model trained for the same time.

Hardware: RTX 4060 Laptop GPU, 8 GB (PyTorch sees 7.62 GiB). bf16 autocast, fp32 weights,
GPT-2 BPE tokenizer (50,257 tokens), 1,024-token context.

## Results

| Setup | Result | File |
|---|---|---|
| No tricks, 8 × 1,024 tokens per step | 30.3M fits (peak 5.66 GiB), 51.5M runs out of memory | `results/ladder-no-tricks.jsonl` |
| GPT-2 small (124.4M), tricks added one by one | OOM → 2.98 GiB (accumulation 1 × 8) → 2.60 GiB (checkpointing) → 1.91 GiB (AdamW 8-bit) | `results/tricks-gpt2-small.jsonl` |
| All tricks on | 577.1M fits (peak 7.08 GiB), 596.8M runs out of memory | `results/ladder-all-tricks.jsonl` |
| AdamW vs AdamW 8-bit, 124M, 15 min of real text | val loss 5.798 vs 5.808 | `runs/check-124m-*/log.jsonl` |
| One night (6 h each) | 124M: best val loss 3.457 · 355M: 3.864 · 577M: 4.474 | `runs/night-*/log.jsonl` |

In 6 hours the 124M model read 446M tokens and the 577M model read 76M. So the biggest model that fits is
577M parameters, and the biggest one worth training in one night is 124M.

## 6 hours vs 12 hours

The same 124M setup as `night-124m`, trained from scratch with `--hours 12` (its own cosine schedule over 12 h).

| Run | Tokens read | tokens/s | Best val loss | File |
|---|---|---|---|---|
| 6 h (`night-124m`) | 446M | 20,662 | 3.457 | `runs/night-124m/log.jsonl` |
| 12 h (`night-124m-12h`) | 894M | 20,700 | 3.342 | `runs/night-124m-12h/log.jsonl` |

- At hour 6 the 12 h run was behind (3.677 vs 3.508): its learning rate was still high. It passed the 6 h run's best at hour 10.
- The 12 h run was paused once at 8 h 15 min and continued with `--resume`. The log keeps the `train` events of the
  ~7 minutes after the last checkpoint that the resume replays; drop those (minutes > the `resume` event's) when plotting.
  Its `peak_allocated_gib` (5.68) is higher than a run without a pause, because the checkpoint was loaded on the GPU.
- `runs/repro-124m-6h-partial/log.jsonl` is an earlier copy of the 6 h run with the same settings, stopped by a laptop
  shutdown after 3 h 39 min. Over its 14 evals it stays within 0.087 val loss of `night-124m`.
- `results/temps-12h.csv` + `temp_guard_12h.sh`: GPU temperature during the 12 h run (avg 63.5 °C, max 67 °C).

## Train longer or train bigger? (124M vs 355M, 12 h each)

The 355M setup of `night-355m`, trained from scratch with `--hours 12`, against the 124M 12 h run above.

| Run | Tokens read | tokens/s | Best / last val loss | File |
|---|---|---|---|---|
| 124M, 12 h (`night-124m-12h`) | 894M | 20,700 | 3.342 / 3.399 | `runs/night-124m-12h/log.jsonl` |
| 355M, 12 h (`night-355m-12h`) | 290M | 6,724 | 3.539 / 3.584 | `runs/night-355m-12h/log.jsonl` |
| 124M, 3.9 h (`night-124m-3.9h`) | 290M | 20,663 | 3.590 / 3.602 | `runs/night-124m-3.9h/log.jsonl` |

- Same time budget: the 124M wins by 0.41 at 6 h (3.457 vs 3.864) and by 0.20 at 12 h (3.342 vs 3.539).
  The extra 6 h gained the 355M 0.325 and the 124M 0.115. Single runs.
- From hour 6 on, the gap (355M − 124M, every 15-min eval) moves between 0.09 and 0.28 with no trend.
- `night-124m-3.9h` exists only to compare at the same text: its schedule ends at 290M tokens, like the 355M 12 h run.
  Both finished: 3.602 vs 3.584 (last eval), inside the eval noise. Per token read, the two sizes are even here;
  the 355M reads about a third as many tokens per hour.
- Tokens read per parameter in 12 h: 124M 7.2, 355M 0.82 (Hoffmann et al. 2022's compute-optimal rule of thumb is ~20).
- The 355M 12 h run used gradient checkpointing (peak 6.11 GiB) and ran without a pause.
- `results/temps-355m-12h.csv` (avg 69.9 °C, max 74 °C, avg 83.1 W) and `results/temps-124m-3.9h.csv`
  (avg 64.2 °C, max 67 °C), logged by `temp_guard_355m_12h.sh` / `temp_guard_124m_3.9h.sh`.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --name night-355m-12h \
    --n-layer 24 --n-head 16 --n-embd 1024 --batch-size 4 --accumulation 2 --checkpointing \
    --optimizer adamw8bit --lr 3e-4 --hours 12
python train.py --name night-124m-3.9h --n-layer 12 --n-head 12 --n-embd 768 --batch-size 4 \
    --accumulation 2 --optimizer adamw8bit --lr 6e-4 --hours 3.9
```

Notes on the files:
- `*-idle.jsonl` and `idle-remeasure.log` are a later re-run. The memory numbers match the first runs; the speeds
  are lower because the laptop was on battery, so the speeds reported above come from the 6-hour runs instead.
- `results/fastest-2.jsonl` is the search for the fastest setup that fits for 355M and 124M.
- `results/temps-day2.csv` is the GPU temperature log of the 355M and 124M nights (`temp_guard.sh`).
- Evals use 160 sequences, so val-loss differences under ~0.05 are noise.
- Each night size used its own fastest setup and learning rate (see the `start` line of each `log.jsonl`).

## Files

- `model.py`: the GPT (nanoGPT-style), plus optional gradient checkpointing.
- `prepare_data.py`: streams FineWeb-Edu (sample-10BT), tokenizes it with tiktoken's GPT-2 encoding, writes
  `data/train.bin` and `data/val.bin` (1B + 5M tokens by default).
- `probe.py`: a few training steps of one size on random tokens; records memory per stage, peak memory,
  tokens/s, or the OOM message.
- `ladder.py`: runs `probe.py` for each size in its own process, so an OOM never affects the next size.
- `tricks.py`: turns the tricks on one at a time for one size.
- `train.py`: trains one model for a fixed wall-clock budget (LR schedule by elapsed time), evals and samples
  every 15 min, saves `best.pt` / `last.pt`, `--resume` continues.
- `run_day2.sh`, `temp_guard.sh`: the 355M and 124M nights, and the temperature guard that ran next to them.

## Run it

```bash
pip install -r requirements.txt          # Python 3.12; torch with CUDA
python prepare_data.py                   # ~1.9 GB of tokens in data/
python ladder.py --tag no-tricks
python tricks.py --name gpt2-small --n-layer 12 --n-head 12 --n-embd 768
python ladder.py --tag all-tricks -- --batch-size 1 --accumulation 8 --checkpointing --optimizer adamw8bit --expandable
# sizes between the ladder rungs (e.g. 26 layers × 1280 = 577M) ran with probe.py directly
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --name night-577m \
    --n-layer 26 --n-head 20 --n-embd 1280 --batch-size 1 --accumulation 8 --checkpointing \
    --optimizer adamw8bit --lr 3e-4 --hours 6
./run_day2.sh                            # the 355M and 124M nights
```

The data and the checkpoints are not in this repo; `prepare_data.py` and `train.py` recreate them.

## Credits

- Model and training loop follow Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT).
- Training text: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) by Hugging Face
  (ODC-By 1.0). The dataset is not redistributed here.
- GPT-2 tokenizer via [tiktoken](https://github.com/openai/tiktoken) (MIT); 8-bit optimizer from
  [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (MIT).

Related project: [tiny-gpt-armenian](https://github.com/khachaturnalbandyan/tiny-gpt-armenian).

## License

MIT, see `LICENSE`.
