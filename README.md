# Karoc

A ~105M-parameter language model for web code (HTML, CSS, JS/JSX, TS/TSX, Python), built from scratch in PyTorch: its own byte-level BPE tokenizer, a GPT-style decoder, and a simple resumable training loop.

> Learning project. Expect short, imperfect completions, not a replacement for a large coding assistant.

## Project layout

```
main.py             command entry point
requirements.txt
src/
  tokenizer.py      byte-level BPE with chat special tokens
  filter.py         quality filters (drops minified or generated files)
  sources.py        downloads and filters the training corpus
  data.py           trains the tokenizer, builds train.bin
  model.py          GPT-style transformer and configs
  train.py          training loop (resumes from ckpt.pt)
  generate.py       text generation
design/             design rules and generated palette examples
```

## Model

| Setting | Value |
|---|---|
| Layers / heads / width | 14 / 12 / 768 |
| Context length | 512 tokens |
| Vocabulary | 8,192 (own BPE) |
| Parameters | about 105.8M |

## Setup

```bash
# CPU-only PyTorch (smaller download); skip this line on a GPU machine
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The code data comes from The Stack, which is gated on Hugging Face:

1. Accept the terms on the `bigcode/the-stack-dedup` dataset page.
2. Create an access token and expose it as `HF_TOKEN`.
   In Codespaces: GitHub Settings, Codespaces, Secrets.

## Build the data

```bash
python main.py fetch 420     # about 420 MB of filtered text
python main.py tokenizer     # trains the BPE tokenizer, writes tok.json
python main.py data          # writes train.bin (about 112M tokens)
```

`corpus/`, `raw/`, `train.bin` and checkpoints are git-ignored. GitHub rejects files over 100 MB, so rebuild the data on each machine.

## Train

```bash
nohup python main.py train 6000 > train.log 2>&1 &
tail -f train.log
```

- Saves `ckpt.pt` every 100 steps. Run the same command again to resume.
- Keep the step count the same when resuming, because the learning-rate schedule depends on it.
- A step is 16,384 tokens. 6000 steps is about 98M tokens.

## Generate

```bash
python main.py generate "import React from 'react';"
```

## Speed notes

| Machine | Rough speed |
|---|---|
| Ryzen 5 5600G (CPU) | about 400-490 tokens/s, so about 2.5 days for 6000 steps |
| Free GPU notebook | likely much faster; measure with the `tok/s` log line |

## Data sources

Code from The Stack (permissively licensed), MDN, React and React Router docs, Wikipedia, and generated design examples. Check each source's license before publishing weights.

## Roadmap

- [x] Own tokenizer, data pipeline, resumable training
- [ ] Pretraining run (6000 steps)
- [ ] Instruction tuning (request to code pairs)
- [ ] Small agent loop around the model