import os
import numpy as np
from tqdm import tqdm
from src.tokenizer import Tokenizer
from src.sources import SHARES


def train_tokenizer(vocab_size=8192, path="tok.json", tok_mb=20):
    parts = []
    for name, share in SHARES.items():
        p = f"corpus/{name}.txt"
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                parts.append(f.read(int(tok_mb * 1e6 * share)))
    text = "\n".join(parts).replace("<|eos|>", "\n")
    tok = Tokenizer()
    tok.train(text, vocab_size)
    tok.save(path)
    print("saved", path, "vocab size", tok.vocab_size)


def build_bin(tok_path="tok.json", out="train.bin"):
    tok = Tokenizer.load(tok_path)
    total = 0
    with open(out, "wb") as f:
        for name in SHARES:
            p = f"corpus/{name}.txt"
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as src:
                docs = src.read().split("\n<|eos|>\n")
            buf = []
            for d in tqdm(docs, desc=name):
                buf.extend(tok.encode(d))
                buf.append(tok.eos_id)
                if len(buf) > 1_000_000:
                    np.array(buf, dtype=np.uint16).tofile(f)
                    total += len(buf)
                    buf = []
            if buf:
                np.array(buf, dtype=np.uint16).tofile(f)
                total += len(buf)
    print(total, "tokens saved")
