import math, os, time
import numpy as np
import torch
from tqdm import tqdm
from src.model import GPT, CFG


def run(max_steps=6000, batch_size=8, accum=4, lr_max=6e-4, lr_min=6e-5, warmup=200):
    torch.set_num_threads(os.cpu_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(CFG).to(device)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params on {device}")

    data = np.memmap("train.bin", dtype=np.uint16, mode="r")
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.1)

    start = 0
    if os.path.exists("ckpt.pt"):
        ck = torch.load("ckpt.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print("resumed from step", start)

    def get_batch():
        ix = np.random.randint(0, len(data) - CFG.block_size - 1, batch_size)
        x = torch.stack([torch.from_numpy(data[i:i + CFG.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + CFG.block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    def lr_at(s):
        if s < warmup:
            return lr_max * (s + 1) / warmup
        p = min(1.0, (s - warmup) / max(1, max_steps - warmup))
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))

    def save(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, "ckpt.tmp")
        os.replace("ckpt.tmp", "ckpt.pt")

    tokens_per_step = batch_size * accum * CFG.block_size
    t0 = time.time()
    for step in tqdm(range(start, max_steps), initial=start, total=max_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(accum):
            x, y = get_batch()
            _, loss = model(x, y)
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 10 == 0:
            tps = (step - start + 1) * tokens_per_step / (time.time() - t0)
            tqdm.write(f"step {step} loss {loss.item():.3f} | {tps:.0f} tok/s")
        if step % 100 == 0 and step > start:
            save(step + 1)
    save(max_steps)
