import torch
from src.model import GPT, CFG
from src.tokenizer import Tokenizer


def run(prompt):
    tok = Tokenizer.load("tok.json")
    model = GPT(CFG)
    model.load_state_dict(torch.load("ckpt.pt", map_location="cpu")["model"])
    model.eval()
    idx = torch.tensor([tok.encode(prompt)])
    out = model.generate(idx, max_new=100)
    print(tok.decode(out[0].tolist()))
