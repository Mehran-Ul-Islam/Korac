import glob, os, subprocess
from datasets import load_dataset
from tqdm import tqdm
from src.filter import keep, clean_doc

TOTAL_MB = 420
SHARES = {"tsx": .15, "ts": .10, "jsx": .15, "js": .10, "python": .15,
          "html": .10, "css": .10, "docs": .08, "wiki": .04, "design": .03}

STACK = {"tsx": "typescript", "ts": "typescript", "jsx": "jsx", "js": "javascript",
         "html": "html", "css": "css"}

DOC_REPOS = [
    ("https://github.com/mdn/content",
     ["files/en-us/web/html", "files/en-us/web/css", "files/en-us/web/javascript"]),
    ("https://github.com/reactjs/react.dev", ["src/content"]),
    ("https://github.com/remix-run/react-router", ["docs"]),
]


def stack(lang):
    ds = load_dataset("bigcode/the-stack-dedup", data_dir=f"data/{lang}",
                      split="train", streaming=True)
    for x in ds:
        yield x["content"]


def python():
    for x in load_dataset("codeparrot/codeparrot-clean-valid", split="train"):
        yield x["content"]


def wiki():
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    for x in ds:
        if len(x["text"]) > 1500:
            yield x["text"][:4000]


def docs():
    for url, subdirs in DOC_REPOS:
        dest = os.path.join("raw", url.split("/")[-1])
        if not os.path.exists(dest):
            subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)
        for sub in subdirs:
            for path in glob.glob(os.path.join(dest, sub, "**", "*.md*"), recursive=True):
                with open(path, encoding="utf-8", errors="ignore") as f:
                    yield clean_doc(f.read())


def design():
    for path in glob.glob("design/*.txt") + glob.glob("design/*.md"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            for part in f.read().split("\n<|eos|>\n"):
                yield part


LOADERS = {n: (lambda d=d: stack(d)) for n, d in STACK.items()}
LOADERS.update({"python": python, "docs": docs, "wiki": wiki, "design": design})

def fetch_all(total_mb=TOTAL_MB):
    os.makedirs("corpus", exist_ok=True)
    for name, share in SHARES.items():
        budget = int(total_mb * 1e6 * share)
        out = f"corpus/{name}.txt"
        if os.path.exists(out) and os.path.getsize(out) >= budget * 0.9:
            print("have", name)
            continue
        print("fetching", name)
        size, scanned = 0, 0
        with open(out, "w", encoding="utf-8") as f, tqdm(total=budget, unit="ch") as bar:
            for text in LOADERS[name]():
                scanned += 1
                if scanned > 150_000 and size < budget * 0.5:
                    print(f"gave up on {name}: only {size / 1e6:.1f} MB found in {scanned} files")
                    break
                if not keep(name, text):
                    continue
                f.write(text + "\n<|eos|>\n")
                size += len(text)
                bar.update(len(text))
                if size >= budget:
                    break
    os.makedirs("corpus", exist_ok=True)
    for name, share in SHARES.items():
        budget = int(total_mb * 1e6 * share)
        out = f"corpus/{name}.txt"
        if os.path.exists(out) and os.path.getsize(out) >= budget * 0.9:
            print("have", name)
            continue
        print("fetching", name)
        size = 0
        with open(out, "w", encoding="utf-8") as f, tqdm(total=budget, unit="ch") as bar:
            for text in LOADERS[name]():
                if not keep(name, text):
                    continue
                f.write(text + "\n<|eos|>\n")
                size += len(text)
                bar.update(len(text))
                if size >= budget:
                    break
