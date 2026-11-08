import sys
from src import data, train, generate, sources

cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

if cmd == "fetch":
    sources.fetch_all(float(sys.argv[2]) if len(sys.argv) > 2 else sources.TOTAL_MB)
elif cmd == "tokenizer":
    data.train_tokenizer()
elif cmd == "data":
    data.build_bin()
elif cmd == "train":
    train.run(max_steps=int(sys.argv[2])) if len(sys.argv) > 2 else train.run()
elif cmd == "generate":
    generate.run(sys.argv[2] if len(sys.argv) > 2 else "def add(a, b):")
else:
    print("usage: python main.py [fetch [MB] | tokenizer | data | train [steps] | generate \"prompt\"]")
