import json, re

PAT = re.compile(r"\s*\w+|\s*[^\w\s]+|\s+")
SPECIAL = ["<|eos|>", "<|user|>", "<|assistant|>", "<|file|>", "<|end|>"]
SPECIAL_PAT = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL) + ")")


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special = {}
        self.cache = {}

    def _add_special(self):
        for s in SPECIAL:
            sid = len(self.vocab)
            self.vocab[sid] = s.encode("utf-8")
            self.special[s] = sid

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def eos_id(self):
        return self.special["<|eos|>"]

    def train(self, text, vocab_size):
        counts = {}
        for w in PAT.findall(text):
            key = tuple(w.encode("utf-8"))
            counts[key] = counts.get(key, 0) + 1
        words = [list(k) for k in counts]
        freqs = list(counts.values())

        pair_counts, where = {}, {}
        for i, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_counts[p] = pair_counts.get(p, 0) + freqs[i]
                where.setdefault(p, set()).add(i)

        for new_id in range(256, vocab_size - len(SPECIAL)):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            for i in where.pop(best, ()):
                w, f = words[i], freqs[i]
                for p in zip(w, w[1:]):
                    left = pair_counts.get(p, 0) - f
                    if left > 0:
                        pair_counts[p] = left
                    else:
                        pair_counts.pop(p, None)
                w = merge(w, best, new_id)
                words[i] = w
                for p in zip(w, w[1:]):
                    pair_counts[p] = pair_counts.get(p, 0) + f
                    where.setdefault(p, set()).add(i)
            if new_id % 200 == 0:
                print(f"merge {new_id}/{vocab_size}")
        self._add_special()

    def _encode_word(self, word):
        if word in self.cache:
            return self.cache[word]
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = merge(ids, best, self.merges[best])
        self.cache[word] = ids
        return ids

    def encode(self, text):
        ids = []
        for part in SPECIAL_PAT.split(text):
            if part in self.special:
                ids.append(self.special[part])
            elif part:
                for w in PAT.findall(part):
                    ids.extend(self._encode_word(w))
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump([list(p) for p in self.merges], f)

    @classmethod
    def load(cls, path):
        t = cls()
        with open(path) as f:
            merges = json.load(f)
        for i, (a, b) in enumerate(merges):
            t.merges[(a, b)] = 256 + i
            t.vocab[256 + i] = t.vocab[a] + t.vocab[b]
        t._add_special()
        return t
