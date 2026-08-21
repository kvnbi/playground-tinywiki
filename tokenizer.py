import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TOKENIZER_DIR = ROOT / "tokenizers"

PRETOKEN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+"
)

VOCAB_SIZES = [256, 512, 1024, 2048]


def apply_merge(symbols, pair, merged):
    out = []
    i = 0
    limit = len(symbols)
    while i < limit:
        if i < limit - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class CharTokenizer:
    kind = "char"

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for i, t in enumerate(self.tokens)}

    @classmethod
    def train(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        payload = {"kind": self.kind, "tokens": self.tokens}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["tokens"])


class BPETokenizer:
    kind = "bpe"

    def __init__(self, base, merges):
        self.base = list(base)
        self.merges = [tuple(m) for m in merges]
        self.tokens = self.base + [a + b for a, b in self.merges]
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for i, t in enumerate(self.tokens)}
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.cache = {}

    @classmethod
    def train(cls, text, vocab_size):
        base = sorted(set(text))
        counts = Counter(PRETOKEN.findall(text))
        splits = {word: tuple(word) for word in counts}
        merges = []

        while len(base) + len(merges) < vocab_size:
            pairs = Counter()
            for word, freq in counts.items():
                symbols = splits[word]
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq
            if not pairs:
                break
            best, freq = pairs.most_common(1)[0]
            if freq < 2:
                break
            merged = best[0] + best[1]
            merges.append(best)
            splits = {w: apply_merge(s, best, merged) for w, s in splits.items()}

        return cls(base, merges)

    def truncate(self, vocab_size):
        return BPETokenizer(self.base, self.merges[: vocab_size - len(self.base)])

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode_chunk(self, chunk):
        cached = self.cache.get(chunk)
        if cached is not None:
            return cached
        symbols = list(chunk)
        while len(symbols) >= 2:
            best_rank = None
            best_index = None
            for i in range(len(symbols) - 1):
                rank = self.ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = i
            if best_index is None:
                break
            symbols[best_index : best_index + 2] = [
                symbols[best_index] + symbols[best_index + 1]
            ]
        ids = [self.stoi[s] for s in symbols if s in self.stoi]
        self.cache[chunk] = ids
        return ids

    def encode(self, text):
        ids = []
        for chunk in PRETOKEN.findall(text):
            ids.extend(self.encode_chunk(chunk))
        return ids

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        payload = {"kind": self.kind, "base": self.base, "merges": self.merges}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["base"], payload["merges"])


def load_tokenizer(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["kind"] == "char":
        return CharTokenizer(payload["tokens"])
    return BPETokenizer(payload["base"], payload["merges"])


def unigram_bits_per_char(tokenizer, train_text, val_text):
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    counts = Counter(train_ids)
    denominator = len(train_ids) + tokenizer.vocab_size
    bits = 0.0
    for token_id in val_ids:
        probability = (counts.get(token_id, 0) + 1) / denominator
        bits -= math.log2(probability)
    return bits / len(val_text)


def report(name, tokenizer, train_text, val_text):
    ids = tokenizer.encode(train_text)
    lossless = tokenizer.decode(ids) == train_text
    chars_per_token = len(train_text) / len(ids)
    bpc = unigram_bits_per_char(tokenizer, train_text, val_text)
    counts = Counter(ids)
    starved = sum(1 for t in range(tokenizer.vocab_size) if counts.get(t, 0) < 10)
    print(f"  {name:<10} {tokenizer.vocab_size:>6,} {len(ids):>13,} "
          f"{chars_per_token:>12.2f} {'ok' if lossless else 'LOSSY':>10} "
          f"{bpc:>9.3f} {starved:>8,} {starved / tokenizer.vocab_size:>7.0%}")
    return lossless


def main():
    TOKENIZER_DIR.mkdir(exist_ok=True)
    train_text = (DATA_DIR / "train.txt").read_text(encoding="utf-8")
    val_text = (DATA_DIR / "val.txt").read_text(encoding="utf-8")

    chunks = PRETOKEN.findall(train_text)
    if "".join(chunks) != train_text:
        raise SystemExit("pre-tokenizer is lossy, refusing to continue")
    print(f"pre-tokenizer: {len(chunks):,} chunks, {len(set(chunks)):,} unique, lossless")

    print(f"\ntraining BPE to {max(VOCAB_SIZES):,} tokens on the training split only")
    full = BPETokenizer.train(train_text, max(VOCAB_SIZES))
    print(f"learned {len(full.merges):,} merges")

    print(f"\n  {'tokenizer':<10} {'vocab':>6} {'train tokens':>13} "
          f"{'chars/token':>12} {'roundtrip':>10} {'BPC':>9} {'starved':>8} {'':>7}")

    char = CharTokenizer.train(train_text)
    char.save(TOKENIZER_DIR / "char.json")
    all_ok = report("char", char, train_text, val_text)

    for size in VOCAB_SIZES:
        tokenizer = full.truncate(size)
        tokenizer.save(TOKENIZER_DIR / f"bpe-{size}.json")
        all_ok = report(f"bpe-{size}", tokenizer, train_text, val_text) and all_ok

    if not all_ok:
        raise SystemExit("a tokenizer failed the roundtrip check")

    print(f"\n  saved to {TOKENIZER_DIR.name}/")
    sample = "= Quantum computing =\n\nA quantum computer is a machine"
    bpe = full.truncate(1024)
    print(f"\n  sample under bpe-1024:")
    print("  " + " | ".join(bpe.decode([i]) for i in bpe.encode(sample)))


if __name__ == "__main__":
    main()
