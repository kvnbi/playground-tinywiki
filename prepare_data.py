import json
import re
import ssl
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import certifi

ARTICLES = ["Artificial intelligence", "Manhattan Project", "Quantum computing"]

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data"

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

DROP_SECTIONS = {
    "references", "external links", "further reading", "see also", "notes",
    "citations", "bibliography", "sources", "footnotes", "works cited",
    "general references", "explanatory notes", "general and cited references",
}

HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")

CHAR_MAP = {
    "\u2060": "", "\u2061": "", "\u00ad": "", "\u200b": "",
    "\u2011": "-", "\u2212": "-",
    "×": "x", "ø": "o",
    "α": "alpha", "β": "beta", "δ": "delta",
    "ψ": "psi", "Ω": "Omega",
    "°C": " degrees C", "°F": " degrees F", "°": " degrees",
}

PROTECTED = set(" \n") | set(map(chr, range(0x20, 0x7F)))
RARE_THRESHOLD = 10
MATH_LINE_MAX = 12
HOLDOUT_FRACTION = 0.075
HOLDOUT_START = 0.45


def fetch(title):
    cache = RAW_DIR / (title.replace(" ", "_") + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    params = urllib.parse.urlencode({
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "extracts", "explaintext": "1", "redirects": "1", "titles": title,
    })
    url = "https://en.wikipedia.org/w/api.php?" + params
    req = urllib.request.Request(url, headers={
        "User-Agent": "playground-tinywiki/0.1 (educational language-model project)"
    })
    with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
        data = json.load(resp)

    page = data["query"]["pages"][0]
    if "extract" not in page:
        raise SystemExit(f"No article text returned for {title!r}")
    text = page["extract"]
    cache.write_text(text, encoding="utf-8")
    return text


def strip_junk_sections(text):
    kept, dropping_at_depth = [], None
    for line in text.split("\n"):
        match = HEADING.match(line)
        if match:
            depth, name = len(match.group(1)), match.group(2).strip().lower()
            if dropping_at_depth is not None and depth <= dropping_at_depth:
                dropping_at_depth = None
            if name in DROP_SECTIONS:
                dropping_at_depth = depth
        if dropping_at_depth is None:
            kept.append(line)
    return "\n".join(kept)


def fold_characters(text):
    for source, replacement in CHAR_MAP.items():
        text = text.replace(source, replacement)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def drop_math_residue(text):
    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if (stripped
                and not HEADING.match(line)
                and len(stripped) <= MATH_LINE_MAX
                and not re.search(r"[A-Za-z]{3}", stripped)):
            continue
        kept.append(line)
    return "\n".join(kept)


def clean(text, title):
    text = fold_characters(strip_junk_sections(text))
    text = text.replace("\r\n", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = drop_math_residue(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return f"= {title} =\n\n{text.strip()}\n"


def drop_rare_characters(articles):
    counts = Counter("".join(articles.values()))
    doomed = {c for c, n in counts.items()
              if c not in PROTECTED and n < RARE_THRESHOLD}
    if doomed:
        table = str.maketrans({c: "" for c in doomed})
        articles = {t: text.translate(table) for t, text in articles.items()}
    return articles, sorted(doomed)


def split_holdout(text, fraction=HOLDOUT_FRACTION, start_at=HOLDOUT_START):
    length = len(text)
    start = int(length * start_at)
    end = start + int(length * fraction)

    previous_break = text.rfind("\n\n", 0, start)
    if previous_break != -1:
        start = previous_break + 2
    next_break = text.find("\n\n", end)
    if next_break != -1:
        end = next_break

    return text[:start] + text[end:], text[start:end]


def stats(label, text):
    words = text.split()
    unique = len({w.lower() for w in words})
    print(f"  {label:<24} {len(text):>8,} chars  {len(words):>7,} words  "
          f"{unique:>6,} unique  {len(set(text)):>4} distinct characters")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Articles:")
    articles = {t: clean(fetch(t), t) for t in ARTICLES}
    articles, dropped = drop_rare_characters(articles)
    for title, text in articles.items():
        stats(title, text)

    if dropped:
        print(f"\n  dropped {len(dropped)} vestigial characters: "
              + " ".join(repr(c) for c in dropped))

    train_parts, val_parts = [], []
    for text in articles.values():
        train_part, val_part = split_holdout(text)
        train_parts.append(train_part)
        val_parts.append(val_part)

    train, val = "\n\n".join(train_parts), "\n\n".join(val_parts)
    (OUT_DIR / "train.txt").write_text(train, encoding="utf-8")
    (OUT_DIR / "val.txt").write_text(val, encoding="utf-8")

    print()
    stats("TRAIN", train)
    stats("HELD-OUT", val)
    unseen = sorted(set(val) - set(train))
    share = len(val) / (len(train) + len(val))
    print(f"\n  held-out share:                    {share:.1%}")
    print(f"  held-out chars unseen in training: {unseen if unseen else 'none'}")
    print("  written to data/train.txt and data/val.txt")


if __name__ == "__main__":
    main()
