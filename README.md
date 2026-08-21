# playground-tinywiki

I'm training a language model from scratch on three English Wikipedia articles.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Get the text

```bash
.venv/bin/python prepare_data.py
```

## The text

English Wikipedia articles: Artificial intelligence, Manhattan Project, and Quantum computing.

Pulled as plain text and kept the section headings. Deleted the references and external links sections. Accented letters and symbols got folded down to plain ASCII, and the rare leftovers got thrown out. I also deleted the math that was in the quantum article.

I also cut a chunk out of the middle of each article and never train on it. That's how I will tell whether the model actually learned English or just memorized the pages.

214,518 characters to train on. 27,209 held back. 86 different characters.

## Tokenizing

```bash
.venv/bin/python tokenizer.py
```

Turns the text into numbers and saves the results to `tokenizers/`.

## The model

```bash
.venv/bin/python model.py
```

A simple transformer. 3.2 million parameters.

## License

The code is MIT. See LICENSE.

The article text in `data/` is not mine and is not MIT. It comes from English Wikipedia and is licensed CC BY-SA 4.0. If you pass it on, keep it under that license and keep the attribution.

- https://en.wikipedia.org/wiki/Artificial_intelligence
- https://en.wikipedia.org/wiki/Manhattan_Project
- https://en.wikipedia.org/wiki/Quantum_computing
