import argparse
from pathlib import Path

import torch

from model import Config, Transformer, pick_device
from tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parent
TOKENIZER_DIR = ROOT / "tokenizers"


def load_checkpoint(path, device):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model = Transformer(Config(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    tokenizer = load_tokenizer(TOKENIZER_DIR / f"{state['tokenizer']}.json")
    return model, tokenizer, state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/model.pt")
    parser.add_argument("--title", default="Quantum computing")
    parser.add_argument("--length", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = pick_device()
    model, tokenizer, state = load_checkpoint(args.checkpoint, device)
    top_k = args.top_k if args.top_k > 0 else None

    prompt = f"= {args.title} =\n\n"
    start = torch.tensor([tokenizer.encode(prompt)] * args.samples, device=device)
    out = model.generate(start, args.length, temperature=args.temperature, top_k=top_k)

    for row in out:
        print(tokenizer.decode(row.tolist()))
        print()


if __name__ == "__main__":
    main()
