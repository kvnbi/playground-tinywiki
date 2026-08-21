import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from model import Config, Transformer, pick_device
from tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TOKENIZER_DIR = ROOT / "tokenizers"
CHECKPOINT_DIR = ROOT / "checkpoints"


def get_batch(ids, block_size, batch_size, device):
    start = torch.randint(0, len(ids) - block_size - 1, (batch_size,), device=device)
    offsets = torch.arange(block_size, device=device)
    index = start[:, None] + offsets
    return ids[index], ids[index + 1]


def learning_rate_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, ids, tokenizer, block_size):
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    covered = []
    for start in range(0, len(ids) - block_size - 1, block_size):
        x = ids[start:start + block_size].unsqueeze(0)
        y = ids[start + 1:start + 1 + block_size].unsqueeze(0)
        _, loss = model(x, y)
        total_nats += loss.item() * y.numel()
        total_tokens += y.numel()
        covered.append(y)
    model.train()
    characters = len(tokenizer.decode(torch.cat(covered, dim=1)[0].tolist()))
    return total_nats / math.log(2) / characters, total_nats / total_tokens


def longest_verbatim(generated, source):
    best = ""
    for i in range(len(generated)):
        length = len(best) + 1
        while i + length <= len(generated) and generated[i:i + length] in source:
            best = generated[i:i + length]
            length += 1
    return best


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="char")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    device = pick_device()
    tokenizer = load_tokenizer(TOKENIZER_DIR / f"{args.tokenizer}.json")
    train_text = (DATA_DIR / "train.txt").read_text(encoding="utf-8")
    val_text = (DATA_DIR / "val.txt").read_text(encoding="utf-8")
    train_ids = torch.tensor(tokenizer.encode(train_text), device=device)
    val_ids = torch.tensor(tokenizer.encode(val_text), device=device)

    config = Config(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.width,
        dropout=args.dropout,
    )
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    name = args.name or f"{args.tokenizer}-{args.layers}L-{args.width}w"
    checkpoint_path = CHECKPOINT_DIR / f"{name}.pt"
    tokens_per_step = args.batch * args.block
    steps_per_epoch = max(1, len(train_ids) // tokens_per_step)

    print(f"run          {name}")
    print(f"device       {device}")
    print(f"tokenizer    {args.tokenizer}, vocab {tokenizer.vocab_size:,}")
    print(f"model        {config.n_layer} layers, {config.n_embd} wide, "
          f"{model.parameter_count():,} parameters")
    print(f"train tokens {len(train_ids):,}   held out {len(val_ids):,}")
    print(f"one pass     {steps_per_epoch} steps")
    print()

    best_bpc = float("inf")
    stale = 0
    started = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        lr = learning_rate_at(step, args.steps, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_ids, config.block_size, args.batch, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            bpc, val_nats = evaluate(model, val_ids, tokenizer, config.block_size)
            improved = bpc < best_bpc
            if improved:
                best_bpc = bpc
                stale = 0
                torch.save({
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "tokenizer": args.tokenizer,
                    "bpc": bpc,
                    "step": step,
                }, checkpoint_path)
            else:
                stale += 1
            if not args.quiet:
                mark = "  best" if improved else ""
                print(f"  step {step:>5}  epoch {step/steps_per_epoch:>5.1f}  "
                      f"train {loss.item():.3f}  held out {val_nats:.3f}  "
                      f"bpc {bpc:.3f}{mark}")
            if stale >= args.patience:
                print(f"\n  stopping, no improvement for {args.patience} checks")
                break

    elapsed = time.time() - started
    print(f"\nbest held out bpc  {best_bpc:.4f}")
    print(f"time               {elapsed:.0f}s")
    print(f"saved              {checkpoint_path.relative_to(ROOT)}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    prompt = "= Quantum computing =\n\n"
    start = torch.tensor([tokenizer.encode(prompt)], device=device)
    sample = tokenizer.decode(model.generate(start, 400, temperature=0.8, top_k=40)[0].tolist())
    copied = longest_verbatim(sample[len(prompt):], train_text)
    print(f"longest copy       {len(copied)} characters")
    if copied:
        print(f"                   {copied[:90]!r}")
    print(f"\nsample from step {state['step']}:\n")
    print(sample)


if __name__ == "__main__":
    main()
