import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import load_tokenizer


@dataclass
class Config:
    vocab_size: int = 86
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.2


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).type_as(x) * self.weight


def rope_tables(head_dim, length, base=10000.0):
    powers = torch.arange(0, head_dim, 2).float() / head_dim
    inverse = 1.0 / (base ** powers)
    angles = torch.outer(torch.arange(length).float(), inverse)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    left, right = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([left * cos - right * sin, left * sin + right * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=2)
        shape = (batch, length, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight

        cos, sin = rope_tables(config.n_embd // config.n_head, config.block_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self.init_weights)
        depth_scale = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(parameter, mean=0.0, std=depth_scale)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens, targets=None):
        length = tokens.shape[1]
        cos, sin = self.cos[:length], self.sin[:length]
        x = self.drop(self.embed(tokens))
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(new_tokens):
            window = tokens[:, -self.config.block_size:]
            logits, _ = self(window)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values
                logits[logits < cutoff[:, [-1]]] = -float("inf")
            probabilities = F.softmax(logits, dim=-1)
            tokens = torch.cat([tokens, torch.multinomial(probabilities, 1)], dim=1)
        return tokens


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    torch.manual_seed(0)
    tokenizer = load_tokenizer("tokenizers/char.json")
    config = Config(vocab_size=tokenizer.vocab_size)
    device = pick_device()
    model = Transformer(config).to(device)

    print(f"device      {device}")
    print(f"vocab       {config.vocab_size}")
    print(f"layers      {config.n_layer}")
    print(f"width       {config.n_embd}")
    print(f"parameters  {model.parameter_count():,}")

    model.eval()
    inputs = torch.randint(0, config.vocab_size, (4, 64), device=device)
    targets = torch.randint(0, config.vocab_size, (4, 64), device=device)
    logits, loss = model(inputs, targets)
    expected = math.log(config.vocab_size)
    print(f"\nlogits      {tuple(logits.shape)}")
    print(f"loss        {loss.item():.4f}")
    print(f"expected    {expected:.4f}")
    print(f"difference  {abs(loss.item() - expected):.4f}")

    changed = inputs.clone()
    changed[:, -1] = (changed[:, -1] + 1) % config.vocab_size
    before, _ = model(inputs)
    after, _ = model(changed)
    drift = (before[:, :-1] - after[:, :-1]).abs().max().item()
    print(f"\ncausality   changing the last token moves earlier logits by {drift:.2e}")

    prompt = "= Quantum computing =\n\n"
    start = torch.tensor([tokenizer.encode(prompt)], device=device)
    out = model.generate(start, 200)
    print("\nuntrained output:\n")
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
