"""A tiny synthetic optimization smoke test, not a pretrained model."""

from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

from .model import TinyLanguageModel, cache_bytes
from .spec import load_preset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("olmo2", "gemma3"))
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.manual_seed(51)
    spec = load_preset(args.family)
    model = TinyLanguageModel(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(spec.vocab_size, (2, 9))
    for step in range(args.steps):
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, spec.vocab_size), ids[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"{args.family} step={step + 1} synthetic_loss={loss.item():.4f}")
    print(f"trainable_parameters={sum(p.numel() for p in model.parameters())}")
    print(f"analytical_raw_kv_bytes_at_32_tokens={cache_bytes(spec, 1, 32, 4)}")


if __name__ == "__main__":
    main()
