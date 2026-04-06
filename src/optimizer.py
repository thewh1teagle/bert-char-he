"""Optimizer and LR schedule for MLM pretraining."""

from __future__ import annotations

import math

import torch
from model import NeoBERTForMLM


def cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: NeoBERTForMLM, lr: float, weight_decay: float) -> torch.optim.AdamW:
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"}
    return torch.optim.AdamW([
        {
            "params": [p for n, p in model.named_parameters() if not any(t in n for t in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(t in n for t in no_decay)],
            "weight_decay": 0.0,
        },
    ], lr=lr)


def build_scheduler(optimizer: torch.optim.AdamW, warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, warmup_steps, total_steps),
    )
