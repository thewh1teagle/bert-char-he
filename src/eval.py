"""Evaluation for MLM pretraining: loss and masked token accuracy."""

from __future__ import annotations

import torch


def evaluate(model, eval_loader, accelerator) -> dict[str, float]:
    model.eval()
    total_loss, total_correct, total_masked, steps = 0.0, 0, 0, 0

    with torch.no_grad():
        for batch in eval_loader:
            with accelerator.autocast():
                out = model(**batch)

            total_loss += out["loss"].item()
            steps += 1

            labels = batch["labels"]
            mask = labels != -100
            preds = out["logits"].argmax(-1)
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_masked += mask.sum().item()

    model.train()
    return {
        "eval_loss": total_loss / max(1, steps),
        "eval_acc": total_correct / max(1, total_masked),
    }
