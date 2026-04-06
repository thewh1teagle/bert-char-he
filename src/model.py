"""NeoBERT MLM model for Hebrew pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn
from encoder import build_encoder


class NeoBERTForMLM(nn.Module):
    def __init__(self, vocab_size: int, dropout_rate: float = 0.1, flash_attention: bool = False) -> None:
        super().__init__()
        self.encoder = build_encoder(flash_attention=flash_attention)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.dropout(
            self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        )
        logits = self.mlm_head(hidden)  # [B, S, vocab_size]

        output: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            output["loss"] = loss

        return output
