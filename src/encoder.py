"""ModernBERT encoder for Hebrew G2P, initialized from scratch.

Architecture: ModernBERT configured to ~20M params:
  - 6 layers, 512 hidden, 8 heads, 1536 FFN
  - RoPE positional embeddings
  - ALiBi-free, uses SDPA
  - 4096 token context

Vocab size matches the tokenizer built in tokenization.py.
"""

from __future__ import annotations

from transformers import ModernBertConfig, ModernBertModel

from tokenization import build_vocab


def build_encoder() -> ModernBertModel:
    config = ModernBertConfig(
        vocab_size=len(build_vocab()),
        pad_token_id=0,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1536,
        max_position_embeddings=4096,
    )
    return ModernBertModel(config)
