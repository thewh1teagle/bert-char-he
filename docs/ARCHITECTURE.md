# Architecture

## Goal

Pretrain a Hebrew character-level BERT encoder using Masked Language Modeling (MLM) on raw unvocalized Hebrew text. The pretrained encoder can then be loaded into a downstream G2P model.

## Model

`NeoBERTForMLM` in `src/model.py`:

1. **Encoder** — NeoBERT initialized from scratch (~19M params). 6 layers, 512 hidden, 8 heads, SwiGLU, RoPE, Pre-RMSNorm. See `src/encoder.py`.
2. **MLM head** — single linear layer: `hidden_size → vocab_size` (104 classes).

## Tokenizer

Character-level, 104-token vocab (identical to renikud):
- 5 special tokens: `[PAD] [CLS] [SEP] [UNK] [MASK]`
- Hebrew letters א–ת (including final forms) + maqaf, geresh, gershayim
- ASCII lowercase, digits, punctuation, space

Built in-memory from `src/tokenization.py` — no external file needed.

## Training

See [TRAINING.md](TRAINING.md).
