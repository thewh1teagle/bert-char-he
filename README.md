# bert-char-he

Hebrew character-level BERT (~20M params) pretrained with MLM on raw unvocalized Hebrew text.

## Features

- **Character-level** — 104-token vocab (Hebrew letters, ASCII, punctuation). No subword tokenization.
- **NeoBERT architecture** — RoPE embeddings, SwiGLU activation, Pre-RMSNorm, full attention every layer
- **ONNX exportable** — set `NEOBERT_ONNX_EXPORT=1` to switch to ONNX-compatible ops
- **Compact** — 6 layers, 512 hidden, 8 heads, 2048 FFN, 4096 token context
- **Downstream-ready** — used as encoder in [renikud](https://github.com/thewh1teagle/renikud) for Hebrew G2P

## Usage

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("thewh1teagle/bert-char-he")
model = AutoModel.from_pretrained("thewh1teagle/bert-char-he", trust_remote_code=True)
```

## Quick Start

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("thewh1teagle/bert-char-he")
model = AutoModelForMaskedLM.from_pretrained("thewh1teagle/bert-char-he", trust_remote_code=True)
model.eval()

sentence = "של[MASK]ם עול[MASK]"  # שלום עולם with two masked characters

inputs = tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted = logits.argmax(-1)
input_ids = inputs["input_ids"].clone()
input_ids[input_ids == tokenizer.mask_token_id] = predicted[input_ids == tokenizer.mask_token_id]
print(tokenizer.decode(input_ids[0], skip_special_tokens=True))  # שלום עולם
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/TRAINING.md](docs/TRAINING.md) for details.
