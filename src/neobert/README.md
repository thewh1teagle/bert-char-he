# NeoBERT

Source: https://huggingface.co/chandar-lab/NeoBERT

Fetched and modified for this project:

- Reduced from 28 to 6 layers and hidden size from 768 to 512 to get ~20M parameters (vs 250M original). Smaller model means smaller ONNX export size.
- Added `NEOBERT_ONNX_EXPORT=1` env var support for ONNX-compatible ops (real-valued RoPE, pure PyTorch SwiGLU).
- Added MLM loss computation to `NeoBERTLMHead.forward()` — original had no `labels` support.
