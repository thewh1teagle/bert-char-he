"""Masked-token inference for debugging the MLM model.

Example:
    uv run src/infer.py --checkpoint outputs/bert-char-he/checkpoint-9000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from model import build_model
from tokenization import build_tokenizer

EXAMPLE_TEXT = "של[MASK]ם עול[MASK]"


def load_model(checkpoint: str):
    from safetensors.torch import load_file
    model = build_model()
    state = load_file(str(Path(checkpoint) / "model.safetensors"), device="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def run(model, tokenizer: PreTrainedTokenizerFast, text: str, device: torch.device) -> None:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    preds = out.logits.argmax(-1)
    mask_id = tokenizer.mask_token_id
    result_ids = input_ids.clone()
    result_ids[input_ids == mask_id] = preds[input_ids == mask_id]

    from decode import ids_to_str
    print(f"input : {text}")
    print(f"output: {ids_to_str(result_ids[0].tolist(), tokenizer)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=build_tokenizer(),
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )

    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device)
    run(model, tokenizer, EXAMPLE_TEXT, device)


if __name__ == "__main__":
    main()
