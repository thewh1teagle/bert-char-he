"""Dataset loading and collation for MLM pretraining."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast


def prepare_dataset(txt_path: str, tokenizer: PreTrainedTokenizerFast, cache_dir: str) -> Dataset:
    """Tokenize a plain-text file and cache the result to Arrow.

    Each non-empty line becomes one example. Re-uses the cache if it already exists.
    """
    cache_path = Path(cache_dir)
    if cache_path.exists():
        return load_from_disk(str(cache_path))

    with open(txt_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    raw = Dataset.from_dict({"text": lines})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding=False,
            return_special_tokens_mask=True,
        )

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"], desc="tokenizing")
    tokenized.save_to_disk(str(cache_path))
    return tokenized


def make_dataloaders(
    args,
    tokenizer: PreTrainedTokenizerFast,
) -> tuple[DataLoader, DataLoader]:
    train_ds = prepare_dataset(args.train_dataset, tokenizer, args.train_dataset + ".cache")
    eval_ds = prepare_dataset(args.eval_dataset, tokenizer, args.eval_dataset + ".cache")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    return train_loader, eval_loader
