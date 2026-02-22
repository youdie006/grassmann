from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class WikiText2Dataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        sequence_len: int = 128,
        tokenizer_name: str = "bert-base-uncased",
        source: str = "local_parquet",
        data_root: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.block_size = sequence_len
        self.source = source

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        texts = self._load_split_texts(
            split=split,
            source=source,
            data_root=data_root,
            max_samples=max_samples,
        )
        all_text = "\n\n".join(texts)

        tokenized = self.tokenizer(
            all_text,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
            truncation=False,
        )
        self.token_ids = tokenized["input_ids"]

        self.num_blocks = len(self.token_ids) // (sequence_len + 1)

        self.token_ids = self.token_ids[: self.num_blocks * (sequence_len + 1)]

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start_idx = idx * (self.block_size + 1)
        end_idx = start_idx + self.block_size + 1

        tokens = torch.tensor(self.token_ids[start_idx:end_idx], dtype=torch.long)

        inputs = tokens[:-1]
        targets = tokens[1:]

        return inputs, targets

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @staticmethod
    def _load_split_texts(
        split: str,
        source: str,
        data_root: Optional[str],
        max_samples: Optional[int],
    ):
        split = split.lower()
        if source == "hf":
            # Lazy import so local-parquet mode does not require datasets.
            from datasets import load_dataset

            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            texts = [example["text"] for example in dataset]
        elif source == "local_parquet":
            if data_root is None:
                raise ValueError("data_root is required when source='local_parquet'")
            # Lazy import so users can still run hf mode without pyarrow.
            import pyarrow.parquet as pq

            file_map = {
                "train": "train-00000-of-00001.parquet",
                "validation": "validation-00000-of-00001.parquet",
                "test": "test-00000-of-00001.parquet",
            }
            if split not in file_map:
                raise ValueError(f"Unsupported split: {split}")
            parquet_path = Path(data_root) / file_map[split]
            if not parquet_path.exists():
                raise FileNotFoundError(f"Missing parquet file: {parquet_path}")
            table = pq.read_table(parquet_path, columns=["text"])
            texts = table["text"].to_pylist()
        else:
            raise ValueError(f"Unknown source: {source}")

        if max_samples is not None:
            texts = texts[:max_samples]
        return texts
