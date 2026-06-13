import random
from typing import List, Optional

from datasets import load_dataset


def load_texts(
    dataset_name: str,
    text_field: str,
    n_samples: int,
    subset: Optional[str] = None,
    split: str = "train",
    seed: int = 42,
    min_length: int = 20,
    streaming: bool = False,
) -> List[str]:
    """
    Load up to n_samples text strings from a HuggingFace dataset.

    streaming=True  — use for large corpora (C4); samples are drawn from a
                      shuffle buffer without downloading the full dataset.
    streaming=False — use for small datasets (GSM8K, MedMCQA); loads the full
                      split and samples randomly with the given seed.
    """
    if streaming:
        ds = load_dataset(
            dataset_name, subset,
            split=split,
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        samples: List[str] = []
        for item in ds:
            text = item[text_field].strip()
            if len(text) >= min_length:
                samples.append(text)
            if len(samples) >= n_samples:
                break
    else:
        ds = load_dataset(
            dataset_name, subset,
            split=split,
        )
        indices = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        samples = []
        for i in indices:
            text = ds[i][text_field].strip()
            if len(text) >= min_length:
                samples.append(text)
            if len(samples) >= n_samples:
                break

    if len(samples) < n_samples:
        print(f"[warn] {dataset_name}: requested {n_samples} samples, got {len(samples)}")
    return samples
