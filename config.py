from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch


@dataclass
class EPMIConfig:
    # ── Model ──────────────────────────────────────────────────────────────────
    model_name: str = "allenai/OLMoE-1B-7B-0924"
    device: str = "auto"       # "auto" | "cuda" | "cpu"
    dtype: str = "bfloat16"    # "bfloat16" | "float16" | "float32"
    batch_size: int = 8
    max_length: int = 512

    # ── Model architecture (verified against model.config at load time) ────────
    n_moe_layers: int = 16
    n_experts: int = 64
    top_k: int = 8

    # ── Pretraining reference (C4) ─────────────────────────────────────────────
    pretrain_dataset: str = "allenai/c4"
    pretrain_subset: str = "en"
    pretrain_split: str = "train"
    pretrain_n_samples: int = 256   # bump to 1024 once the pipeline is stable

    # ── Target domains ─────────────────────────────────────────────────────────
    # {name: (dataset_name, subset_or_None, split, text_field)}
    domains: Dict[str, Tuple] = field(default_factory=lambda: {
        "gsm8k": ("openai/gsm8k", "main", "train", "question"),
        # "medmcqa": ("medmcqa", None, "train", "question"),  # add back later
    })
    n_samples_per_domain: int = 512

    # ── Metric options ─────────────────────────────────────────────────────────
    question_only: bool = True          # feed question text only (no answer/options)
    top_k_layers_fraction: float = 1/3  # RED aggregates top-(L * frac) layers

    # ── Stability check ────────────────────────────────────────────────────────
    n_stability_draws: int = 3
    stability_seed_base: int = 42       # draw i uses seed = base + (i+1)*100

    def get_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }[self.dtype]

    def get_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
