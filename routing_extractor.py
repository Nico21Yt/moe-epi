from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import EPMIConfig


@dataclass
class RoutingStats:
    """
    Per-layer routing statistics accumulated over a text corpus.
    All tensors are on CPU, dtype float32.
    """
    # Marginal routing distribution P_l: average softmax probability per expert
    # across all (non-padding) tokens.  Shape: [n_moe_layers, n_experts]
    avg_routing_probs: torch.Tensor

    # Routing entropy H_l = -sum_e P_l(e) log P_l(e).  Shape: [n_moe_layers]
    entropy: torch.Tensor

    # Expert load from top-k selection: fraction of tokens whose top-k includes
    # expert e.  Shape: [n_moe_layers, n_experts]
    expert_load: torch.Tensor

    # Total non-padding tokens processed
    n_tokens: int


def extract_routing_stats(
    texts: List[str],
    model,
    tokenizer,
    config: EPMIConfig,
) -> RoutingStats:
    """
    Run a single frozen forward pass over `texts` and collect per-layer routing
    statistics.  Padding tokens are excluded via attention_mask before any
    accumulation.

    Returns RoutingStats with:
      avg_routing_probs — feeds RED (entropy) and EAD (JSD)
      expert_load       — feeds RLC (TODO)
    """
    device = config.get_device()
    n_layers = config.n_moe_layers
    n_experts = config.n_experts

    # Float64 accumulators avoid precision loss when summing over many tokens.
    sum_probs = torch.zeros(n_layers, n_experts, dtype=torch.float64)
    sum_load  = torch.zeros(n_layers, n_experts, dtype=torch.float64)
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), config.batch_size), desc="routing fwd"):
            batch = texts[start : start + config.batch_size]

            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)   # [B, T]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_router_logits=True,
            )

            router_logits_tuple = outputs.router_logits
            if router_logits_tuple is None:
                raise RuntimeError(
                    "Model returned router_logits=None.  "
                    "Verify that output_router_logits=True is honoured by this model."
                )

            # Drop None entries (non-MoE layers in hybrid models; not expected
            # for OLMoE but guarded anyway).
            router_logits_list = [rl for rl in router_logits_tuple if rl is not None]
            if len(router_logits_list) != n_layers:
                raise RuntimeError(
                    f"Expected {n_layers} non-None router_logit tensors, "
                    f"got {len(router_logits_list)}.  "
                    f"Update config.n_moe_layers to match the model."
                )

            # Flatten attention mask: [B*T]
            # Row-major flattening matches how the model processes tokens
            # (batch-first, left-to-right), so position i*T+j in the flat
            # mask corresponds to row i*T+j in each router_logits tensor.
            flat_mask = attn_mask.contiguous().view(-1).bool()   # [B*T]
            total_tokens += int(flat_mask.sum().item())

            for l, logits_l in enumerate(router_logits_list):
                # logits_l: [B*T, n_experts], may be bfloat16
                logits_f = logits_l.float()          # promote for numerical stability
                valid    = logits_f[flat_mask]       # [n_valid, n_experts]

                # ── Softmax → marginal routing probability ─────────────────────
                probs = F.softmax(valid, dim=-1)     # [n_valid, n_experts]
                sum_probs[l].add_(probs.sum(dim=0).double().cpu())

                # ── Top-k selection → binary expert load ───────────────────────
                _, top_idx = torch.topk(valid, k=config.top_k, dim=-1)   # [n_valid, k]
                load_onehot = torch.zeros(
                    valid.shape[0], n_experts,
                    dtype=torch.float32, device=valid.device,
                )
                load_onehot.scatter_(1, top_idx, 1.0)
                sum_load[l].add_(load_onehot.sum(dim=0).double().cpu())

            del outputs, input_ids, attn_mask, enc

    if total_tokens == 0:
        raise ValueError(
            "No non-padding tokens accumulated.  "
            "Check that the tokenizer produces non-empty sequences."
        )

    avg_routing_probs = (sum_probs / total_tokens).float()   # [L, E]
    expert_load       = (sum_load  / total_tokens).float()   # [L, E]

    eps = 1e-10
    entropy = -(avg_routing_probs * (avg_routing_probs + eps).log()).sum(dim=-1)  # [L]

    return RoutingStats(
        avg_routing_probs=avg_routing_probs,
        entropy=entropy,
        expert_load=expert_load,
        n_tokens=total_tokens,
    )
