from typing import Optional, Tuple

import torch

from routing_extractor import RoutingStats


_LN2 = torch.tensor(2.0).log().item()   # ≈ 0.6931 — JSD upper bound (natural log)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Jensen–Shannon divergence with natural logarithm.
    JSD(P ∥ Q) = ½ KL(P ∥ A) + ½ KL(Q ∥ A),  A = (P+Q)/2
    Bounded in [0, ln 2].  Well-defined when either distribution has zero mass.
    p, q: 1-D probability vectors of equal length.
    """
    a = (p + q) * 0.5
    kl_p = (p * (torch.log(p + eps) - torch.log(a + eps))).sum()
    kl_q = (q * (torch.log(q + eps) - torch.log(a + eps))).sum()
    return 0.5 * (kl_p + kl_q)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_red(
    target: RoutingStats,
    pretrain: RoutingStats,
    top_k_fraction: float = 1 / 3,
) -> Tuple[float, torch.Tensor]:
    """
    Routing Entropy Drop.

    RED_l  = 1 − H^target_l / H^pretrain_l,  clamped to [0, 1]
    EPMI_RED = mean of the top-(L/3) RED_l values

    Returns (EPMI_RED scalar, RED_l tensor [n_layers]).
    """
    eps = 1e-10
    red_l = 1.0 - target.entropy / (pretrain.entropy + eps)
    red_l = red_l.clamp(min=0.0)      # negative = target more diffuse → no collapse

    n_layers = red_l.shape[0]
    k = max(1, int(n_layers * top_k_fraction))
    top_vals, _ = torch.topk(red_l, k)
    epmi_red = top_vals.mean().item()

    return epmi_red, red_l


def compute_ead(
    target: RoutingStats,
    pretrain: RoutingStats,
) -> Tuple[float, torch.Tensor]:
    """
    Expert Activation Divergence.

    EAD_l  = JSD(P^target_l ∥ P^pretrain_l) / ln(2)   → normalized to [0, 1]
    EPMI_EAD = depth-weighted mean (w_l ∝ l / L, 1-indexed, normalized to sum 1)

    Returns (EPMI_EAD scalar, EAD_l tensor [n_layers]).
    """
    n_layers = target.avg_routing_probs.shape[0]

    ead_l = torch.stack([
        _jsd(target.avg_routing_probs[l], pretrain.avg_routing_probs[l])
        for l in range(n_layers)
    ]) / _LN2   # normalize to [0, 1]

    # Depth weights: w_l ∝ l/L (layer index 1-based), normalized to sum to 1.
    weights = torch.arange(1, n_layers + 1, dtype=torch.float32) / n_layers
    weights = weights / weights.sum()

    epmi_ead = (weights * ead_l).sum().item()
    return epmi_ead, ead_l


def compute_rlc(
    target: RoutingStats,
    per_token_losses: torch.Tensor,      # [n_tokens]          — one CE loss per token
    per_token_expert_ids: torch.Tensor,  # [n_tokens, n_layers, top_k] — top-k assignments
) -> Tuple[float, torch.Tensor]:
    """
    Residual Loss Concentration.  TODO: implement.

    RLC_l = Pearson correlation between expert_load_l and mean_token_loss_l,
            where mean_token_loss_l[e] = mean CE loss of tokens routed to expert e.

    Requires (a) per-token CE losses and (b) per-layer top-k expert assignments,
    both from a forward pass with labels.  The routing extractor must be extended
    to return per_token_expert_ids before this function can be called.
    """
    raise NotImplementedError(
        "RLC is not yet implemented.  "
        "Implement per-token loss collection in routing_extractor first."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Composite
# ──────────────────────────────────────────────────────────────────────────────

def compute_epmi(
    epmi_red: float,
    epmi_ead: float,
    epmi_rlc: Optional[float] = None,
) -> float:
    """Composite EPMI = equal-weighted mean of available sub-metrics."""
    components = [epmi_red, epmi_ead]
    if epmi_rlc is not None:
        components.append(epmi_rlc)
    return sum(components) / len(components)
