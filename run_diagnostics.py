"""
EPMI diagnostics for OLMoE.

Usage:
    python run_diagnostics.py

Results are saved to  results/run_<YYYYMMDD_HHMMSS>/  containing:
    config.json          — all config parameters
    pretrain_stats.pt    — RoutingStats tensor bundle for C4 reference
    <domain>_stats.pt    — RoutingStats tensor bundle per domain
    metrics.json         — scalar EPMI values, per-layer RED/EAD vectors, stability
    summary.txt          — the same table that was printed to stdout

Edit EPMIConfig in config.py to change model, datasets, sample counts, etc.
Set pretrain_n_samples=256 for a quick smoke test; bump to 1024 for the final run.
"""

import dataclasses
import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import EPMIConfig
from data_loader import load_texts
from metrics import compute_ead, compute_epmi, compute_red
from routing_extractor import RoutingStats, extract_routing_stats


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(config: EPMIConfig):
    device = config.get_device()
    dtype  = config.get_dtype()

    print(f"\nLoading {config.model_name}  device={device}  dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=dtype,
        device_map="auto" if device == "cuda" else device,
        trust_remote_code=True,
    )
    model.eval()

    mcfg     = model.config
    n_layers = mcfg.num_hidden_layers
    n_exp    = mcfg.num_experts
    top_k    = mcfg.num_experts_per_tok
    print(f"num layers = {n_layers}, experts per layer = {n_exp}, top_k = {top_k}")

    if (n_layers, n_exp, top_k) != (config.n_moe_layers, config.n_experts, config.top_k):
        print(
            f"[warn] Architecture mismatch — config had "
            f"({config.n_moe_layers}, {config.n_experts}, {config.top_k}), "
            f"updating to ({n_layers}, {n_exp}, {top_k})."
        )
        config.n_moe_layers = n_layers
        config.n_experts    = n_exp
        config.top_k        = top_k

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Per-domain computation
# ──────────────────────────────────────────────────────────────────────────────

def run_domain(
    texts: List[str],
    pretrain_stats: RoutingStats,
    model,
    tokenizer,
    config: EPMIConfig,
) -> tuple:
    """Returns (RoutingStats, metrics_dict)."""
    stats            = extract_routing_stats(texts, model, tokenizer, config)
    epmi_red, red_l  = compute_red(stats, pretrain_stats, config.top_k_layers_fraction)
    epmi_ead, ead_l  = compute_ead(stats, pretrain_stats)
    epmi             = compute_epmi(epmi_red, epmi_ead)
    metrics = {
        "n_tokens": stats.n_tokens,
        "EPMI_RED": epmi_red,
        "EPMI_EAD": epmi_ead,
        "EPMI":     epmi,
        "red_l":    red_l,    # tensor [L]
        "ead_l":    ead_l,    # tensor [L]
    }
    return stats, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Saving
# ──────────────────────────────────────────────────────────────────────────────

def _routing_stats_to_dict(rs: RoutingStats) -> dict:
    return {
        "avg_routing_probs": rs.avg_routing_probs.tolist(),   # [L, E]
        "entropy":           rs.entropy.tolist(),              # [L]
        "expert_load":       rs.expert_load.tolist(),          # [L, E]
        "n_tokens":          rs.n_tokens,
    }


def save_results(
    run_dir: Path,
    config: EPMIConfig,
    pretrain_stats: RoutingStats,
    domain_stats: Dict[str, RoutingStats],    # name → RoutingStats
    domain_metrics: Dict[str, Dict],          # name → metrics dict
    stability_records: Dict[str, Dict],       # name → stability info
    summary_text: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── config ────────────────────────────────────────────────────────────────
    cfg_dict = dataclasses.asdict(config)
    cfg_dict.pop("domains", None)   # not JSON-serializable directly
    cfg_dict["domains"] = {
        k: list(v) for k, v in config.domains.items()
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    # ── RoutingStats tensors (.pt) ────────────────────────────────────────────
    torch.save(
        {
            "avg_routing_probs": pretrain_stats.avg_routing_probs,
            "entropy":           pretrain_stats.entropy,
            "expert_load":       pretrain_stats.expert_load,
            "n_tokens":          pretrain_stats.n_tokens,
        },
        run_dir / "pretrain_stats.pt",
    )
    for name, rs in domain_stats.items():
        torch.save(
            {
                "avg_routing_probs": rs.avg_routing_probs,
                "entropy":           rs.entropy,
                "expert_load":       rs.expert_load,
                "n_tokens":          rs.n_tokens,
            },
            run_dir / f"{name}_stats.pt",
        )

    # ── metrics JSON ──────────────────────────────────────────────────────────
    metrics_out = {
        "pretrain": _routing_stats_to_dict(pretrain_stats),
        "domains":  {},
        "stability": stability_records,
    }
    for name, m in domain_metrics.items():
        metrics_out["domains"][name] = {
            "n_tokens": m["n_tokens"],
            "EPMI_RED": m["EPMI_RED"],
            "EPMI_EAD": m["EPMI_EAD"],
            "EPMI":     m["EPMI"],
            "red_per_layer": m["red_l"].tolist(),
            "ead_per_layer": m["ead_l"].tolist(),
        }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ── summary text ──────────────────────────────────────────────────────────
    with open(run_dir / "summary.txt", "w") as f:
        f.write(summary_text)

    print(f"\nResults saved to  {run_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cv(vals: List[float]) -> float:
    m = statistics.mean(vals)
    return (statistics.stdev(vals) / m) if m != 0 else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    config   = EPMIConfig()
    run_dir  = Path("results") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    lines: List[str] = []   # collected for summary.txt

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    # ── 1. Load model ──────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(config)
    log(f"Model:  {config.model_name}")
    log(f"Layers: {config.n_moe_layers}  Experts/layer: {config.n_experts}  top_k: {config.top_k}")

    # ── 2. Pretraining reference ───────────────────────────────────────────────
    log(f"\nBuilding pretraining reference ({config.pretrain_n_samples} samples from C4)...")
    pretrain_texts = load_texts(
        config.pretrain_dataset, "text", config.pretrain_n_samples,
        subset=config.pretrain_subset, split=config.pretrain_split,
        seed=0, streaming=True,
    )
    pretrain_stats = extract_routing_stats(pretrain_texts, model, tokenizer, config)
    log(
        f"  {pretrain_stats.n_tokens:,} tokens | "
        f"mean H = {pretrain_stats.entropy.mean():.4f} | "
        f"range [{pretrain_stats.entropy.min():.4f}, {pretrain_stats.entropy.max():.4f}]"
    )

    # ── 3. Diagnostic table ────────────────────────────────────────────────────
    log()
    log("=" * 70)
    log(f"{'Domain':<12}  {'EPMI_RED':>10}  {'EPMI_EAD':>10}  {'EPMI':>10}  {'Tokens':>8}")
    log("=" * 70)

    domain_stats:   Dict[str, RoutingStats] = {}
    domain_metrics: Dict[str, Dict]         = {}

    for name, (ds_name, subset, split, text_field) in config.domains.items():
        texts = load_texts(
            ds_name, text_field, config.n_samples_per_domain,
            subset=subset, split=split,
            seed=config.stability_seed_base, streaming=False,
        )
        rs, m = run_domain(texts, pretrain_stats, model, tokenizer, config)
        domain_stats[name]   = rs
        domain_metrics[name] = m
        log(
            f"{name:<12}  {m['EPMI_RED']:>10.4f}  {m['EPMI_EAD']:>10.4f}"
            f"  {m['EPMI']:>10.4f}  {m['n_tokens']:>8,}"
        )
    log("=" * 70)

    # ── 4. Per-layer detail ────────────────────────────────────────────────────
    log()
    log("Per-layer RED and EAD  (layer 0 = shallowest, layer L-1 = deepest):")
    col_headers = "  ".join(f"{'RED_'+n:>9}  {'EAD_'+n:>9}" for n in domain_metrics)
    log(f"{'Layer':>5}  {col_headers}")
    for l in range(config.n_moe_layers):
        row = f"{l:>5}  "
        for m in domain_metrics.values():
            row += f"  {m['red_l'][l].item():>9.4f}  {m['ead_l'][l].item():>9.4f}"
        log(row)

    # ── 5. Stability check ─────────────────────────────────────────────────────
    n_draws   = config.n_stability_draws
    draw_size = config.n_samples_per_domain
    log()
    log(f"Stability ({n_draws} draws × {draw_size} samples; pretrain ref fixed):")

    stability_records: Dict[str, Dict] = {}
    for name, (ds_name, subset, split, text_field) in config.domains.items():
        red_vals: List[float] = []
        ead_vals: List[float] = []
        for draw in range(n_draws):
            seed = config.stability_seed_base + (draw + 1) * 100
            texts = load_texts(
                ds_name, text_field, draw_size,
                subset=subset, split=split,
                seed=seed, streaming=False,
            )
            _, m = run_domain(texts, pretrain_stats, model, tokenizer, config)
            red_vals.append(m["EPMI_RED"])
            ead_vals.append(m["EPMI_EAD"])

        cv_red = _cv(red_vals)
        cv_ead = _cv(ead_vals)
        stability_records[name] = {
            "EPMI_RED_draws": red_vals,
            "EPMI_RED_mean":  statistics.mean(red_vals),
            "EPMI_RED_cv":    cv_red,
            "EPMI_EAD_draws": ead_vals,
            "EPMI_EAD_mean":  statistics.mean(ead_vals),
            "EPMI_EAD_cv":    cv_ead,
        }
        log(f"\n  {name}:")
        log(
            f"    EPMI_RED  draws={[f'{v:.4f}' for v in red_vals]}"
            f"  mean={statistics.mean(red_vals):.4f}  CV={cv_red:.3f}"
            f"  {'OK' if cv_red < 0.10 else 'UNSTABLE'}"
        )
        log(
            f"    EPMI_EAD  draws={[f'{v:.4f}' for v in ead_vals]}"
            f"  mean={statistics.mean(ead_vals):.4f}  CV={cv_ead:.3f}"
            f"  {'OK' if cv_ead < 0.10 else 'UNSTABLE'}"
        )

    log()
    log("Done.")

    # ── 6. Save everything ─────────────────────────────────────────────────────
    save_results(
        run_dir        = run_dir,
        config         = config,
        pretrain_stats = pretrain_stats,
        domain_stats   = domain_stats,
        domain_metrics = domain_metrics,
        stability_records = stability_records,
        summary_text   = "\n".join(lines),
    )


if __name__ == "__main__":
    main()
