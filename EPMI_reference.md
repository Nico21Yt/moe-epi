# EPMI: When to Create New Experts in MoE Post-Training

A reference note on the core idea of the proposal *"When to Create: Expert Pool
Misalignment as a Pre-Adaptation Diagnostic for MoE Post-Training"* and on the
diagnostic metrics it introduces.

---

## 1. Core Idea

### The gap

Mixture-of-Experts (MoE) post-training methods split into two camps, and neither
answers a prior question.

- **Fixed-expert methods** (ESFT, PERFT, LoRAMoE, and the more recent
  routing-informed variants MoE-Sieve, DR-LoRA, LoRA-SMoE) assume the existing
  expert pool is the right starting point. They only ask *which* experts to
  update, or *how much* capacity to give each existing expert.
- **Expert-creation methods** (MoExtend, UpIT) do add new experts, but they do so
  **unconditionally** — because the setting calls for it (a new modality, a
  dense-to-MoE conversion), not because the current pool was diagnosed as
  inadequate.

The unanswered question sits before the choice of adaptation strategy:

> Given an already-trained MoE model and a small sample of target-domain data,
> can the current expert pool absorb the new supervision signal at all — or does
> the pool itself need to grow?

### The proposal

Introduce the **Expert Pool Misalignment Index (EPMI)**, a *pre-adaptation*
diagnostic that estimates how poorly a target domain is served by an
already-trained MoE model.

Key properties:

- Computed from a **single forward pass** of a small calibration set
  (`D_cal`, default size 512) through the **frozen** model — no gradient update.
- Built from **three interpretable sub-metrics**, each targeting a distinct axis
  of misalignment.
- Sub-metrics are normalized to a common scale, so EPMI carries an interpretable
  unit even before any experiment is run.

EPMI then guides **Expert Creation Intervention (ECI)**, which inserts a small
number of new experts into the layers most affected by misalignment.

### The central empirical question

Does EPMI have **predictive validity** for the utility of expert creation?
That is, is the performance gain of ECI over budget-matched fine-tuning
systematically larger when EPMI is higher?

The proposal deliberately does **not** assume a functional form in advance:

- If a **threshold-like regime** emerges (gain ≈ 0 below some EPMI, positive
  above), characterize it via segmented regression and report the break point.
- If the relationship is **smoothly monotonic**, report that and reframe the
  recommendation as a continuous scaling rule (create more experts at higher
  EPMI; create none at very low EPMI).

The hypothesis is falsified **only** if EPMI has no systematic predictive
relationship with the performance gain.

### One-line reframing

> MoE post-training shifts from *"which experts to update"* to
> *"when the expert pool itself requires growth."*

---

## 2. Setup and Notation

| Symbol | Meaning |
|---|---|
| `M` | Pre-trained MoE language model |
| `L` | Number of MoE layers |
| `E` | Number of experts per layer (e.g. 64 in OLMoE) |
| `k` | Experts activated per token (top-k; e.g. 8 in OLMoE) |
| `D_target` | Target-domain dataset for post-training |
| `D_cal` | Small calibration subset of `D_target` (default size 512) |
| `H^pre_l` | Routing entropy at layer `l` on a pretraining reference set |
| `H^target_l` | Routing entropy at layer `l` on `D_cal` |
| `P^pre_l` | Empirical expert-selection distribution at layer `l` on the pretraining reference set |
| `P^target_l` | Empirical expert-selection distribution at layer `l` on `D_cal` |

The pretraining reference quantities (`H^pre`, `P^pre`) are estimated from
samples of the pretraining data mix (1,024 samples by default; publicly
available for OLMoE). When pretraining data is unavailable, a general-purpose
proxy corpus is used and sensitivity to the proxy is reported.

---

## 3. The Three Sub-Metrics

Each sub-metric is computed **per layer**, then **aggregated across layers** into
a single scalar. The three scalars are finally combined into the composite EPMI.

The three axes are orthogonal:

| Sub-metric | Axis of misalignment | Question it answers |
|---|---|---|
| **RED** | Routing concentration | Are target tokens collapsing onto a few experts? |
| **EAD** | Routing structure | Are target tokens activating a *different set* of experts? |
| **RLC** | Difficulty × overload | Are the overloaded experts also the failing ones? |

### 3.1 RED — Routing Entropy Drop (capacity collapse)

**Intuition.** Measures whether the target domain forces the router to concentrate
tokens onto a small number of experts, relative to how diffuse routing was under
pretraining. A relative *drop* in entropy signals capacity collapse.

**Per-layer definition.**

```
RED_l = 1 - H^target_l / H^pre_l
```

- `RED_l ≈ 0` when routing stays diffuse under the target domain (no collapse).
- `RED_l → 1` when tokens collapse onto a few experts (severe collapse).

Here `H_l` is the entropy of the routing distribution at layer `l`:

```
H_l = - Σ_{e=1..E} p_l(e) · log p_l(e)
```

where `p_l(e)` is the (average) routing probability mass on expert `e` at
layer `l`.

**Why relative, not absolute.** Different models and layers have different
baseline entropies; only the drop *relative to the model's own pretraining state*
is a comparable misalignment signal. This is why RED depends on a pretraining
reference distribution.

**Aggregation.** Mean of `RED_l` over the **top-K layers** with the highest
values, with `K = L / 3` by default:

```
EPMI_RED = mean( top-K of { RED_l } )
```

### 3.2 EAD — Expert Activation Divergence (structural mismatch)

**Intuition.** RED only measures *how concentrated* routing is, not *where* the
mass goes. Two routing distributions can have identical entropy while placing
their mass on entirely different experts. EAD captures this structural component.

**Per-layer definition.** Jensen–Shannon divergence between the target and
pretraining routing distributions:

```
EAD_l = JSD( P^target_l || P^pre_l )
```

where `JSD` is the Jensen–Shannon divergence:

```
JSD(P || Q) = 1/2 · KL(P || A) + 1/2 · KL(Q || A),   with A = (P + Q) / 2
KL(P || A)  = Σ_e P(e) · log( P(e) / A(e) )
```

`P^target_l` and `P^pre_l` are the empirical expert-selection distributions over
`D_cal` and the pretraining reference set, respectively. JSD is symmetric,
bounded, and well-defined even when supports differ — appropriate for comparing
routing distributions.

**Aggregation.** Depth-weighted mean, weighting later layers more heavily, since
later layers in transformer-based MoEs encode more task-specific representations:

```
EPMI_EAD = Σ_l w_l · EAD_l,   with w_l ∝ l / L
```

(The weights `w_l` are normalized to sum to 1.)

### 3.3 RLC — Residual Loss Concentration (difficulty–overload co-localization)

**Intuition.** RED and EAD are purely routing-level: they describe *how the router
behaves*, not *whether the chosen experts succeed*. RLC connects routing to task
difficulty by asking whether the experts receiving the most tokens are also the
ones producing the highest losses.

**Per-layer definition.** Correlation, across the `E` experts of layer `l`,
between expert load and the mean loss of tokens routed to each expert:

```
RLC_l = Corr( expert_load_l , mean_token_loss_l )
```

- `expert_load_l` — vector (length `E`) of load fractions across experts.
- `mean_token_loss_l` — vector (length `E`) of the average loss of tokens routed
  to each expert.

Both vectors are computed on `D_cal` with the **frozen** model.

**What RLC does and does not claim (important).** A high RLC alone does **not**
prove a new expert is needed — high-loss tokens concentrating on certain experts
could also reflect a domain that is simply hard, or experts that are trainable but
merely under-trained. RLC is a **triangulation** signal, not a standalone verdict:

- **High RED + High EAD + High RLC** → routing is concentrated, structurally
  different from pretraining, *and* the overloaded experts are the failing ones
  → strong signal of a structural bottleneck (consider creating experts).
- **High RED + High EAD + Low RLC** → the pool is concentrated but not yet
  failing → fine-tuning existing experts may suffice.

**Stability caveat.** A correlation measured on 512 examples can be noisy. RLC
stability across calibration draws is reported as part of diagnostic validation;
if insufficiently stable, the calibration size is increased to 1,024 or RLC is
reported with a confidence interval rather than as a point estimate.

**Aggregation.** Combined into the scalar `EPMI_RLC` (aggregated across layers,
analogous to the other sub-metrics).

---

## 4. Composite EPMI

The three aggregated sub-metrics are combined as an **equal-weighted sum**:

```
EPMI = ( EPMI_RED + EPMI_EAD + EPMI_RLC ) / 3
```

- **Equal weighting is the default** because there is no prior evidence that one
  sub-metric should dominate. A sub-metric ablation tests whether reweighting or
  dropping a component changes predictive validity; if it does, that is treated
  as an empirical finding about which aspect of misalignment matters most.

- **On the threshold.** For practitioners wanting a default operating point, a
  composite EPMI above roughly **0.4** is suggested as a *tentative* cutoff,
  motivated by the interpretable scales of the sub-metrics — **not** a
  theoretically derived constant, and explicitly not tuned on the evaluation data.
  The more important claim is about the **continuous relationship** between EPMI
  and the performance gain (a scatter plot), not a fixed trigger rule.

### Computation chain (summary)

```
For each MoE layer l:
    RED_l  = 1 - H^target_l / H^pre_l                  # needs full routing distribution + pretrain ref
    EAD_l  = JSD( P^target_l || P^pre_l )              # needs full routing distribution + pretrain ref
    RLC_l  = Corr( expert_load_l , mean_token_loss_l ) # needs top-k load + per-token loss

Aggregate across layers:
    EPMI_RED = mean( top-(L/3) of { RED_l } )
    EPMI_EAD = Σ_l w_l · EAD_l,   w_l ∝ l / L
    EPMI_RLC = aggregate({ RLC_l })

Combine:
    EPMI = ( EPMI_RED + EPMI_EAD + EPMI_RLC ) / 3
```

**What each metric consumes (practical note for implementation):**

| Metric | Needs full routing distribution (softmax over E) | Needs top-k selection (load) | Needs per-token loss | Needs pretraining reference |
|---|:---:|:---:|:---:|:---:|
| RED | ✓ | | | ✓ |
| EAD | ✓ | | | ✓ |
| RLC | | ✓ | ✓ | |

RED and EAD can be computed from routing logits alone (cheapest). RLC
additionally requires per-token losses.

---

## 5. From Diagnosis to Action: ECI

When EPMI is high enough to recommend structural expansion, **Expert Creation
Intervention (ECI)** inserts new experts into the layers with the highest EAD
scores. The number inserted scales with the routing entropy drop signal, capped
at **20% of the original expert count per layer**.

**Three initialization strategies**, each paired to a dominant sub-metric:

| Strategy | What it does | Driven by |
|---|---|---|
| **Cold** | New expert from random weights | Ablation reference — isolates whether structural expansion *alone* helps |
| **Cloned** | Copy the highest-load expert, then add weight-norm-scaled Gaussian noise to diverge | RED + EAD dominant |
| **Residual** | Pre-train the new expert to model the gap (residual) between target supervision and current predictions on `D_cal` | RLC dominant |

**Expert onboarding (critical).** A newly inserted expert is useless if the router
never selects it. A two-phase procedure is used:

1. **Router warm-up** (~100 gradient steps): freeze the original model; train only
   the router extension and new-expert parameters on `D_cal`, with a soft routing
   bias encouraging selection of the new expert for high-loss tokens.
2. **Full post-training**: update all trainable parameters jointly. Original
   experts and router are fine-tuned at **10%** of the new-expert learning rate,
   so new experts specialize while original capabilities are preserved.

---

## 6. Why Three Metrics (Recap)

Each sub-metric has a blind spot the others cover:

- **RED** sees concentration but not *where* the mass goes.
- **EAD** sees structural relocation but not whether the chosen experts *succeed*.
- **RLC** connects to failure, but alone cannot separate "hard domain" from
  "under-trained expert" — it only becomes decisive *in conjunction* with RED and
  EAD.

Together they cover three orthogonal axes — concentration, structure, and
difficulty–overload co-localization — and the proposal's bet is that the
composite diagnoses "can the existing pool absorb this domain?" more reliably than
any single signal.
