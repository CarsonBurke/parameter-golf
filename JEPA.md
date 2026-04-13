# JEPA for LM Pretraining — Design Document

## The Thesis
CE trains the model to classify tokens. JEPA trains the model to understand dynamics. For a 16MB parameter budget, dynamics are cheaper to learn than classification.

CE: "Which of these 8192 BPE tokens comes next?"
JEPA: "What does the world look like after the next token?"

CE forces the model to internalize the tokenizer — an arbitrary, frequency-driven codebook. Every parameter spent memorizing that "▁the" is token 47 and "▁The" is token 2031 is a parameter not learning that they mean the same thing.

JEPA lets the model build its own representation of "what comes next." Token identity falls out as a byproduct of representation quality, not as a direct optimization target.

## Why Tied Embeddings Make This Work

With tied embeddings, the hidden state space and the token embedding space are **the same vector space**. `tok_emb.weight` defines both:
- Input: token → embedding vector (lookup)
- Output: hidden state → nearest embedding → token (decoding)

This means there's no gap between "predict the next latent" and "predict the next token." They're the same problem viewed from different angles:
- JEPA: predict `hidden[t+1]` — a vector in R^D
- CE: classify `hidden[t]` against all embeddings in `tok_emb.weight`

If the model learns to predict `hidden[t+1]` accurately, that prediction lives near the correct token's embedding. The embedding table is the codebook. No separate decoder needed.

## What the Model Actually Learns

In an autoregressive transformer:
```
hidden[t]   = f(tokens[0..t])      # representation of context up to t
hidden[t+1] = f(tokens[0..t+1])    # representation after seeing one more token
```

The difference between `hidden[t]` and `hidden[t+1]` encodes the *effect* of `token[t+1]` on the model's world state. Predicting `hidden[t+1]` from `hidden[t]` is learning: **"given everything I've seen, how will my understanding change when I see the next piece?"**

This is deeper than token classification. The model must learn:
- What kind of continuation is plausible (syntax, semantics)
- How its own internal representation responds to new information (dynamics)
- Which aspects of context matter for prediction (attention, abstraction)

Token identity is recoverable from this — it's the token whose embedding is closest to the predicted hidden state. But the model never optimizes for token identity directly. It optimizes for understanding.

## Architecture

```
                     input tokens
                          │
                    ┌─────▼─────┐
                    │  tok_emb  │  (shared embedding space — ties latent to token space)
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │ transformer│  (learns dynamics)
                    │  blocks    │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │ final_norm │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │ predictor  │  (predicts next hidden state)
                    └─────┬─────┘
                          │
                     JEPA loss: ||ẑ[t+1] - z[t+1]||

    eval only: hidden → dot(tok_emb.weight) → logits → BPB
```

No CE head during training. No classification objective. The trunk learns representation dynamics. The predictor learns to anticipate the next state. Token decoding happens only at eval time through the shared embedding table.

### Projector Stance (Updated)

The tied embedding argument still holds for **eval**: BPB is measured from the native hidden state through `tok_emb.weight`, so the trunk must ultimately learn a token-aligned space.

But the LeWorldModel result changes the training recommendation. If the trunk output is normalized (`LayerNorm`/`RMSNorm`), applying anti-collapse regularization directly in that space may be the wrong geometry. A small **training-only projector** gives us an unnormalized space where JEPA and anti-collapse regularization can act cleanly, while leaving eval unchanged.

So the updated stance is:
- **No permanent bottleneck in the artifact**
- **Allow a small training-only projector** after `final_norm`
- **Eval still reads native hidden states** through tied embeddings

This is not a semantic decoder. It is a scaffold to make the JEPA objective trainable.

### Predictor Design

The predictor maps `hidden[t]` → `predicted_hidden[t+1]`. It should be:
- **Lightweight:** a few percent of the trunk's parameters
- **Expressive enough** to model the token-conditional state transition
- **Not so powerful** that the trunk can delegate all learning to it

```python
class JEPAPredictor(nn.Module):
    def __init__(self, d_model, expansion=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion, bias=False),
            nn.ReLU(),
            nn.Linear(d_model * expansion, d_model, bias=False),
        )
        # Zero-init output: start as "predict same state" (identity-like)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, hidden):
        return self.net(hidden)
```

**Zero-init output:** At initialization, the predictor outputs zeros. Combined with a residual connection or the trunk's own hidden states, this means the initial prediction is "next state ≈ current state." Training then learns the *delta* — what changes. This is stabler than predicting from scratch.

**Parameter cost:** 512 × 1024 + 1024 × 512 = 1.05M params. ~6% of model. Under int8+zlib: ~0.5MB.

If too expensive, shrink expansion to 1: 512 × 512 × 2 = 524K params. ~3% of model.

**Updated recommendation:** treat predictor size as an ablation, not a free lunch. The LeWorldModel paper found that larger predictors were not monotonically better. The fact that the predictor is free at artifact time does **not** mean it is free in optimization. Start medium. Only scale it up if BPB at 2000 steps improves.

## Target: What to Predict

### Option 1: Direct Self-Target + SIGReg (new default)
```python
h = projector(hidden)                  # optional but recommended if final hidden is normalized
h_target = h[:, 1:]
h_pred = predictor(h[:, :-1])
loss = F.mse_loss(h_pred, h_target) + lambda_sig * sigreg(h)
```

This is the cleanest transfer from LeWorldModel: no stop-gradient, no EMA, just next-state prediction plus a principled anti-collapse regularizer on the latent distribution.

**Why this is better:** stop-gradient and EMA are heuristics. SIGReg is the actual collapse prevention mechanism. If it works, this is simpler and easier to reason about.

### Option 2: Self-Target (stop-gradient fallback)
```python
h_target = h[:, 1:].detach()
h_pred = predictor(h[:, :-1])
loss = F.smooth_l1_loss(h_pred, h_target) + lambda_sig * sigreg(h)
```

Fallback if the fully coupled objective is unstable in the LM setting. This should not be the default starting point anymore.

### Option 3: EMA Target (last resort)
```python
# EMA model provides stable targets
with torch.no_grad():
    hidden_ema = ema_model(tokens)
    h_ema = projector(hidden_ema)         # or share projector weights if practical
h = projector(hidden)
h_target = h_ema[:, 1:]
h_pred = predictor(h[:, :-1])
loss = F.smooth_l1_loss(h_pred, h_target) + lambda_sig * sigreg(h)
```

The EMA model is a slowly-evolving copy of the trunk (exponential moving average, decay ~0.996). It provides **stable targets** that don't shift rapidly. This is what I-JEPA, BYOL, and DINO use.

**Advantage:** The SOTA already uses EMA (decay=0.9965). We piggyback on existing infrastructure.

**Cost:** Doubles forward pass. But the EMA forward is `torch.no_grad()` — no backward pass, no gradient storage. On a 15M param model, the extra forward is ~40% overhead (not 100%, because backward is the expensive part). On our 10-minute budget, this costs ~4 minutes. Tight but possible if training is more efficient per-step.

**Verdict:** Start with Option 1. Keep stop-gradient and EMA as fallback ablations, not as the main design.

### Option 4: Multi-Scale Targets
Predict at multiple time horizons simultaneously:

```python
h = projector(hidden)
scales = [1, 4, 16]
losses = []
for scale in scales:
    if h.size(1) <= scale:
        continue
    # Target: average hidden state over next `scale` positions
    h_cumsum = h.cumsum(dim=1)
    h_pooled = (h_cumsum[:, scale:] - h_cumsum[:, :-scale]) / scale
    h_target = h_pooled                    # detach() only in the stop-grad variant

    h_pred = scale_predictors[scale](h[:, :h_target.size(1)])
    losses.append(F.smooth_l1_loss(h_pred, h_target))

loss = sum(losses) / len(losses) + lambda_sig * sigreg(h)
```

**Scale 1:** "What token comes next?" — fine-grained, equivalent to token prediction
**Scale 4:** "What's the next phrase about?" — compositional understanding
**Scale 16:** "Where is this going?" — discourse planning, abstraction

The model can't memorize BPE patterns to solve scale-16. It must learn to compress and abstract.

**This is the strongest argument for JEPA over CE.** CE can only ask "what's the next token?" JEPA can ask "what's the next sentence about?" — a question CE literally cannot express.

Multi-scale JEPA gives the model a *curriculum of abstraction* that CE lacks entirely.

**Parameter cost:** 3 separate predictor heads. If each is 524K params, total is 1.57M. ~10% of model. May need to shrink the per-scale predictors.

Cheaper option: shared predictor trunk with per-scale output heads:
```python
shared = nn.Linear(d_model, d_model, bias=False)       # 262K
scale_heads = nn.ModuleList([
    nn.Linear(d_model, d_model, bias=False)             # 262K each
    for _ in scales
])
# Total: 262K + 3×262K = 1.05M
```

## Anti-Collapse

Without CE, the model has no external anchor forcing token discrimination. The JEPA loss alone could collapse — the trunk maps everything to the same representation, the predictor outputs a constant, loss is zero.

### Primary Defense: SIGReg

The most compelling update from LeWorldModel is to make anti-collapse **distributional**, not heuristic. Instead of only pushing per-dimension variance or decorrelation, enforce that the latent embeddings look like samples from an isotropic Gaussian.

At a high level:
```python
def sigreg(h, num_proj=32):
    h_flat = h.reshape(-1, h.size(-1))                    # [N, D]
    h_flat = h_flat - h_flat.mean(dim=0, keepdim=True)
    dirs = F.normalize(torch.randn(num_proj, h.size(-1), device=h.device), dim=-1)
    proj = h_flat @ dirs.T                                # [N, P]
    return gaussianity_penalty(proj)                      # e.g. match N(0, 1) moments / test statistic
```

The exact test statistic is implementation detail. The design principle is what matters:
- sample random 1D projections
- penalize deviations from standard normal on each projection
- aggregate across projections

This directly attacks collapse because a constant or low-rank representation cannot match an isotropic Gaussian.

### Why This Beats VICReg-Only

Variance and covariance penalties are still useful sanity baselines, but they are weaker. They constrain first- and second-order structure only. SIGReg constrains the latent distribution more globally and, in the paper, replaced stop-gradient, EMA, and a stack of auxiliary terms.

### Backup Defenses

If SIGReg underperforms in the LM setting, the fallback order should be:
1. `SIGReg + stop-grad`
2. `SIGReg + 0.1 CE anchor`
3. `EMA target`

Not the other way around.

### Embedding Table Anchor

The tied embedding table still provides a weak structural anchor. Even without CE, `tok_emb` receives gradients through the trunk. Monitor `tok_emb.weight` cosine similarities during training. If embeddings start collapsing (all similar), that is the early warning.

## Loss Function

### Pure JEPA (Paradigm A)
```python
loss = pred_loss + lambda_sig * sigreg_loss
```
No CE. No EMA. No stop-gradient by default. Token prediction emerges from representation quality + tied embedding decoding.

### JEPA + Lightweight CE Anchor (Paradigm A')
If pure JEPA collapses or BPB is bad, add a small CE term — not as a primary objective, but as a regularizer that keeps the hidden state space aligned with token space:
```python
loss = pred_loss + lambda_sig * sigreg_loss + 0.1 * ce_loss
```
CE is the safety net, not the objective. The 0.1 weight means: "stay roughly token-aligned, but primarily learn dynamics."

This is the pragmatic middle ground if pure JEPA doesn't work out of the box.

### Legacy Baseline: VICReg-Style JEPA
```python
loss = pred_loss + α * var_loss + β * cov_loss
```
Keep this only as a comparison point. It is no longer the preferred design.

## Eval-Time Decoding

At eval (BPB scoring), the predictor is discarded. Only the trunk matters:
```python
hidden = transformer(tok_emb(tokens))
hidden = final_norm(hidden)
logits = F.linear(hidden, tok_emb.weight)                    # tied embedding decoding
logits = softcap * torch.tanh(logits / softcap)              # logit capping
loss = F.cross_entropy(logits.float(), targets)               # BPB computation
```

This is identical to the baseline eval path. The JEPA predictor adds nothing at eval time — all the value is in the trunk representations it shaped during training.

**Implication:** The predictor parameters don't count toward the 16MB artifact. We only save the trunk. The predictor is a training-time scaffold.

The predictor is free in the artifact, but not free in optimization. It still consumes step time, VRAM, and may make the trunk lazy. Use that freedom to test a slightly larger predictor, not to assume that bigger is better.

## Revised Parameter Budget

| Component | Training Params | Artifact Params | Artifact Size |
|-----------|-----------------|-----------------|---------------|
| Trunk (transformer) | 15M | 15M | ~15MB (int8) |
| tok_emb (tied) | included above | included above | included |
| JEPA predictor | 0.5-2M | **0** | **0** |
| Training-only projector | 0.1-0.3M | **0** | **0** |
| Scale heads | 0.5-1M | **0** | **0** |
| SIGReg / regularizer | 0 | 0 | 0 |
| **Total** | **15.6-18.3M** | **15M** | **~15MB** |

The training-only components do not touch the submission budget, but they still must justify themselves by BPB at 2000 steps.

## Ablation Plan

All runs: 2000 steps, RTX 5090 (~20 min). Baseline: 1.2967 BPB.

| # | Name | Description |
|---|------|-------------|
| 1 | `jepa_sigreg_s1` | Pure JEPA, single-scale, direct self-target, SIGReg |
| 2 | `jepa_sigreg_proj` | Same as #1, but with a small training-only projector after `final_norm` |
| 3 | `jepa_sigreg_anchor01` | JEPA + SIGReg + 0.1×CE anchor |
| 4 | `jepa_sigreg_ms` | JEPA + SIGReg, multi-scale (1,4,16) |
| 5 | `jepa_sigreg_stopgrad` | JEPA + SIGReg with detached target |
| 6 | `jepa_sigreg_predsize` | Medium vs larger predictor, holding the rest fixed |
| 7 | `jepa_vicreg_s1` | Legacy JEPA baseline with var/cov instead of SIGReg |

**Run order:** Start with 1 and 2. If projector-space helps, keep it. If pure SIGReg is unstable or weak, try 3. Run 5 only if the direct objective is clearly unstable. Run 6 after a stable base is established. Keep 7 as a sanity baseline, not the main bet.

**Key metrics:**
- `val_bpb` — the only thing that matters
- `jepa/loss` — should decrease
- `jepa/pred_var` — must stay > 0 (collapse indicator)
- `jepa/sigreg` — should decrease without dominating the prediction loss
- `emb/cosine_mean` — mean pairwise cosine sim of tok_emb, should stay < 0.5
