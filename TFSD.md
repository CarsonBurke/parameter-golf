# Teacher-Free Self-Distillation (TFSD)

Source: https://pisoni.ai/posts/teacher-free-self-distillation/

## Core Idea
Replace dot-product logits with negative squared Euclidean distances to class centroids, then self-distill using the model's own predictions as soft targets.

## Mechanism

### Standard CE
```
logit_i = w_i · x         # unbounded dot product
loss = CE(softmax(logits), one_hot(target))
```

### TFSD
```
logit_i = -||x - c_i||^2  # bounded, max at 0 when x == c_i
soft_target = softmax(logits with true class masked to 0)
loss = KL(softmax(logits), soft_target)
```

The "zero-masked" target: take the model's current distance-based logits, set the true class logit to 0 (perfect alignment), leave all other class logits as-is. This preserves inter-class relationships ("dark knowledge") while pulling toward the correct class.

## What It Claims to Solve
- Unbounded logit growth / gradient explosions in CE
- Loss spikes during training
- Semantic fracturing (synonyms pushed apart unnecessarily)
- Better OOD detection via distance thresholds

## Concerns for Parameter Golf
1. **No empirical results** — entirely theoretical, no pretraining benchmarks
2. **Logit softcap already bounds logits** — baseline uses tanh softcap at 30.0
3. **Breaks tied embeddings** — needs separate centroid matrix, costs params under 16MB cap
4. **Self-distillation weak early** — model predictions are noise at training start
5. **Vocab-size distance computation** — O(V × d) per token for centroids

## Potential Adaptations
If we wanted to test a lighter version without breaking tied embeddings:

1. **Soft-target only**: Keep standard dot-product logits but use the zero-masked soft target trick with KL divergence. No architecture change, just loss modification.
2. **Hybrid loss**: `loss = α * CE(logits, target) + (1-α) * KL(logits, soft_target)` with α annealed from 1→0.5 as training progresses (so early training uses hard targets, later uses soft).
3. **Distance regularizer**: Add a small penalty pulling embeddings toward their centroid positions without replacing the logit computation.

## Ablation Plan (if pursued)
```bash
# Would need a custom training script with modified loss
python3 ablation.py --steps 2000 --name tfsd_hybrid --script tfsd_train_gpt.py
```

Compare against baseline_2k. Minimum viable test: hybrid loss (option 2 above) — ~20 lines of code change, no architecture modification.

## Verdict
Interesting theory, unproven. Low priority unless we run out of higher-confidence ideas (NorMuon, architecture changes, quantization improvements). The hybrid soft-target approach is the only variant worth testing — minimal risk, no param cost.
