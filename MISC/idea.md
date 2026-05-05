# Current Idea and Experiment Design

Date: 2026-05-05

Working title:

```text
Token-Routed Evidence-Calibrated Latent Test-Time Scaling for Multimodal Reasoning
```

This document records the current paper-facing idea, implementation status, and experiment design. It should be treated as the latest snapshot of the method direction, not as a final paper draft.

## 1. Core Problem

Multimodal latent test-time scaling optimizes instance-specific hidden states during inference, without updating model parameters. The current challenge is not merely how to search in latent space, but how to make the latent update respect the different roles of generated tokens in multimodal reasoning.

A single scalar reward applied to all updated latent tokens is too coarse:

```text
policy_loss = -R * sum_i log pi_i
```

This treats visually grounded tokens, uncertain answer tokens, and reasoning-transition tokens as if they should all receive the same optimization signal. For VLM reasoning, this is structurally mismatched:

| Token role | Desired supervision |
|---|---|
| Perception tokens | visual evidence / clue consistency |
| Reasoning tokens | answer likelihood / answer margin / decision uncertainty |
| Formatting or filler tokens | should be weakly updated or mostly anchored |

The current idea is to make latent test-time scaling perception-aware and reasoning-aware at the token level.

## 2. Method Overview

The method keeps the same latent test-time optimization backbone:

1. Run the VLM once to obtain an original answer and generated hidden states.
2. Select a short latent slice from the generated trajectory.
3. Optimize this latent slice at test time.
4. Decode candidate answers from the updated latent states.
5. Score candidates with evidence-calibrated rewards.
6. Use the best candidate reward to update the latent slice.
7. Aggregate or select the final answer.

The main change is how the policy loss is assigned to tokens.

Instead of applying one reward to all latent tokens, we split the reward into:

```text
reasoning_score = answer_weight * (answer_reward - answer_center)
                + margin_weight * margin_score

visual_score    = clue_weight * clue_margin_score
                + crop_weight * crop_reward
```

Current main implementation uses:

```text
answer_weight = 1.0
margin_weight = 0.5
clue_weight = 0.3
crop_weight = 0.0
```

So the current active visual score is driven by relative visual clue consistency, while crop evidence is disabled in the stable main configuration.

## 3. Reward Design

### 3.1 Answer Reward

The answer reward scores a short answer under image-conditioned and text-only contexts:

```text
answer_score(a) =
  log p(a | image, question)
- log p(a | question)
```

The goal is to reward answers that are more supported when the image is available, rather than answers that are likely from language priors alone.

### 3.2 Answer Margin

The margin term compares the selected candidate answer against competing answer hypotheses:

```text
margin_score(a) =
  tanh(score(a) - max score(competing answers))
```

Competitors include the original answer and candidates decoded during the same latent optimization step.

This converts answer scoring from an absolute signal into a relative hypothesis-selection signal.

### 3.3 Visual Clue Margin

The visual clue component uses generated visual evidence statements as a compact perception proxy.

For a candidate answer, the method estimates whether the visual clues are better supported by the image when conditioned on that answer:

```text
clue_score(a) =
  log p(clues | image, question, a)
- log p(clues | question, a)
```

Then it uses a relative margin:

```text
clue_margin_score(a) =
  tanh(clue_score(a) - clue_score(competing answer baseline))
```

This is the key perception-side reward in the current stable method.

### 3.4 Crop / Mask Evidence

The phase3 line additionally explored crop evidence:

```text
visual_score =
  clue_weight * clue_margin_score
+ crop_weight * crop_reward
```

However, phase3 did not consistently outperform phase2. The current main line therefore keeps crop evidence as an extension or ablation, not the default method.

## 4. Token Routing

The current implementation supports token-disentangled latent optimization. The most reliable version is hard top-k routing.

### 4.1 Visual Token Score

The visual score for each updated latent token is computed from similarity to the visual-token prototype:

```text
visual_focus_i =
  0.5 * (cosine(h_i, mean(image_token_hidden_states)) + 1)
```

This avoids relying on attention maps, which can be unstable or inaccessible under optimized attention backends.

### 4.2 Reasoning Token Score

Reasoning tokens are identified by token entropy from the LM head:

```text
entropy_i = -sum_v p_i(v) log p_i(v)
```

High entropy indicates uncertainty or decision sensitivity.

### 4.3 Hard Top-K Router

The current stable router is:

```text
visual tokens:
  top 25% tokens by visual_focus

reasoning tokens:
  top 25% tokens by entropy among non-visual tokens
```

The two masks are disjoint:

```text
w_visual_i in {0, 1}
w_reason_i in {0, 1}
w_visual_i * w_reason_i = 0
```

In a typical update slice with about 43 latent tokens, the router selects approximately:

```text
visual tokens: 11
reasoning tokens: 8
```

Unselected tokens remain in the update slice, but mainly receive anchor regularization rather than direct policy-gradient reward.

## 5. Loss Function

The current token-disentangled loss is:

```text
L = L_policy + L_logit + L_anchor
```

Policy loss:

```text
L_policy =
  - reason_policy_weight * reasoning_score * sum_i w_reason_i * log pi_i
  - visual_policy_weight * visual_score    * sum_i w_visual_i * log pi_i
```

Logit entropy loss:

```text
L_logit =
  logit_lr * weighted_mean_i(entropy_i, w_reason_i)
```

Anchor loss:

```text
L_anchor =
  anchor_weight * MSE(H_optimized, H_original)
```

This means:

| Component | Role |
|---|---|
| `reasoning_score` | updates uncertain answer/reasoning tokens |
| `visual_score` | updates visually grounded perception tokens |
| `logit_loss` | mainly stabilizes reasoning-token confidence |
| `anchor_loss` | prevents excessive drift from original VLM hidden states |

## 6. Novelty

The novelty is not only using visual reward. The current method has three separable contributions:

### 6.1 Evidence-Calibrated Reward

The reward compares candidate answers using image-conditioned likelihood, text-only likelihood, answer margin, and visual clue margin. This is more targeted than answer-only reward.

### 6.2 Token-Routed Latent Optimization

The latent update is routed by token function:

```text
perception token  <- visual clue reward
reasoning token   <- answer / margin reward
```

This converts latent TTS from a global scalar update into a role-aware update.

### 6.3 Parallel Hypothesis Search in Latent Space

Each latent step can decode multiple candidates from the same optimized latent slice. Candidates are scored relatively, and the best hypothesis determines the update signal. This makes the method closer to latent-space search than single-trajectory refinement.

## 7. Current Method Variants

| Variant | Meaning | Current role |
|---|---|---|
| `phase2_global` | phase2 reward with global loss | strongest clean non-routing baseline |
| `td_visual_entropy` | soft visual-focus + entropy router | early token-disentangle version |
| `hard_topk` | hard visual top-k + reasoning top-k router | current main token-routing variant |
| `hard_topk_span` | hard top-k with span constraints | routing ablation |
| `entropy_only` | reasoning router only | tests whether visual routing matters |
| `visual_only_router` | visual router only | tests whether entropy branch matters |
| `reason_branch_only` | disable visual policy branch | tests visual-score contribution |
| `visual_branch_only` | disable reasoning policy branch | tests answer/reasoning contribution |
| `phase3_crop` | adds crop evidence reward | extension, not current default |

## 8. Main Claims to Validate

### Claim 1

Evidence-calibrated latent refinement improves VLM reasoning over direct CoT and output-space sampling baselines.

Required evidence:

```text
Ours > CoT
Ours competitive with or better than self-consistency / best-of-N
Ours improves across multiple benchmark types
```

### Claim 2

Token-routed latent optimization is better aligned with multimodal reasoning than a single global latent reward.

Required evidence:

```text
hard_topk or token-disentangle >= phase2_global
visual/reasoning routing ablations show non-trivial contribution
diagnostics show selected visual and reasoning tokens differ
```

### Claim 3

The method is model-general and not tied to Qwen2.5-VL.

Required evidence:

```text
Qwen2.5-VL-7B results
InternVL3.5-8B results
Optional: InternVL3.5-4B scaling trend
```

### Claim 4

The gain is not explained only by output-space reranking.

Required evidence:

```text
Compare against reward-only rerank
Compare against best-of-N visual
Compare against self-consistency
Compare against original LatentSeek-style rule verifier baseline
```

## 9. Experiment Design

### 9.1 Backbones

| Backbone | Role |
|---|---|
| Qwen2.5-VL-7B-Instruct | main backbone |
| InternVL3.5-8B-Instruct | transfer backbone |
| InternVL3.5-4B-Instruct | scale-down robustness check |
| Qwen2.5-VL-3B-Instruct | optional small-model table |

### 9.2 Benchmarks

| Benchmark | Type |
|---|---|
| MathVista testmini | visual math and chart reasoning |
| MMStar val | multimodal perception and reasoning |
| LogicVista train subset | visual logical reasoning |
| RealWorldQA test | real-world visual QA |
| HallusionBench image split | hallucination and visual consistency |
| ScienceQA-IMG test | science image QA |
| MMMU validation single-image | broad multimodal subject reasoning |

For MMMU, only single-image validation cases are used in the current code path. Multi-image cases were filtered out to match the current single-image inference pipeline.

### 9.3 Baselines

| Baseline | Purpose |
|---|---|
| CoT / direct generation | base VLM capability |
| Self-consistency | output-space sampling baseline |
| Best-of-N visual | sample multiple answers and select by visual reward |
| Reward-only rerank | use the reward without latent update |
| Original LatentSeek-Text | original text-only rule verifier adaptation |
| LatentSeek-MM-Rule | LatentSeek text rules plus visual grounding verifier |
| Phase2 global | our reward without token routing |

### 9.4 Main Method

The current paper-facing method should be reported as:

```text
ECL-TTS with token-routed hard-topk latent optimization
```

Default configuration:

```text
answer_weight = 1.0
margin_weight = 0.5
clue_weight = 0.3
crop_weight = 0.0
token_router = hard_topk
visual_topk_ratio = 0.25
reason_topk_ratio = 0.25
max_num_steps = 4
rho = 0.5
latent_start_strategy = full
anchor_weight = 0.05
logit_lr = 0.01
```

LatentSeek baseline uses a different configuration to match the original paper more closely:

```text
max_num_steps = 10
rho = 0.2
lr = 0.05
verifier_max_new_tokens = 4096
```

## 10. Current Result Snapshot

### 10.1 Qwen2.5-VL-7B Hard-TopK Token Routing

| Benchmark | Original | Optimized | Delta | Status |
|---|---:|---:|---:|---|
| MathVista | 67.80 | 70.20 | +2.40 | complete |
| MMStar | 62.00 | 64.33 | +2.33 | complete |
| LogicVista | 44.52 | 46.53 | +2.01 | 447 evaluated |
| RealWorldQA | 63.66 | 67.06 | +3.40 | complete |
| HallusionBench | 70.56 | 71.82 | +1.26 | complete |
| ScienceQA-IMG | 89.74 | 90.43 | +0.69 | complete |
| MMMU single-image val | 50.76 | 51.23 | +0.47 | complete |

Interpretation:

```text
The token-routed method is consistently positive on Qwen2.5-VL-7B, with strongest gains on RealWorldQA, MathVista, MMStar, and LogicVista.
```

### 10.2 InternVL3.5-8B Hard-TopK Token Routing

| Benchmark | Original | Optimized | Delta | Status |
|---|---:|---:|---:|---|
| MathVista | 69.75 | 72.06 | +2.31 | 995 / 1000 partial |
| MMStar | 67.13 | 68.73 | +1.60 | complete |
| RealWorldQA | 65.36 | 65.49 | +0.13 | complete |
| HallusionBench | 66.04 | 66.46 | +0.42 | complete |
| ScienceQA-IMG | 95.04 | 95.74 | +0.69 | complete |
| MMMU single-image val | 59.39 | 60.79 | +1.40 | complete |
| LogicVista | pending | pending | pending | not confirmed |

Interpretation:

```text
Transfer to InternVL is positive but smaller than Qwen on open-world and hallucination benchmarks. MathVista, MMStar, ScienceQA, and MMMU remain useful evidence.
```

### 10.3 Original LatentSeek Baseline Status

Current implementation added two rule-verifier baselines:

| Baseline | Rules |
|---|---|
| `original_latentseek_text_rules` | original four text verifier rules |
| `original_latentseek_multimodal_rules` | original four text rules plus one visual grounding rule |

Current running status:

| Baseline | Benchmark | Original | Optimized | Status |
|---|---|---:|---:|---|
| Text rules | MathVista | 67.80 | 70.20 | complete |
| Text rules | MMStar | 62.00 | 61.60 | complete |
| Text rules | LogicVista | 44.52 | 44.07 | complete |
| Text rules | RealWorldQA | 63.66 | 63.79 | complete |
| Text rules | HallusionBench | running | running | in progress |
| Text rules | ScienceQA-IMG | pending | pending | not started |
| Text rules | MMMU | pending | pending | not started |
| Multimodal rules | MathVista | 68.70 | 71.10 | complete |
| Multimodal rules | MMStar | 62.79 | 61.83 | complete |

Interpretation:

```text
The LatentSeek-style rule baseline is useful but not clearly stronger than our phase2/token-routing line. The multimodal rule version helps MathVista but hurts MMStar, suggesting that a naive visual-grounding verifier is not a robust multimodal reward.
```

## 11. Ablation Plan

### 11.1 Reward Ablations

| Ablation | Purpose |
|---|---|
| answer only | tests whether visual evidence is necessary |
| answer + margin | tests relative answer hypothesis selection |
| answer + clue margin | tests visual evidence calibration |
| answer + clue margin + crop | tests localized evidence extension |

### 11.2 Routing Ablations

| Ablation | Purpose |
|---|---|
| phase2_global | no token routing |
| soft visual + entropy | soft token routing |
| hard_topk | current stable router |
| hard_topk_span | span-constrained router |
| entropy_only | no visual token branch |
| visual_only_router | no entropy branch |
| reason_branch_only | no visual policy loss |
| visual_branch_only | no reasoning policy loss |

### 11.3 Baseline Ablations

| Baseline | Purpose |
|---|---|
| self-consistency | output-space test-time scaling |
| best-of-N visual | output-space visual reward selection |
| reward-only rerank | isolates reward from latent update |
| original LatentSeek text | tests pure text verifier latent update |
| LatentSeek multimodal rule | tests naive visual verifier latent update |

## 12. Diagnostics to Save

Each main run should save:

```text
original answer
optimized answer
original short answer
optimized short answer
reward_history
answer_reward
margin_score
clue_margin_score
reasoning_score
visual_score
policy_loss
logit_loss
anchor_loss
visual token count
reasoning token count
mean visual_focus
mean entropy
wrong -> correct
correct -> wrong
```

These diagnostics support two paper figures:

| Figure | Content |
|---|---|
| Token routing figure | visual-focus top tokens vs entropy top tokens |
| Reward decomposition figure | answer/margin/clue contributions across corrected cases |

## 13. Paper Narrative

The paper should not claim that we improve free-form chain-of-thought reasoning. The safer claim is:

```text
We improve decision-time multimodal reasoning by routing evidence-calibrated rewards to functionally different latent tokens.
```

Suggested wording:

```text
Perception is handled through visual clue consistency and visual-focus token routing.
Reasoning is handled through answer-hypothesis comparison, answer margin, and high-entropy decision-token routing.
```

This gives a clean perception/reasoning story without overclaiming that the model's natural-language rationale is explicitly optimized.

## 14. Immediate Next Experiments

1. Finish `original_latentseek_text_rules` on HallusionBench, ScienceQA, and MMMU.
2. Finish `original_latentseek_multimodal_rules` beyond MathVista/MMStar if compute allows.
3. Aggregate final Qwen and InternVL tables with:

```text
CoT
Self-consistency
Best-of-N
Reward-only rerank
Original LatentSeek-Text
LatentSeek-MM-Rule
Ours
```

4. Run or confirm LogicVista for InternVL3.5-8B hard-topk.
5. Produce ablation table on LogicVista and MathVista for token routing.
6. Save diagnostic examples for qualitative analysis.

## 15. Current Risks

| Risk | Mitigation |
|---|---|
| Visual clue reward may be weak on open-world QA | report benchmark-specific behavior and keep clue reward as relative margin |
| Hard top-k routing may look heuristic | justify as stable, inspectable, and ablate against soft routing |
| LatentSeek rule baseline is expensive | run full text baseline first, multimodal rule as targeted ablation |
| HallusionBench / ScienceQA duplicate IDs affect per-example files | use shard-level `summary.json` for accuracy |
| MMMU multi-image cases are excluded | state current single-image validation protocol explicitly |

## 16. One-Sentence Summary

The current method performs test-time latent refinement for VLMs by decomposing multimodal reward into perception and reasoning signals, then routing these signals to visually grounded and high-entropy latent tokens through hard top-k token selection.
