# LLM Loss Functions Survey

A reproducible code companion to a survey of loss functions and alignment objectives for large language models. Each of the nine experiments is a self-contained script that validates one claim from the survey's sections on policy-gradient methods, offline preference optimization, and inference-time strategies.

---

## Repository layout

```
preference_losses.py    # 9 pairwise preference objectives (DPO, IPO, SimPO, …)
toy_env.py              # token-level unigram toy environment
ollama_utils.py         # minimal Ollama HTTP client + answer-extraction helpers
math_problems.py        # 20 inline GSM8K-style math problems (no download needed)

exp1_reinforce_variance.py      # REINFORCE variance-reduction family
exp2_preference_objectives.py   # controlled comparison of offline objectives
exp3_dpo_vs_ipo_saturation.py   # DPO gradient saturation vs. IPO squared loss
exp4_verifiable_rewards.py      # best-of-N and majority voting (Ollama)
exp5_self_rewarding.py          # LLM-as-judge self-reward loop (Ollama)
exp6_cot_distillation.py        # CoT distillation data generation (Ollama)
exp7_length_generation.py       # generation-time length control (R-DPO / SLiC)
exp8_tdpo_token_credit.py       # token-level credit: DPO vs. TDPO mechanism
exp9_lora_preference.py         # real LoRA fine-tuning: DPO / IPO / SLiC

results/                        # CSV tables and PNG plots written by each script
```

---

## Setup

### Core dependencies (experiments 1–3, 7–8)

```bash
pip install numpy torch matplotlib
```

### Ollama (experiments 4–6)

Experiments 4, 5, and 6 call a locally running [Ollama](https://ollama.com) server for inference. No training occurs; gradients are never computed.

```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:7b          # default model; override with OLLAMA_MODEL env var
```

Set `OLLAMA_HOST` to point at a remote server if needed (default: `http://localhost:11434`).

### LoRA fine-tuning (experiment 9 only)

Experiment 9 requires the heavy ML stack. Uncomment the optional lines in `requirements.txt` and install:

```bash
pip install transformers>=4.44 peft>=0.12 trl>=1.6 datasets>=2.20 accelerate>=0.34
```

---

## Running the experiments

### Toy experiments — no model server required

```bash
python exp1_reinforce_variance.py
python exp2_preference_objectives.py
python exp3_dpo_vs_ipo_saturation.py
python exp7_length_generation.py
python exp8_tdpo_token_credit.py
```

Each script writes a CSV and (if matplotlib is installed) a PNG to `results/`.

### Ollama experiments

Start Ollama first, then:

```bash
python exp4_verifiable_rewards.py --n 8 --limit 20
python exp5_self_rewarding.py --k 4 --limit 12
python exp6_cot_distillation.py --traces 3 --limit 15
```

Pass `--model <name>` to use a different Ollama model. All three accept `--limit N` to run on a subset of the 20 math problems for a quick sanity check.

### LoRA fine-tuning

```bash
python exp9_lora_preference.py --smoke       # 2-step API smoke test
python exp9_lora_preference.py               # full run (~few minutes on MPS/GPU)
```

Override the base model or dataset with environment variables:

```bash
EXP9_MODEL=Qwen/Qwen2.5-0.5B-Instruct EXP9_DATASET=trl-lib/ultrafeedback_binarized \
    python exp9_lora_preference.py
```

---

## Experiment descriptions

### Exp 1 — REINFORCE variance-reduction family (`exp1_reinforce_variance.py`)

**Claim:** REINFORCE, REINFORCE++, RLOO, and GRPO apply the same policy-gradient update and differ only in how they construct the advantage. Each step reduces gradient-estimator variance without a learned value function.

A softmax policy over 6 actions generates 40,000 groups of 8 rollouts. Gradient-estimator variance `E‖g − E[g]‖²` is recorded for each estimator.

| Estimator | Baseline |
|-----------|----------|
| REINFORCE | none (raw reward) |
| REINFORCE++ | online EMA (lagging) |
| RLOO | per-group leave-one-out mean |
| GRPO | group z-score (different scale) |

**Output:** `results/exp1_reinforce_variance.{csv,png}`

---

### Exp 2 — Controlled comparison of offline preference objectives (`exp2_preference_objectives.py`)

**Claim:** When preference data is length-confounded (annotators prefer longer responses regardless of quality), reference-based *sum*-reward objectives (DPO, IPO, CPO, KTO) absorb the length signal and become length-biased; length-normalized objectives (SimPO, ORPO) cannot encode a pure-length preference.

Uses a transparent parametric implicit-reward model `logp_θ(y) = (C0 + δ + γ·q(y)) · L(y)` with two learnable scalars (δ = length coefficient, γ = quality coefficient). All methods are trained on the same length-confounded Bradley-Terry pairs from the same reference.

Metrics reported: ranking accuracy on true quality `q`, length–reward Spearman correlation, and the learned δ and γ.

**Output:** `results/exp2_preference_objectives.{csv,png}`

---

### Exp 3 — DPO saturation vs. IPO squared loss (`exp3_dpo_vs_ipo_saturation.py`)

**Claim:** DPO's `-log σ(β h)` gives a gradient weight `σ(-β h)` that vanishes as the margin `h` grows (already-correct pairs stop learning). IPO's `(h - 1/2τ)²` keeps a gradient that grows linearly in the margin error.

Sweeps `h` from -2 to 8 and plots `|∂L/∂h|` for each objective.

**Output:** `results/exp3_dpo_vs_ipo_saturation.{csv,png}`

---

### Exp 4 — Verifiable rewards: best-of-N and self-consistency (`exp4_verifiable_rewards.py`)

**Claim:** With a verifiable reward (exact-match math correctness), sampling multiple solutions and aggregating sharply improves accuracy over a single greedy sample — the mechanism behind rule-based GRPO / DeepSeek-R1.

Three strategies on 20 GSM8K-style problems via Ollama:

| Strategy | Description |
|----------|-------------|
| greedy (pass@1) | temperature 0, single sample |
| majority@N | N stochastic samples, self-consistency vote |
| best-of-N (oracle) | N stochastic samples, any-correct oracle (pass@N upper bound) |

**Output:** `results/exp4_verifiable_rewards.csv`

---

### Exp 5 — Self-rewarding / LLM-as-judge (`exp5_self_rewarding.py`)

**Claim:** A model can act as its own reward model, scoring candidate responses and selecting (y_w, y_l) preference pairs — the data-generation step of the iterative self-rewarding loop — without human labels.

For each problem: generate K candidate solutions, ask the same model to score each 1–10, pick best/worst, then check agreement against the verifiable math answer.

Metrics: chosen-best accuracy (y_w is correct) and judge-ranking agreement (judge ranked a correct answer above an incorrect one on mixed problems).

**Output:** `results/exp5_self_rewarding.csv`

---

### Exp 6 — CoT distillation data generation (`exp6_cot_distillation.py`)

**Claim:** Filtering teacher rationales to those that reach the correct answer (Magister et al.) yields higher-quality distillation signal than using all traces.

A teacher model generates step-by-step reasoning traces via Ollama. Only traces whose extracted final number matches the ground-truth answer are kept. The script reports acceptance rate and problem coverage, and writes an accepted-traces JSONL file that a downstream SFT step would consume.

**Output:** `results/exp6_cot_distillation.csv`, `results/exp6_cot_distillation_dataset.jsonl`

---

### Exp 7 — Generation-time length control (`exp7_length_generation.py`)

**Claim:** Under length-confounded preferences, DPO and SLiC lower the EOS probability and generate longer text; SimPO's length normalization keeps generated length near the reference. R-DPO's penalty is re-absorbed in a unigram setting (documented limitation).

A unigram+EOS generative toy (6 content tokens + EOS) is trained under each objective on length-confounded Bradley-Terry pairs. After training, 4,000 fresh sequences are sampled and mean generated length and quality are reported.

**Output:** `results/exp7_length_generation.{csv,png}`

---

### Exp 8 — Token-level credit assignment: DPO vs. TDPO (`exp8_tdpo_token_credit.py`)

**Claim:** Standard DPO assigns an identical sequence-level gradient weight to every token. A token-factorized credit assignment (TDPO-style) concentrates gradient mass on *discriminative* tokens that still need to move, and assigns nearly zero weight to already-separated tokens.

Toy: T independent token slots, half already-separated (large positive reference logit), half needs-work (logit ≈ 0). Per-token `|∂L/∂θ_i|` is compared between DPO (uniform) and TDPO-style (per-token sigmoid).

**Output:** `results/exp8_tdpo_token_credit.{csv,png}`

---

### Exp 9 — LoRA preference optimization (`exp9_lora_preference.py`)

**Claim:** Holding the base model, dataset, optimizer, LoRA config, and step budget fixed, differences in held-out implicit-reward accuracy are attributable to the loss function alone — the controlled evidence the survey's cross-paper comparison tables cannot provide.

Trains `Qwen/Qwen2.5-0.5B-Instruct` with LoRA on `trl-lib/ultrafeedback_binarized` under three Psi-PO link functions available in TRL ≥ 1.6:

| Label | TRL loss_type | Link function |
|-------|---------------|---------------|
| DPO (sigmoid link) | `sigmoid` | `-log σ(β h)` |
| IPO (squared link) | `ipo` | `(h - 1/2τ)²` |
| SLiC (hinge margin) | `hinge` | `max(0, δ - h)` |

Reports held-out reward accuracy (fraction of pairs with `reward(chosen) > reward(rejected)`), mean reward margin, and final train loss.

**Output:** `results/exp9_lora_preference.{csv,png}`

---

## Core modules

### `preference_losses.py`

Implements all 9 pairwise preference objectives as pure-PyTorch scalar losses operating on sequence-level log-probabilities. All share a common calling convention:

```python
loss = dpo_loss(logp_w, logp_ref_w, logp_l, logp_ref_l, beta=0.1)
```

| Function | Paper | Key property |
|----------|-------|--------------|
| `dpo_loss` | Rafailov et al., 2023 | Reference-based, log-sigmoid |
| `ipo_loss` | Azar et al., 2023 | Squared loss, avoids saturation |
| `simpo_loss` | Meng et al., 2024 | Reference-free, length-normalized |
| `orpo_loss` | Hong et al., 2024 | SFT + odds-ratio, reference-free |
| `cpo_loss` | Xu et al., 2024 | DPO + behavioral-cloning NLL anchor |
| `slic_loss` | Zhao et al., 2022 | Margin hinge + SFT regularizer |
| `rdpo_loss` | Park et al., 2024 | DPO with explicit length penalty |
| `rebel_loss` | Gao et al., 2024 | Regresses implicit reward gap onto scalar signal |
| `kto_loss` | Ethayarajh et al., 2024 | Prospect-theory value, unpaired-friendly |

All functions are registered in `PAIRWISE_LOSSES` dict for use by experiment scripts.

### `toy_env.py`

A token-level unigram policy over a small vocabulary. Sequence log-probabilities are exact dot products `counts @ log_softmax(theta)`, making every gradient computation fully vectorized while preserving DPO's length-bias mechanism. Provides:

- `build_toy_env` — constructs response pool with length-independent quality scores
- `sample_preference_pairs` — Bradley-Terry pairs with optional length-bias confound
- `ranking_accuracy` — fraction of pairs ordered correctly by an implicit reward
- `length_reward_correlation` — Spearman-style rank correlation between length and reward

### `ollama_utils.py`

Zero-dependency (stdlib only) HTTP client for the Ollama `/api/generate` endpoint. Also provides:

- `require_ready` — fails fast with actionable instructions if server or model is missing
- `extract_final_number` — parses GSM8K-style `#### <n>` markers and numeric phrases
- `majority_vote` — self-consistency aggregation over a list of numeric predictions

### `math_problems.py`

20 inline `MathProblem(question, answer)` instances styled after GSM8K, ordered hardest-first so small `--limit` runs still exercise problems where greedy decoding fails.

---

## Results summary

All outputs land in `results/`. Pre-computed results from the original runs are included in the repository.

| Experiment | Output files |
|------------|-------------|
| Exp 1 | `exp1_reinforce_variance.{csv,png}` |
| Exp 2 | `exp2_preference_objectives.{csv,png}` |
| Exp 3 | `exp3_dpo_vs_ipo_saturation.{csv,png}` |
| Exp 4 | `exp4_verifiable_rewards.csv` |
| Exp 5 | `exp5_self_rewarding.csv` |
| Exp 6 | `exp6_cot_distillation.{csv,jsonl}` |
| Exp 7 | `exp7_length_generation.{csv,png}` |
| Exp 8 | `exp8_tdpo_token_credit.{csv,png}` |
| Exp 9 | `exp9_lora_preference.{csv,png}` |

---

## License

MIT © 2026 Aryan Jadon
