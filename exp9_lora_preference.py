"""Experiment 9: a *real* LoRA preference-optimization run on one fixed base
model and one fixed dataset, comparing three Psi-PO link functions.

Motivation
----------
Experiments 1-3/7/8 are deliberately tiny toy models, and experiments 4-6 are
inference-only (Ollama cannot compute gradients). This experiment closes the
remaining gap: it actually *fine-tunes* a small language model with LoRA under
several preference objectives, holding the base model, dataset, optimizer,
LoRA configuration, and step budget fixed -- so any difference in the learned
implicit reward is attributable to the loss function alone. This is the
controlled, single-base-model evidence the survey's comparison tables cannot
provide across papers.

Why DPO / IPO / SLiC (and not SimPO / ORPO)
-------------------------------------------
The installed TRL (>=1.6) consolidated its trainers and *removed* the separate
CPO/ORPO trainers (SimPO was implemented inside CPOTrainer, ORPO inside
ORPOTrainer). The DPO-family loss types that remain in `DPOConfig.loss_type`
include `sigmoid` (DPO), `ipo` (IPO), and `hinge` (SLiC-style margin). These
are exactly the three Psi-PO link functions the survey unifies in Section
"The Psi-PO framework" and contrasts in the DPO-vs-IPO saturation toy
(Experiment 3), so this run substantiates that unification with real weights
rather than chasing objectives the toolchain no longer ships. We state this
substitution explicitly rather than silently swapping objectives.

What is reported
----------------
For each objective, after an identical LoRA training budget we report the
held-out implicit-reward accuracy (fraction of pairs with
reward(chosen) > reward(rejected)), the mean reward margin, and the final
train loss. These come straight from TRL's own eval metrics.

Run:  python exp9_lora_preference.py            # full (a few minutes on MPS)
      python exp9_lora_preference.py --smoke    # 2 steps, API smoke test
"""

from __future__ import annotations

import argparse
import csv
import os

import torch


MODEL_NAME = os.environ.get("EXP9_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET = os.environ.get("EXP9_DATASET", "trl-lib/ultrafeedback_binarized")

# (label, TRL loss_type) -- all three are Psi-PO link functions.
OBJECTIVES = [
    ("DPO (sigmoid link)", "sigmoid"),
    ("IPO (squared link)", "ipo"),
    ("SLiC (hinge margin)", "hinge"),
]


def device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_data(n_train: int, n_eval: int):
    from datasets import load_dataset

    train = load_dataset(DATASET, split=f"train[:{n_train}]")
    # Some splits are named "test"; fall back to a tail slice of train.
    try:
        ev = load_dataset(DATASET, split=f"test[:{n_eval}]")
    except Exception:
        ev = load_dataset(DATASET, split=f"train[{n_train}:{n_train + n_eval}]")
    return train, ev


def run_objective(label: str, loss_type: str, train_ds, eval_ds,
                  *, max_steps: int, beta: float, lr: float, seed: int):
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dev = device_str()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    out_dir = os.path.join("results", f"_exp9_{loss_type}")
    cfg = DPOConfig(
        output_dir=out_dir,
        loss_type=[loss_type],
        beta=beta,
        learning_rate=lr,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=max_steps,
        warmup_ratio=0.1,
        logging_steps=max(1, max_steps // 5),
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        max_length=512,
        seed=seed,
        bf16=False, fp16=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # LoRA: reference = adapter-disabled base
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=lora,
    )

    train_out = trainer.train()
    metrics = trainer.evaluate()

    return {
        "objective": label,
        "loss_type": loss_type,
        "train_loss": float(train_out.training_loss),
        "eval_loss": float(metrics.get("eval_loss", float("nan"))),
        "reward_accuracy": float(metrics.get("eval_rewards/accuracies", float("nan"))),
        "reward_margin": float(metrics.get("eval_rewards/margins", float("nan"))),
        "reward_chosen": float(metrics.get("eval_rewards/chosen", float("nan"))),
        "reward_rejected": float(metrics.get("eval_rewards/rejected", float("nan"))),
    }


def plot(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [r["objective"] for r in results]
    acc = [r["reward_accuracy"] for r in results]
    margin = [r["reward_margin"] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(range(len(labels)), acc, color=["#1f77b4", "#d62728", "#2ca02c"])
    ax1.axhline(0.5, ls="--", c="gray", lw=1)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax1.set_ylabel("held-out reward accuracy")
    ax1.set_title("Implicit-reward accuracy (chosen > rejected)")
    ax1.set_ylim(0, 1)
    ax2.bar(range(len(labels)), margin, color=["#1f77b4", "#d62728", "#2ca02c"])
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("mean reward margin")
    ax2.set_title("Implicit-reward margin")
    fig.suptitle("Exp 9: LoRA preference optimization, same base + data "
                 f"({MODEL_NAME.split('/')[-1]})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 steps, tiny data")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.n_train, args.n_eval = 2, 16, 8

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    print(f"\n=== Experiment 9: LoRA preference optimization ===")
    print(f"model={MODEL_NAME}  device={device_str()}  dataset={DATASET}")
    print(f"steps={args.steps}  n_train={args.n_train}  n_eval={args.n_eval}  "
          f"beta={args.beta}  lr={args.lr}\n")

    train_ds, eval_ds = load_data(args.n_train, args.n_eval)

    results = []
    for label, loss_type in OBJECTIVES:
        print(f"--- training {label} (loss_type={loss_type}) ---")
        res = run_objective(label, loss_type, train_ds, eval_ds,
                            max_steps=args.steps, beta=args.beta,
                            lr=args.lr, seed=args.seed)
        print(f"    reward_acc={res['reward_accuracy']:.3f}  "
              f"margin={res['reward_margin']:.3f}  "
              f"train_loss={res['train_loss']:.3f}\n")
        results.append(res)

    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "exp9_lora_preference.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    png_path = os.path.join(out_dir, "exp9_lora_preference.png")
    plot(results, png_path)

    print("--- Summary (held-out) ---")
    for r in results:
        print(f"  {r['objective']:<22} acc={r['reward_accuracy']:.3f}  "
              f"margin={r['reward_margin']:+.3f}")
    print(f"\nSaved: {csv_path}")
    print("All three are Psi-PO link functions trained on the SAME base model "
          "and data; differences isolate the loss function.")


if __name__ == "__main__":
    main()
