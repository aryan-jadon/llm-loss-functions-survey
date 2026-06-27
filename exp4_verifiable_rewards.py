"""Experiment 4 -- Verifiable rewards: best-of-N and self-consistency.

Validates the central DeepSeek-R1 / GRPO claim (Section 5.x and Section 8):
when a *verifiable* reward is available (here, exact-match math correctness),
sampling multiple solutions and aggregating them sharply improves accuracy over
a single greedy sample -- the mechanism that makes rule-based verifiable rewards
effective without a learned reward model.

We compare three inference-time strategies on the same problems / same model:
  * greedy       : 1 sample at temperature 0 (pass@1 baseline)
  * majority@N   : N samples, self-consistency majority vote (Wang et al.)
  * best-of-N    : N samples, an ORACLE verifier keeps any correct one
                   (pass@N -- the upper bound a perfect verifiable reward gives)

This is inference-only, so it runs on Ollama (Metal GPU) with no training.

Usage:
    python exp4_verifiable_rewards.py --n 8 --limit 20
"""

from __future__ import annotations

import argparse
import csv
import os

from math_problems import PROBLEMS
from ollama_utils import (
    DEFAULT_MODEL,
    extract_final_number,
    generate,
    is_correct,
    majority_vote,
    require_ready,
)

COT_SYSTEM = (
    "You are a careful math tutor. Solve the problem step by step. "
    "End your response with a line of the form '#### <number>' giving the final "
    "numeric answer only."
)


def solve_once(question: str, *, model: str, temperature: float,
               seed: int | None) -> float | None:
    prompt = f"{question}\n\nThink step by step, then give the final answer."
    text = generate(prompt, model=model, temperature=temperature,
                    system=COT_SYSTEM, num_predict=512, seed=seed)
    return extract_final_number(text)


def run(model: str, n_samples: int, limit: int, temperature: float) -> dict:
    problems = PROBLEMS[:limit]
    greedy_hits = majority_hits = bestof_hits = 0

    for idx, prob in enumerate(problems):
        # Greedy / pass@1 baseline (temperature 0, fixed seed).
        greedy_pred = solve_once(prob.question, model=model, temperature=0.0, seed=0)
        greedy_ok = is_correct(greedy_pred, prob.answer)

        # N stochastic samples reused for both majority vote and best-of-N.
        preds = [solve_once(prob.question, model=model, temperature=temperature,
                            seed=1000 + k) for k in range(n_samples)]
        maj_pred = majority_vote(preds)
        maj_ok = is_correct(maj_pred, prob.answer)
        bestof_ok = any(is_correct(p, prob.answer) for p in preds)

        greedy_hits += greedy_ok
        majority_hits += maj_ok
        bestof_hits += bestof_ok
        print(f"  [{idx + 1:>2}/{len(problems)}] gold={prob.answer:g}  "
              f"greedy={'Y' if greedy_ok else 'n'}  "
              f"maj@{n_samples}={'Y' if maj_ok else 'n'}  "
              f"best@{n_samples}={'Y' if bestof_ok else 'n'}")

    total = len(problems)
    return {
        "n_problems": total,
        "n_samples": n_samples,
        "greedy_acc": greedy_hits / total,
        "majority_acc": majority_hits / total,
        "bestof_acc": bestof_hits / total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=8, help="samples per problem")
    ap.add_argument("--limit", type=int, default=len(PROBLEMS))
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", default="exp4_verifiable_rewards.csv",
                    help="output CSV filename (written under results/)")
    args = ap.parse_args()

    require_ready(args.model)

    print("\n=== Experiment 4: verifiable rewards (best-of-N + majority) ===")
    print(f"model={args.model}  N={args.n}  problems={min(args.limit, len(PROBLEMS))}"
          f"  temp={args.temperature}\n")

    res = run(args.model, args.n, args.limit, args.temperature)

    print("\n--- Accuracy ---")
    print(f"  greedy (pass@1)        : {res['greedy_acc']:.3f}")
    print(f"  majority@{res['n_samples']} (self-consist.): {res['majority_acc']:.3f}")
    print(f"  best-of-{res['n_samples']} (oracle/pass@N): {res['bestof_acc']:.3f}")

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, args.out)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "accuracy", "n_problems", "n_samples", "model"])
        w.writerow(["greedy_pass@1", res["greedy_acc"], res["n_problems"], 1, args.model])
        w.writerow(["majority_vote", res["majority_acc"], res["n_problems"],
                    res["n_samples"], args.model])
        w.writerow(["best_of_n_oracle", res["bestof_acc"], res["n_problems"],
                    res["n_samples"], args.model])
    print(f"\nSaved: {csv_path}")
    print("Expectation (paper): best-of-N >= majority@N >= greedy. Verifiable "
          "rewards turn extra inference compute into accuracy gains.")


if __name__ == "__main__":
    main()
