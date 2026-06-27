"""Experiment 5 -- Self-Rewarding / LLM-as-judge.

Validates the Self-Rewarding Language Models mechanism (Section 5): a model can
act as its own reward model, scoring candidate responses and selecting a
preferred (y_w) and dispreferred (y_l) pair -- the data that would feed a DPO
step in the iterative self-rewarding loop. No human labels and no training are
required for the data-generation step, so it runs on Ollama.

We test whether the model's *self-assigned* scores agree with a verifiable
ground-truth signal (math correctness). High agreement means the LLM-as-judge
produces useful preference pairs; this is the key assumption behind self-reward.

For each problem:
  1. Generate K candidate solutions (varied temperature).
  2. Ask the SAME model to score each candidate 1-10 (LLM-as-judge).
  3. Pick y_w = argmax score, y_l = argmin score.
  4. Check, using the verifiable answer, whether y_w is actually correct and
     whether the judge ranked a correct answer above an incorrect one.

Usage:
    python exp5_self_rewarding.py --k 4 --limit 12
"""

from __future__ import annotations

import argparse
import csv
import os
import re

from math_problems import PROBLEMS
from ollama_utils import (
    DEFAULT_MODEL,
    extract_final_number,
    generate,
    is_correct,
    require_ready,
)

SOLVE_SYSTEM = (
    "You are a math solver. Solve step by step and end with '#### <number>'."
)

JUDGE_SYSTEM = (
    "You are a strict grader. You will see a math problem and a candidate "
    "solution. Rate the solution's correctness and reasoning quality on an "
    "integer scale from 1 (clearly wrong) to 10 (clearly correct and complete). "
    "Respond with ONLY the line 'Score: <n>'."
)

_SCORE_RE = re.compile(r"score\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_score(text: str) -> float | None:
    m = _SCORE_RE.search(text or "")
    if m:
        try:
            return max(1.0, min(10.0, float(m.group(1))))
        except ValueError:
            return None
    return None


def judge(question: str, solution: str, *, model: str) -> float | None:
    prompt = (f"Problem:\n{question}\n\nCandidate solution:\n{solution}\n\n"
              f"Give your score.")
    text = generate(prompt, model=model, temperature=0.0,
                    system=JUDGE_SYSTEM, num_predict=32)
    return parse_score(text)


def run(model: str, k: int, limit: int, temperature: float) -> dict:
    problems = PROBLEMS[:limit]
    yw_correct = 0          # chosen-as-best is actually correct
    valid_pairs = 0         # problems with >=1 correct and >=1 incorrect candidate
    judge_ranks_right = 0   # among those, judge put a correct above an incorrect

    for idx, prob in enumerate(problems):
        cands = []
        for s in range(k):
            text = generate(prob.question, model=model,
                            temperature=temperature, system=SOLVE_SYSTEM,
                            num_predict=512, seed=2000 + s)
            pred = extract_final_number(text)
            score = judge(prob.question, text, model=model)
            cands.append({"text": text, "pred": pred,
                          "correct": is_correct(pred, prob.answer),
                          "score": score if score is not None else 0.0})

        best = max(cands, key=lambda c: c["score"])
        yw_correct += best["correct"]

        corrects = [c for c in cands if c["correct"]]
        wrongs = [c for c in cands if not c["correct"]]
        if corrects and wrongs:
            valid_pairs += 1
            best_correct = max(c["score"] for c in corrects)
            best_wrong = max(c["score"] for c in wrongs)
            if best_correct > best_wrong:
                judge_ranks_right += 1

        print(f"  [{idx + 1:>2}/{len(problems)}] gold={prob.answer:g}  "
              f"chosen_correct={'Y' if best['correct'] else 'n'}  "
              f"#correct={len(corrects)}/{k}")

    total = len(problems)
    return {
        "n_problems": total,
        "k": k,
        "chosen_best_accuracy": yw_correct / total,
        "judge_ranking_agreement": (judge_ranks_right / valid_pairs
                                    if valid_pairs else float("nan")),
        "valid_pairs": valid_pairs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=4, help="candidates per problem")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    require_ready(args.model)

    print("\n=== Experiment 5: self-rewarding / LLM-as-judge ===")
    print(f"model={args.model}  K={args.k}  "
          f"problems={min(args.limit, len(PROBLEMS))}\n")

    res = run(args.model, args.k, args.limit, args.temperature)

    print("\n--- Results ---")
    print(f"  chosen-best accuracy (y_w correct) : {res['chosen_best_accuracy']:.3f}")
    print(f"  judge ranking agreement vs truth   : {res['judge_ranking_agreement']:.3f}"
          f"  (on {res['valid_pairs']} mixed problems)")

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "exp5_self_rewarding.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "n_problems", "k", "model"])
        w.writerow(["chosen_best_accuracy", res["chosen_best_accuracy"],
                    res["n_problems"], res["k"], args.model])
        w.writerow(["judge_ranking_agreement", res["judge_ranking_agreement"],
                    res["valid_pairs"], res["k"], args.model])
    print(f"\nSaved: {csv_path}")
    print("Interpretation: high judge agreement => the LLM-as-judge yields useful "
          "(y_w, y_l) preference pairs, supporting the self-rewarding loop.")


if __name__ == "__main__":
    main()
