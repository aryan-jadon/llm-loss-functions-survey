"""Experiment 6 -- Chain-of-Thought distillation data generation.

Validates the data-generation half of CoT distillation (Section 7): a teacher
model produces step-by-step reasoning traces, and -- crucially -- the traces are
FILTERED to those that reach the correct final answer (Magister et al.), which
the paper notes yields higher-quality training signal than unfiltered traces.

Ollama is inference-only, so this script produces the *distillation dataset*
(prompt -> teacher rationale) that an SFT/CoT-KD step would consume. It does not
fine-tune a student; it quantifies how rationale filtering changes dataset
quality and size, which is the claim under test.

Output: a JSONL file of accepted (question, rationale, answer) examples plus a
summary of acceptance rate.

Usage:
    python exp6_cot_distillation.py --traces 3 --limit 15
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from math_problems import PROBLEMS
from ollama_utils import (
    DEFAULT_MODEL,
    extract_final_number,
    generate,
    is_correct,
    require_ready,
)

TEACHER_SYSTEM = (
    "You are an expert teacher creating training data. Produce a clear, "
    "step-by-step reasoning trace for the problem, then end with '#### <number>'."
)


def generate_traces(question: str, *, model: str, n_traces: int,
                    temperature: float) -> list[str]:
    out = []
    for s in range(n_traces):
        text = generate(question, model=model, temperature=temperature,
                        system=TEACHER_SYSTEM, num_predict=512, seed=3000 + s)
        out.append(text)
    return out


def run(model: str, n_traces: int, limit: int, temperature: float,
        jsonl_path: str) -> dict:
    problems = PROBLEMS[:limit]
    total_traces = 0
    accepted_traces = 0
    problems_with_any = 0

    with open(jsonl_path, "w") as fout:
        for idx, prob in enumerate(problems):
            traces = generate_traces(prob.question, model=model,
                                     n_traces=n_traces, temperature=temperature)
            kept_here = 0
            for tr in traces:
                total_traces += 1
                pred = extract_final_number(tr)
                if is_correct(pred, prob.answer):
                    accepted_traces += 1
                    kept_here += 1
                    fout.write(json.dumps({
                        "question": prob.question,
                        "rationale": tr.strip(),
                        "answer": prob.answer,
                    }) + "\n")
            problems_with_any += (kept_here > 0)
            print(f"  [{idx + 1:>2}/{len(problems)}] gold={prob.answer:g}  "
                  f"kept {kept_here}/{n_traces} traces")

    return {
        "n_problems": len(problems),
        "n_traces_per_problem": n_traces,
        "total_traces": total_traces,
        "accepted_traces": accepted_traces,
        "acceptance_rate": accepted_traces / total_traces if total_traces else 0.0,
        "coverage": problems_with_any / len(problems) if problems else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--traces", type=int, default=3, help="traces per problem")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    require_ready(args.model)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "exp6_cot_distillation_dataset.jsonl")

    print("\n=== Experiment 6: CoT distillation data generation ===")
    print(f"model={args.model}  traces/problem={args.traces}  "
          f"problems={min(args.limit, len(PROBLEMS))}\n")

    res = run(args.model, args.traces, args.limit, args.temperature, jsonl_path)

    print("\n--- Dataset summary (after correctness filtering) ---")
    print(f"  total traces generated : {res['total_traces']}")
    print(f"  accepted (correct)     : {res['accepted_traces']}")
    print(f"  acceptance rate        : {res['acceptance_rate']:.3f}")
    print(f"  problem coverage       : {res['coverage']:.3f}")
    print(f"  distillation dataset   : {jsonl_path}")

    csv_path = os.path.join(out_dir, "exp6_cot_distillation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in res.items():
            w.writerow([k, v])
    print(f"  summary                : {csv_path}")
    print("\nNote: filtering keeps only traces with the correct final answer, "
          "producing the higher-quality SFT signal described in Section 7.")


if __name__ == "__main__":
    main()
