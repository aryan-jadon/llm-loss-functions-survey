"""Experiment 8 -- Token-level credit assignment: DPO vs. TDPO (mechanism).

Claim under test (Section 5: TDPO).
Standard DPO assigns a *single sequence-level* implicit reward, so every token in
a pair receives the **same** gradient weight sigma(-beta*h) regardless of whether
that token is already well-separated by the reference. TDPO reformulates
preference optimization at the token level so that gradient concentrates on the
tokens that are genuinely discriminative and still need to move, rather than
being spread uniformly (the survey's motivation: "DPO effectively assigns the
full contrastive signal to every token equally").

Scope / honesty note.
The canonical TDPO (Zeng et al., 2024) uses a token-level MDP with a sequential
forward-KL term whose exact bookkeeping is easy to misstate from a single
equation. To avoid shipping a possibly-incorrect reproduction, this experiment
demonstrates the *mechanism* with a transparent, self-consistent token-factorized
credit assignment, clearly labeled as such. It validates the **direction** of
TDPO's claim (non-uniform, discriminative-token-focused credit), not the exact
TDPO2 update. This mirrors how `kto_loss` is a simplified comparison form.

Toy.
A sequence is a set of T independent token slots. At slot i the policy has a
single logit theta_i favoring the winner token over the loser token (loser logit
fixed at 0), so the policy's per-token winner preference is theta_i and the
sequence log-ratio margin is h = sum_i (theta_i - theta_ref_i). Half the slots
are ALREADY-SEPARATED by the reference (theta_ref_i large), half are NEEDS-WORK
(theta_ref_i ~ 0). We start the policy at the reference and measure, per token,
the gradient magnitude each objective applies.

  * DPO weight per token:        beta * sigma(-beta * h)        (uniform)
  * TDPO-style weight per token:  beta * sigma(-beta * theta_i)  (per-token)

Usage:
    python exp8_tdpo_token_credit.py
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

BETA = 1.0
SEED = 0


def build_tokens(n_sep: int = 8, n_work: int = 8, *, sep_level: float = 2.5
                 ) -> tuple[torch.Tensor, np.ndarray]:
    """Reference per-token winner-preference logits and a needs-work mask.

    Already-separated tokens have a large positive reference logit (the reference
    already strongly prefers the winner token); needs-work tokens have ~0.
    """
    theta_ref = torch.tensor(
        [sep_level] * n_sep + [0.0] * n_work, dtype=torch.float32
    )
    needs_work = np.array([False] * n_sep + [True] * n_work)
    return theta_ref, needs_work


def per_token_grad_magnitudes(theta_ref: torch.Tensor, *, method: str
                              ) -> np.ndarray:
    """|dL/dtheta_i| at theta = theta_ref for DPO or the TDPO-style objective."""
    theta = theta_ref.clone().detach().requires_grad_(True)
    # Per-token policy winner-preference is theta_i; sequence margin h sums them
    # relative to the reference.
    h = (theta - theta_ref).sum()

    if method == "DPO":
        # One sequence-level loss -> identical weight on every token.
        loss = -F.logsigmoid(BETA * h)
    elif method == "TDPO":
        # Token-factorized credit: each token carries its own margin beta*theta_i,
        # so already-separated tokens (large theta_i) get a vanishing weight while
        # needs-work tokens (theta_i ~ 0) keep a large weight.
        loss = -F.logsigmoid(BETA * theta).sum()
    else:
        raise ValueError(method)

    (g,) = torch.autograd.grad(loss, theta)
    return g.abs().detach().numpy()


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(SEED)
    theta_ref, needs_work = build_tokens()
    sep = ~needs_work

    print("\n=== Experiment 8: token-level credit (DPO vs TDPO mechanism) ===")
    print(f"{'method':>7}{'grad@sep':>12}{'grad@work':>12}"
          f"{'work/sep':>10}{'%mass@work':>12}")

    rows = []
    for method in ("DPO", "TDPO"):
        g = per_token_grad_magnitudes(theta_ref, method=method)
        g_sep = float(g[sep].mean())
        g_work = float(g[needs_work].mean())
        ratio = g_work / (g_sep + 1e-12)
        mass_work = float(g[needs_work].sum() / (g.sum() + 1e-12))
        rows.append((method, g_sep, g_work, ratio, mass_work))
        print(f"{method:>7}{g_sep:>12.4f}{g_work:>12.4f}"
              f"{ratio:>10.2f}{mass_work * 100:>11.1f}%")

    csv_path = os.path.join(out_dir, "exp8_tdpo_token_credit.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "grad_separated", "grad_needswork",
                     "ratio_work_over_sep", "frac_grad_mass_on_needswork"])
        for method, gs, gw, ratio, mass in rows:
            wr.writerow([method, f"{gs:.6f}", f"{gw:.6f}", f"{ratio:.4f}",
                         f"{mass:.4f}"])

    # Optional plot: per-token gradient magnitude profile.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(theta_ref))
        plt.figure(figsize=(6.4, 4))
        for method, color in (("DPO", "#1f77b4"), ("TDPO", "#d62728")):
            g = per_token_grad_magnitudes(theta_ref, method=method)
            plt.plot(x, g, marker="o", label=method, color=color)
        nsep = int(sep.sum())
        plt.axvspan(-0.5, nsep - 0.5, color="green", alpha=0.06)
        plt.axvspan(nsep - 0.5, len(x) - 0.5, color="orange", alpha=0.08)
        plt.text(nsep / 2 - 0.5, plt.ylim()[1] * 0.95, "already-separated",
                 ha="center", va="top", fontsize=8)
        plt.text(nsep + (len(x) - nsep) / 2 - 0.5, plt.ylim()[1] * 0.95,
                 "needs-work", ha="center", va="top", fontsize=8)
        plt.xlabel("token slot")
        plt.ylabel("|gradient| applied to token")
        plt.title("DPO spreads credit uniformly; TDPO concentrates it on\n"
                  "discriminative (needs-work) tokens")
        plt.legend()
        plt.tight_layout()
        png = os.path.join(out_dir, "exp8_tdpo_token_credit.png")
        plt.savefig(png, dpi=130)
        print(f"\nSaved plot: {png}")
    except Exception as exc:  # matplotlib optional
        print(f"\n(matplotlib unavailable, skipped plot: {exc})")

    print(f"Saved: {csv_path}")
    print("Observation: DPO applies an identical per-token gradient weight "
          "(work/sep ratio ~ 1, ~50% of mass on needs-work tokens by count); the "
          "TDPO-style token-factorized credit puts most gradient mass on the "
          "needs-work tokens and almost none on already-separated ones.")


if __name__ == "__main__":
    main()
