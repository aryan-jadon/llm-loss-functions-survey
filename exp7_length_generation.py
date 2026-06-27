"""Experiment 7 -- Generation-time length control: R-DPO and SLiC.

Why this experiment exists
--------------------------
The static comparison in Experiment 2 deliberately *excludes* R-DPO and SLiC,
because their benefits are about the length of *generated* text, which a
fixed-response toy cannot show (the R-DPO penalty is simply re-absorbed by the
reward parameter). This experiment closes that gap with a small **generative**
toy: a unigram policy with an explicit end-of-sequence (EOS) token, so the policy
actually *controls how long its own samples are* via the EOS probability.

Claim under test (Section 5: R-DPO / SimPO / SLiC).
When preference data is length-confounded (annotators prefer longer responses
regardless of quality), a reference-based *sum* reward (DPO) raises the relative
log-probability of long sequences by lowering the EOS probability -- so the
trained policy *generates longer text*. SLiC's raw-log-prob hinge is length-
sensitive for the same reason and inflates length too. SimPO normalizes the
reward by length, so a pure-length preference cannot be expressed and generated
length stays at the reference.

An honest negative result for R-DPO.
This toy *also* demonstrates *why* R-DPO cannot be isolated in a unigram setting,
confirming the caveat in Experiment 2. R-DPO subtracts alpha*(|y_w|-|y_l|) from
the margin, but on a fixed response pool those lengths are constants w.r.t. the
policy, so the penalty only *re-weights* pairs -- it does not change the gradient
direction. Because a unigram policy's only lever for separating length-
confounded pairs is the EOS probability (there is no length-independent feature
to shift toward), R-DPO tracks DPO and still inflates length: the penalty is
re-absorbed. R-DPO's real benefit requires a feature-rich, generation-time
setup, exactly as the survey states; we report it here as a documented
limitation rather than a flattering number.

We train each objective on the SAME length-confounded pairs from the SAME
reference policy, then GENERATE fresh samples and report the mean generated
length and mean (length-independent) quality.

This is a mechanistic demonstration on a toy model, not a benchmark.

Usage:
    python exp7_length_generation.py
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch

from preference_losses import dpo_loss, rdpo_loss, simpo_loss, slic_loss

# --- toy configuration -------------------------------------------------------
N_CONTENT = 6          # number of content tokens
EOS = N_CONTENT        # EOS is the last vocab index
VOCAB = N_CONTENT + 1
MAX_LEN = 200          # generation cap (kept well above observed lengths)
STEPS = 200
LR = 0.02
BETA = 0.5
ALPHA_RDPO = 0.05      # R-DPO length penalty (kept small to show re-absorption)
LENGTH_BIAS = 0.6      # strength of the pure-length annotator confound
SEED = 0

METHODS = ("DPO", "R-DPO", "SimPO", "SLiC")


def make_reference(rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed reference logits and per-content-token quality weights."""
    theta_ref = torch.tensor(rng.normal(0.0, 0.3, size=VOCAB), dtype=torch.float32)
    # Give EOS a moderate baseline so reference sequences have a sensible length.
    theta_ref[EOS] = 0.0
    quality = torch.tensor(rng.normal(0.0, 1.0, size=N_CONTENT), dtype=torch.float32)
    return theta_ref, quality


def sample_sequences(theta: torch.Tensor, *, n: int, rng: np.random.Generator
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Autoregressively sample n sequences from a unigram+EOS policy.

    Returns (counts, content_lengths): counts is (n, VOCAB) including the single
    EOS token per sequence; content_lengths excludes EOS.
    """
    probs = torch.softmax(theta, dim=-1).detach().numpy()
    counts = np.zeros((n, VOCAB), dtype=np.float64)
    lengths = np.zeros(n, dtype=np.float64)
    for i in range(n):
        L = 0
        while L < MAX_LEN:
            tok = rng.choice(VOCAB, p=probs)
            if tok == EOS:
                break
            counts[i, tok] += 1.0
            L += 1
        counts[i, EOS] += 1.0          # terminal EOS token is part of the sequence
        lengths[i] = L
    return counts, lengths


def seq_quality(counts: np.ndarray, lengths: np.ndarray, quality: torch.Tensor
                ) -> np.ndarray:
    """Length-independent quality = mean per-content-token quality weight."""
    q = quality.numpy()
    content = counts[:, :N_CONTENT]
    denom = np.maximum(lengths, 1.0)
    return (content @ q) / denom


def logp(theta: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Sequence log-prob under the unigram+EOS policy = counts @ log_softmax."""
    return counts @ torch.log_softmax(theta, dim=-1)


def build_pairs(qual: np.ndarray, lengths: np.ndarray, *, rng: np.random.Generator,
                n_pairs: int = 6000, bt_temp: float = 0.5, length_bias: float = LENGTH_BIAS
                ) -> tuple[np.ndarray, np.ndarray]:
    """Bradley-Terry pairs with an explicit, quality-independent length confound."""
    R = qual.shape[0]
    Lz = (lengths - lengths.mean()) / (lengths.std() + 1e-8)
    i = rng.integers(0, R, size=n_pairs)
    j = rng.integers(0, R, size=n_pairs)
    j[i == j] = (j[i == j] + 1) % R
    logit = (qual[i] - qual[j]) / bt_temp + length_bias * (Lz[i] - Lz[j])
    i_wins = rng.random(n_pairs) < 1.0 / (1.0 + np.exp(-logit))
    winner = np.where(i_wins, i, j)
    loser = np.where(i_wins, j, i)
    return winner, loser


def train_one(method: str, counts_t: torch.Tensor, lengths_t: torch.Tensor,
              theta_ref: torch.Tensor, winner: np.ndarray, loser: np.ndarray
              ) -> torch.Tensor:
    """Train a fresh policy from the reference init under one objective."""
    torch.manual_seed(SEED)
    theta = theta_ref.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=LR)
    w = torch.tensor(winner, dtype=torch.long)
    l = torch.tensor(loser, dtype=torch.long)

    logp_ref_all = logp(theta_ref, counts_t).detach()
    len_w = lengths_t[w].clamp(min=1.0)
    len_l = lengths_t[l].clamp(min=1.0)

    for _ in range(STEPS):
        opt.zero_grad()
        logp_all = logp(theta, counts_t)
        lw, ll = logp_all[w], logp_all[l]
        lrw, lrl = logp_ref_all[w], logp_ref_all[l]
        if method == "DPO":
            loss = dpo_loss(lw, lrw, ll, lrl, beta=BETA)
        elif method == "R-DPO":
            loss = rdpo_loss(lw, lrw, ll, lrl, len_w, len_l, beta=BETA, alpha=ALPHA_RDPO)
        elif method == "SimPO":
            loss = simpo_loss(lw, ll, len_w, len_l, beta=2.0, gamma=0.5)
        elif method == "SLiC":
            loss = slic_loss(lw, ll, delta=1.0, lam=0.1)
        else:
            raise ValueError(method)
        loss.backward()
        opt.step()
    return theta.detach()


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    theta_ref, quality = make_reference(rng)

    # Fixed pool of reference samples that the preference pairs are drawn from.
    counts, lengths = sample_sequences(theta_ref, n=600, rng=rng)
    qual = seq_quality(counts, lengths, quality)
    counts_t = torch.tensor(counts, dtype=torch.float32)
    lengths_t = torch.tensor(lengths, dtype=torch.float32)

    winner, loser = build_pairs(qual, lengths, rng=rng)

    # Reference generation statistics (the baseline the policies start from).
    ref_counts, ref_len = sample_sequences(theta_ref, n=4000, rng=rng)
    ref_qual = seq_quality(ref_counts, ref_len, quality)
    ref_mean_len = float(ref_len.mean())
    ref_mean_qual = float(ref_qual.mean())

    print("\n=== Experiment 7: generation-time length control (R-DPO / SLiC) ===")
    print(f"reference policy: mean gen length={ref_mean_len:.2f}  "
          f"mean quality={ref_mean_qual:+.3f}\n")
    print(f"{'method':>7}{'gen_len':>10}{'len_vs_ref':>12}{'quality':>10}")

    rows = []
    for method in METHODS:
        theta = train_one(method, counts_t, lengths_t, theta_ref, winner, loser)
        gcounts, glen = sample_sequences(theta, n=4000, rng=np.random.default_rng(SEED + 7))
        gqual = seq_quality(gcounts, glen, quality)
        mean_len = float(glen.mean())
        mean_qual = float(gqual.mean())
        delta_len = mean_len - ref_mean_len
        rows.append((method, mean_len, delta_len, mean_qual))
        print(f"{method:>7}{mean_len:>10.2f}{delta_len:>+12.2f}{mean_qual:>+10.3f}")

    csv_path = os.path.join(out_dir, "exp7_length_generation.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "gen_length", "gen_length_minus_ref", "quality"])
        for method, mlen, dlen, mq in rows:
            wr.writerow([method, f"{mlen:.4f}", f"{dlen:+.4f}", f"{mq:+.4f}"])
        wr.writerow(["REFERENCE", f"{ref_mean_len:.4f}", "+0.0000",
                     f"{ref_mean_qual:+.4f}"])

    # Optional plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r[0] for r in rows]
        dlens = [r[2] for r in rows]
        # Color by faithfully-demonstrable behavior; R-DPO is grayed out because
        # the unigram toy re-absorbs its penalty (see module docstring).
        palette = {"DPO": "#d62728", "SLiC": "#d62728",
                   "SimPO": "#2ca02c", "R-DPO": "#999999"}
        colors = [palette.get(n, "#1f77b4") for n in names]
        plt.figure(figsize=(6, 4))
        plt.bar(names, dlens, color=colors)
        plt.axhline(0.0, color="gray", lw=0.8)
        plt.ylabel("change in generated length vs. reference")
        plt.title("Length-confounded preferences: DPO/SLiC inflate generated\n"
                  "length; SimPO does not; R-DPO re-absorbed (gray, see note)")
        plt.tight_layout()
        png = os.path.join(out_dir, "exp7_length_generation.png")
        plt.savefig(png, dpi=130)
        print(f"\nSaved plot: {png}")
    except Exception as exc:  # matplotlib optional
        print(f"\n(matplotlib unavailable, skipped plot: {exc})")

    print(f"Saved: {csv_path}")
    print("Observation: under length-confounded preferences, the reference-based "
          "sum reward (DPO) and the raw-log-prob hinge (SLiC) lower the EOS "
          "probability and generate longer text; SimPO's length normalization "
          "keeps generated length near the reference. R-DPO tracks DPO here "
          "because a unigram policy's only separating lever is length, so its "
          "penalty is re-absorbed (documented limitation; needs a feature-rich "
          "setup, per the survey).")


if __name__ == "__main__":
    main()
