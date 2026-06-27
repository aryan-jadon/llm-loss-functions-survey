"""Experiment 2 -- Controlled comparison of offline preference objectives.

This is the "anchoring experiment" recommended in the review: one fixed
reference, one fixed length-confounded dataset, identical training for every
objective. Because all methods see the same data and start from the same point,
the comparison is genuinely controlled (unlike the cross-paper tables in
Section 9, whose base models differ).

Claim under test (Section 5, R-DPO / SimPO): when preference data is
length-confounded (annotators prefer longer responses regardless of quality),
reference-based *sum* rewards (DPO, IPO, CPO, KTO, REBEL) absorb the length
signal and become length-biased -- which hurts true-quality ranking -- whereas
length-normalized objectives (SimPO, ORPO) are structurally unable to encode a
pure-length preference and stay quality-aligned.

Why a feature-based toy?
------------------------
A unigram softmax policy cannot raise the log-prob of *all* longer sequences
independently of which tokens they contain, so DPO's length bias never appears
(verified: the previous unigram version gave DPO a length-reward correlation of
~0.09). We therefore use a transparent, identifiable implicit-reward model with
two learnable scalars:

    logp_theta(y) = (C0 + delta + gamma * q(y)) * L(y)

where q(y) is the response's *per-token* quality (the length-independent ground
truth reward), L(y) is its length, C0 < 0 is a fixed base, `delta` is a pure
length coefficient, and `gamma` scales quality. The reference is delta=gamma=0,
so logp_ref(y) = C0 * L(y).

This makes the two effects identifiable and faithful:
  * DPO-style reward  r = beta*(logp_theta - logp_ref) = beta*(delta + gamma*q)*L
    -- a SUM, so it scales with length; fitting length-confounded data drives
    delta > 0 and the reward correlates with length.
  * SimPO reward      r = logp_theta / L = C0 + delta + gamma*q
    -- length-normalized, so `delta` is just a constant offset and the reward is
    affine in q alone: no pure-length term can be expressed.

Scope note: SLiC and R-DPO are intentionally excluded from this table. Their
benefits (SLiC's raw-log-prob length sensitivity; R-DPO's reduction of
*generated* length) require a generation-time setup; in a fixed-response toy the
R-DPO penalty is simply re-absorbed by `delta`, so it cannot be shown faithfully
here. Both remain implemented and tested in `preference_losses.py`.

Usage:
    python exp2_preference_objectives.py
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch

from preference_losses import PAIRWISE_LOSSES
from toy_env import length_reward_correlation, ranking_accuracy

BETA = 0.5
C0 = -3.0           # fixed negative base so per-token log-prob stays < 0 (ORPO-safe)
STEPS = 1000
LR = 0.05
LENGTH_FLAG = 0.25  # |length-reward corr| above this is flagged as length-biased

# Methods this static toy can faithfully compare (see module docstring).
METHODS = ["DPO", "IPO", "CPO", "KTO", "REBEL", "SimPO", "ORPO"]
# Length-normalized objectives use a /L reward, so a pure-length term cannot be
# expressed; REBEL regresses onto an external (here length-free) reward target,
# so it also avoids the confound -- both are length-robust for different reasons.
LENGTH_NORM = {"SimPO", "ORPO"}
# Pairwise sum-reward objectives that absorb the length confound (expected bias).
EXPECT_BIASED = {"DPO", "IPO", "CPO", "KTO"}
# Objectives expected to stay length-robust (normalization or length-free target).
EXPECT_ROBUST = {"SimPO", "ORPO", "REBEL"}


def build_data(*, seed: int = 0, n: int = 500, min_len: int = 4, max_len: int = 40):
    """Per-token quality q (true reward) and length L for each response."""
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 1.0, size=n)
    L = rng.integers(min_len, max_len + 1, size=n).astype(np.float64)
    return (torch.tensor(q, dtype=torch.float32),
            torch.tensor(L, dtype=torch.float32))


def sample_pairs(q, L, *, seed: int = 1, n_pairs: int = 8000, bt_temp: float = 0.5,
                 length_bias: float = 1.5):
    """Bradley-Terry pairs with an explicit, quality-independent length confound."""
    rng = np.random.default_rng(seed)
    qn, Ln = q.numpy(), L.numpy()
    R = qn.shape[0]
    Lz = (Ln - Ln.mean()) / (Ln.std() + 1e-8)

    i = rng.integers(0, R, size=n_pairs)
    j = rng.integers(0, R, size=n_pairs)
    j[i == j] = (j[i == j] + 1) % R

    logit = (qn[i] - qn[j]) / bt_temp + length_bias * (Lz[i] - Lz[j])
    i_wins = rng.random(n_pairs) < 1.0 / (1.0 + np.exp(-logit))
    winner = np.where(i_wins, i, j)
    loser = np.where(i_wins, j, i)
    return torch.tensor(winner), torch.tensor(loser)


def logp(delta, gamma, q, L):
    """Implicit-reward model: logp_theta(y) = (C0 + delta + gamma*q) * L."""
    return (C0 + delta + gamma * q) * L


def implicit_reward(delta, gamma, q, L, method):
    """Each method's OWN reward, computed consistently from the same params."""
    lp = logp(delta, gamma, q, L)
    lp_ref = C0 * L
    if method in LENGTH_NORM:
        return lp / L                      # length-normalized reward
    return BETA * (lp - lp_ref)            # reference-based sum reward


def train_one(method, q, L, winner, loser, *, seed: int = 0):
    torch.manual_seed(seed)
    delta = torch.nn.Parameter(torch.zeros(()))
    gamma = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam([delta, gamma], lr=LR)
    loss_fn = PAIRWISE_LOSSES[method]

    lp_ref_all = (C0 * L).detach()
    reward_diff = (q[winner] - q[loser]).detach()   # length-free target for REBEL

    for _ in range(STEPS):
        opt.zero_grad()
        lp_all = logp(delta, gamma, q, L)
        loss = loss_fn(
            logp_w=lp_all[winner], logp_l=lp_all[loser],
            logp_ref_w=lp_ref_all[winner], logp_ref_l=lp_ref_all[loser],
            len_w=L[winner], len_l=L[loser],
            reward_diff=reward_diff, beta=BETA, tau=0.5,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta, gamma], max_norm=5.0)
        opt.step()

    with torch.no_grad():
        r_hat = implicit_reward(delta, gamma, q, L, method)
        acc = ranking_accuracy(r_hat, q)
        len_corr = length_reward_correlation(r_hat, L)
    return acc, len_corr, delta.item(), gamma.item()


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    q, L = build_data(seed=0)
    winner, loser = sample_pairs(q, L, seed=1, length_bias=1.5)

    print("\n=== Experiment 2: controlled offline preference objectives ===")
    print("(same reference + same length-confounded dataset for every method)\n")
    print(f"{'Method':<8}{'RankAcc(q)':>12}{'Length corr':>14}"
          f"{'delta':>9}{'gamma':>9}")

    rows = []
    for m in METHODS:
        acc, len_corr, delta, gamma = train_one(m, q, L, winner, loser)
        flag = "  <- length-biased" if abs(len_corr) > LENGTH_FLAG else ""
        print(f"{m:<8}{acc:>12.3f}{len_corr:>14.3f}{delta:>9.3f}{gamma:>9.3f}{flag}")
        rows.append({"method": m, "ranking_accuracy": round(acc, 4),
                     "length_reward_corr": round(len_corr, 4),
                     "delta_length_coef": round(delta, 4),
                     "gamma_quality_coef": round(gamma, 4),
                     "family": "length-biased" if m in EXPECT_BIASED
                     else "length-robust"})

    csv_path = os.path.join(out_dir, "exp2_preference_objectives.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "ranking_accuracy", "length_reward_corr",
                           "delta_length_coef", "gamma_quality_coef", "family"])
        writer.writeheader()
        writer.writerows(rows)

    # Optional plot if matplotlib is available.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4.3))
        styles = {"length-biased": ("#c44e52", "o", "sum-reward (length-biased)"),
                  "length-robust": ("#55a868", "s", "length-free / normalized")}
        seen = set()
        label_offsets = {"SimPO": (7, 5), "ORPO": (7, -11), "REBEL": (9, -2)}
        for r in rows:
            color, marker, label = styles[r["family"]]
            plt.scatter(r["length_reward_corr"], r["ranking_accuracy"],
                        c=color, marker=marker, s=90, edgecolor="black",
                        linewidth=0.6, zorder=3,
                        label=label if label not in seen else None)
            seen.add(label)
            plt.annotate(r["method"],
                         (r["length_reward_corr"], r["ranking_accuracy"]),
                         textcoords="offset points",
                         xytext=label_offsets.get(r["method"], (7, 4)),
                         fontsize=8)
        plt.axvline(0.0, color="gray", lw=0.6, ls="--")
        plt.xlabel("length--reward correlation  (0 = quality-aligned, "
                   "$+$ = length-biased)")
        plt.ylabel("ranking accuracy on true quality $q$")
        plt.title("Length bias vs. quality ranking across objectives")
        plt.legend(loc="lower left", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        png = os.path.join(out_dir, "exp2_preference_objectives.png")
        plt.savefig(png, dpi=130)
        print(f"Saved plot: {png}")
    except Exception as exc:  # matplotlib optional
        print(f"(matplotlib unavailable, skipped plot: {exc})")

    # Data-driven summary (no hard-coded conclusion).
    def mean_over(group, key):
        vals = [r[key] for r in rows if r["method"] in group]
        return float(np.mean(vals)) if vals else float("nan")

    biased_corr = mean_over(EXPECT_BIASED, "length_reward_corr")
    robust_corr = mean_over(EXPECT_ROBUST, "length_reward_corr")
    biased_acc = mean_over(EXPECT_BIASED, "ranking_accuracy")
    robust_acc = mean_over(EXPECT_ROBUST, "ranking_accuracy")

    print(f"\nSaved: {csv_path}")
    print("--- Summary (mean over family) ---")
    print(f"  length-biased  (DPO/IPO/CPO/KTO, sum reward): "
          f"length corr {biased_corr:+.3f}, RankAcc {biased_acc:.3f}")
    print(f"  length-robust  (SimPO/ORPO/REBEL)          : "
          f"length corr {robust_corr:+.3f}, RankAcc {robust_acc:.3f}")
    if biased_corr > robust_corr + 0.1 and robust_acc >= biased_acc:
        print("  => Reproduced: pairwise sum-reward methods absorb the length "
              "confound (higher length corr, lower quality ranking); "
              "length-normalized objectives (and REBEL's length-free target) "
              "stay quality-aligned. Matches Section 5 (SimPO).")
    else:
        print("  => Effect not clearly reproduced under current settings; "
              "inspect delta (length coefficient) and tuning before citing.")
    print("Notes: REBEL is length-robust here because it regresses onto a "
          "length-free reward target; SLiC and R-DPO are excluded from this "
          "static comparison (their benefits need a generation-time setup).")


if __name__ == "__main__":
    main()
