"""A tractable token-level toy environment for preference optimization.

Why a token-level model?
------------------------
To reproduce the paper's claims about *length bias* (Section 5: R-DPO, SimPO) we
need sequence log-probabilities that genuinely scale with length. We therefore
use a unigram policy over a tiny vocabulary:

    log pi_theta(y | x) = sum_t log softmax(theta)[y_t]

Because the policy is unigram, a response is fully described by its per-token
*count vector* c in R^V, and

    log pi_theta(y) = c . log_softmax(theta).

This makes every sequence log-prob an exact, fully-vectorized dot product while
preserving the property that longer sequences accumulate more (more negative)
log-probability -- the mechanism behind DPO's length bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ToyData:
    counts: torch.Tensor          # (R, V) per-response token counts
    lengths: torch.Tensor         # (R,)   token counts |y|
    true_reward: torch.Tensor     # (R,)   length-independent quality
    theta_ref: torch.Tensor       # (V,)   fixed reference policy logits
    quality: torch.Tensor         # (V,)   per-token quality weights
    vocab_size: int


def make_responses(rng: np.random.Generator, *, n_responses: int, vocab_size: int,
                   min_len: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample random token sequences as count vectors with varied lengths."""
    counts = np.zeros((n_responses, vocab_size), dtype=np.float64)
    lengths = rng.integers(min_len, max_len + 1, size=n_responses)
    for i, L in enumerate(lengths):
        toks = rng.integers(0, vocab_size, size=L)
        for t in toks:
            counts[i, t] += 1.0
    return counts, lengths.astype(np.float64)


def build_toy_env(*, seed: int = 0, n_responses: int = 400, vocab_size: int = 8,
                  min_len: int = 4, max_len: int = 40) -> ToyData:
    """Construct responses, a reference policy, and length-independent rewards."""
    rng = np.random.default_rng(seed)

    counts, lengths = make_responses(
        rng, n_responses=n_responses, vocab_size=vocab_size,
        min_len=min_len, max_len=max_len,
    )

    # Per-token quality: some tokens are "good", some "bad". Reward is the MEAN
    # quality per token, so it is deliberately independent of response length.
    quality = rng.normal(0.0, 1.0, size=vocab_size)
    token_freq = counts / lengths[:, None]
    true_reward = token_freq @ quality  # (R,), length-independent by construction

    theta_ref = torch.tensor(rng.normal(0.0, 0.3, size=vocab_size), dtype=torch.float32)

    return ToyData(
        counts=torch.tensor(counts, dtype=torch.float32),
        lengths=torch.tensor(lengths, dtype=torch.float32),
        true_reward=torch.tensor(true_reward, dtype=torch.float32),
        theta_ref=theta_ref,
        quality=torch.tensor(quality, dtype=torch.float32),
        vocab_size=vocab_size,
    )


def sequence_logprobs(theta: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """log pi_theta(y) for every response = counts @ log_softmax(theta)."""
    return counts @ torch.log_softmax(theta, dim=-1)


def sample_preference_pairs(data: ToyData, *, seed: int = 0, n_pairs: int = 4000,
                            bt_temp: float = 0.5, length_bias: float = 0.0
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate Bradley-Terry preference pairs.

    length_bias > 0 injects an annotator preference for *longer* responses,
    independent of true quality -- the confound R-DPO/SimPO are designed to
    correct. Returns index tensors (winner_idx, loser_idx).
    """
    rng = np.random.default_rng(seed)
    R = data.true_reward.shape[0]
    r = data.true_reward.numpy()
    L = data.lengths.numpy()
    Ln = (L - L.mean()) / (L.std() + 1e-8)

    i = rng.integers(0, R, size=n_pairs)
    j = rng.integers(0, R, size=n_pairs)
    same = i == j
    j[same] = (j[same] + 1) % R

    logit = (r[i] - r[j]) / bt_temp + length_bias * (Ln[i] - Ln[j])
    p_i_wins = 1.0 / (1.0 + np.exp(-logit))
    i_wins = rng.random(n_pairs) < p_i_wins

    winner = np.where(i_wins, i, j)
    loser = np.where(i_wins, j, i)
    return torch.tensor(winner, dtype=torch.long), torch.tensor(loser, dtype=torch.long)


def ranking_accuracy(implicit_reward: torch.Tensor, true_reward: torch.Tensor,
                     *, seed: int = 123, n_pairs: int = 5000) -> float:
    """Fraction of random response pairs ordered correctly by implicit reward."""
    rng = np.random.default_rng(seed)
    R = true_reward.shape[0]
    i = rng.integers(0, R, size=n_pairs)
    j = rng.integers(0, R, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    pred = implicit_reward[i] > implicit_reward[j]
    gold = true_reward[i] > true_reward[j]
    return (pred == gold).float().mean().item()


def length_reward_correlation(implicit_reward: torch.Tensor,
                              lengths: torch.Tensor) -> float:
    """Spearman-style rank correlation between length and implicit reward.

    A large positive value indicates length bias: the objective rewards longer
    responses regardless of quality.
    """
    def rank(x: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(torch.argsort(x))
        return order.float()

    a = rank(implicit_reward)
    b = rank(lengths)
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()
