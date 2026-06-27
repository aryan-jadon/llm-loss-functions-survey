"""Preference-optimization loss functions from the survey.

All pairwise objectives operate on *sequence-level* log-probabilities and are
written to mirror the equations in the paper (Section 5, Alignment Objectives).

Conventions
-----------
Each pairwise loss receives, as 1-D tensors over a batch of preference pairs:
    logp_w, logp_l         : log pi_theta(y | x)  for winner / loser
    logp_ref_w, logp_ref_l : log pi_ref(y | x)    for winner / loser
    len_w, len_l           : token counts |y_w|, |y_l|

The implicit reward used by DPO-family methods is
    r_hat(x, y) = beta * ( log pi_theta(y|x) - log pi_ref(y|x) ).

Every function returns a scalar mean loss (lower is better).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _logratio(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """log pi_theta(y|x) - log pi_ref(y|x)."""
    return logp - logp_ref


def dpo_loss(logp_w, logp_ref_w, logp_l, logp_ref_l, *, beta: float = 0.1, **_):
    """Direct Preference Optimization (Rafailov et al., 2023), Eq. (DPO)."""
    h = _logratio(logp_w, logp_ref_w) - _logratio(logp_l, logp_ref_l)
    return -F.logsigmoid(beta * h).mean()


def ipo_loss(logp_w, logp_ref_w, logp_l, logp_ref_l, *, tau: float = 0.1, **_):
    """Identity Preference Optimization (Azar et al., 2023).

    Squared loss (h - 1/(2 tau))^2 avoids DPO's log-sigmoid saturation.
    """
    h = _logratio(logp_w, logp_ref_w) - _logratio(logp_l, logp_ref_l)
    return ((h - 1.0 / (2.0 * tau)) ** 2).mean()


def simpo_loss(logp_w, logp_l, len_w, len_l, *, beta: float = 2.0, gamma: float = 0.5, **_):
    """SimPO (Meng et al., 2024): reference-free, length-normalized reward."""
    r_w = logp_w / len_w
    r_l = logp_l / len_l
    return -F.logsigmoid(beta * (r_w - r_l) - gamma).mean()


def orpo_loss(logp_w, logp_l, len_w, len_l, *, lam: float = 0.5, **_):
    """ORPO (Hong et al., 2024): SFT + odds-ratio penalty, reference-free."""
    # Length-normalized log-likelihood (log of the geometric mean of token probs).
    # Clamp into (-inf, ~0) so the implied probability stays strictly inside
    # (0, 1); this keeps the d/dp log(1-p) term in the odds ratio finite and
    # prevents gradient blow-up when the model pushes a sequence's score high.
    log_p_tilde_w = (logp_w / len_w).clamp(min=-30.0, max=-1e-4)
    log_p_tilde_l = (logp_l / len_l).clamp(min=-30.0, max=-1e-4)
    p_w = log_p_tilde_w.exp()
    p_l = log_p_tilde_l.exp()
    log_odds_w = torch.log(p_w) - torch.log1p(-p_w)
    log_odds_l = torch.log(p_l) - torch.log1p(-p_l)
    loss_or = -F.logsigmoid(log_odds_w - log_odds_l).mean()
    loss_sft = -(logp_w / len_w).mean()
    return loss_sft + lam * loss_or


def cpo_loss(logp_w, logp_ref_w, logp_l, logp_ref_l, len_w, *,
             beta: float = 0.1, lam_nll: float = 0.3, **_):
    """CPO (Xu et al., 2024): DPO + behavioral-cloning NLL anchor on the winner."""
    dpo = dpo_loss(logp_w, logp_ref_w, logp_l, logp_ref_l, beta=beta)
    nll = -(logp_w / len_w).mean()
    return dpo + lam_nll * nll


def slic_loss(logp_w, logp_l, *, delta: float = 1.0, lam: float = 0.1, **_):
    """SLiC (Zhao et al., 2022): margin hinge + SFT regularizer (no reference)."""
    hinge = F.relu(delta - (logp_w - logp_l)).mean()
    sft = -(logp_w).mean()
    return hinge + lam * sft


def rdpo_loss(logp_w, logp_ref_w, logp_l, logp_ref_l, len_w, len_l, *,
              beta: float = 0.1, alpha: float = 0.05, **_):
    """R-DPO (Park et al., 2024): DPO with an explicit length penalty."""
    h = _logratio(logp_w, logp_ref_w) - _logratio(logp_l, logp_ref_l)
    margin = beta * h - alpha * (len_w - len_l)
    return -F.logsigmoid(margin).mean()


def rebel_loss(logp_w, logp_ref_w, logp_l, logp_ref_l, reward_diff, *,
               beta: float = 0.1, **_):
    """REBEL (Gao et al., 2024): regress implicit reward gap onto a scalar reward gap."""
    h = _logratio(logp_w, logp_ref_w) - _logratio(logp_l, logp_ref_l)
    return ((beta * h - reward_diff) ** 2).mean()


def kto_loss(logp_w, logp_ref_w, logp_l, logp_ref_l, *,
             beta: float = 0.1, lam_d: float = 1.0, lam_u: float = 1.0, **_):
    """KTO (Ethayarajh et al., 2024), simplified to a paired desirable/undesirable form.

    The winner is treated as a desirable example, the loser as undesirable.
    z0 (the reference point) is approximated by the batch-mean implicit reward.
    """
    r_w = _logratio(logp_w, logp_ref_w)
    r_l = _logratio(logp_l, logp_ref_l)
    z0 = torch.cat([r_w, r_l]).mean().detach()
    v_w = lam_d * torch.sigmoid(beta * (r_w - z0))
    v_l = lam_u * torch.sigmoid(beta * (z0 - r_l))
    # Loss = E[lambda_y - v]; minimizing drives the sigmoid terms toward 1.
    return (lam_d - v_w).mean() + (lam_u - v_l).mean()


# Registry used by the experiment scripts.
PAIRWISE_LOSSES = {
    "DPO": dpo_loss,
    "IPO": ipo_loss,
    "SimPO": simpo_loss,
    "ORPO": orpo_loss,
    "CPO": cpo_loss,
    "SLiC": slic_loss,
    "R-DPO": rdpo_loss,
    "REBEL": rebel_loss,
    "KTO": kto_loss,
}
