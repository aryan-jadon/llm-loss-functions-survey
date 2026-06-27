"""Experiment 3 -- DPO log-sigmoid saturation vs. IPO squared loss.

Validates the claim in Section 5 (IPO): DPO's -log sigma(beta * h) provides a
gradient weight sigma(-beta * h) that vanishes once the implicit reward margin h
is large (already-correct pairs stop learning), whereas IPO's squared loss
(h - 1/(2 tau))^2 keeps a gradient that grows linearly in the margin error.

We sweep the margin h and plot |d loss / d h| for each objective.

Usage:
    python exp3_dpo_vs_ipo_saturation.py
"""

from __future__ import annotations

import csv
import os

import torch
import torch.nn.functional as F


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    beta = 1.0
    tau = 1.0
    h = torch.linspace(-2.0, 8.0, 101, requires_grad=True)

    # DPO gradient magnitude w.r.t. the margin h.
    dpo = -F.logsigmoid(beta * h)
    (g_dpo,) = torch.autograd.grad(dpo.sum(), h, create_graph=False)

    # IPO gradient magnitude w.r.t. the margin h.
    h2 = h.detach().clone().requires_grad_(True)
    ipo = (h2 - 1.0 / (2.0 * tau)) ** 2
    (g_ipo,) = torch.autograd.grad(ipo.sum(), h2)

    hv = h.detach().numpy()
    gd = g_dpo.abs().detach().numpy()
    gi = g_ipo.abs().detach().numpy()

    csv_path = os.path.join(out_dir, "exp3_dpo_vs_ipo_saturation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["margin_h", "dpo_grad_abs", "ipo_grad_abs"])
        for a, b, c in zip(hv, gd, gi):
            writer.writerow([f"{a:.4f}", f"{b:.6f}", f"{c:.6f}"])

    print("\n=== Experiment 3: DPO saturation vs IPO ===")
    print(f"{'margin h':>10}{'|dDPO/dh|':>14}{'|dIPO/dh|':>14}")
    for idx in [0, 25, 50, 60, 70, 80, 100]:
        print(f"{hv[idx]:>10.2f}{gd[idx]:>14.5f}{gi[idx]:>14.5f}")

    # Optional plot if matplotlib is available.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(hv, gd, label="DPO  |d/dh -log sigma(beta h)|")
        plt.plot(hv, gi, label="IPO  |d/dh (h - 1/2tau)^2|")
        plt.axvline(0.0, color="gray", lw=0.6, ls="--")
        plt.xlabel("implicit reward margin  h")
        plt.ylabel("gradient magnitude")
        plt.title("DPO saturates on easy pairs; IPO does not")
        plt.legend()
        plt.tight_layout()
        png = os.path.join(out_dir, "exp3_dpo_vs_ipo_saturation.png")
        plt.savefig(png, dpi=130)
        print(f"\nSaved plot: {png}")
    except Exception as exc:  # matplotlib optional
        print(f"\n(matplotlib unavailable, skipped plot: {exc})")

    print(f"Saved: {csv_path}")
    print("Observation: |dDPO/dh| -> 0 as h grows (vanishing gradient on "
          "already-correct pairs); |dIPO/dh| grows linearly.")


if __name__ == "__main__":
    main()
