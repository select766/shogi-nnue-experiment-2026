#!/usr/bin/env python3
"""
Standalone script: validation loss by game ply (Expert Blending vs Baseline NNUE).

Data source:
  Expert Blending (DNN backbone, checkpoint 400)  vs  Baseline NNUE (83000.ckpt)
  Evaluated on 1M positions from dataset/split_v1_paired_uniform_50/val1
  Binned by nnue_ply, bin_width=10, min_count=10

Output: loss_per_gameply_nnue_ply.png  (same directory as this script)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data (nnue_ply bin center, Expert Blending loss, Baseline NNUE loss, count)
# ---------------------------------------------------------------------------
DATA = [
    #  ply,   EB_loss,   BL_loss,  count
    (   7,  0.000480,  0.000556,  14028),
    (  17,  0.001267,  0.001415,  39585),
    (  27,  0.003674,  0.003943,  64871),
    (  37,  0.009563,  0.010059,  81556),
    (  47,  0.019526,  0.020381,  81415),
    (  57,  0.029697,  0.030928,  81378),
    (  67,  0.038288,  0.039767,  81490),
    (  77,  0.044960,  0.046814,  80431),
    (  87,  0.050430,  0.052418,  78999),
    (  97,  0.052447,  0.054712,  75942),
    ( 107,  0.054059,  0.056210,  69571),
    ( 117,  0.053659,  0.055827,  61200),
    ( 127,  0.052303,  0.054125,  50152),
    ( 137,  0.049084,  0.051454,  39236),
    ( 147,  0.047224,  0.048879,  29447),
]

ply    = np.array([d[0] for d in DATA])
eb     = np.array([d[1] for d in DATA])
bl     = np.array([d[2] for d in DATA])
counts = np.array([d[3] for d in DATA])

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
})

C_EB = "#2166ac"   # blue
C_BL = "#d6604d"   # orange-red

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(5.5, 4.2), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

ax1.plot(ply, eb, "o-",  color=C_EB, label="Expert Blending (DNN backbone)")
ax1.plot(ply, bl, "s--", color=C_BL, label="Baseline NNUE")

ax1.set_ylabel("Mean Validation Loss")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3, linestyle=":")
ax1.set_xlim(0, 155)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))

ax2.bar(ply, counts / 1000, width=8, color="#888888", alpha=0.6)
ax2.set_xlabel("Game Ply")
ax2.set_ylabel("Count (k)")
ax2.grid(True, alpha=0.3, linestyle=":")

fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), "loss_per_gameply_nnue_ply.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
