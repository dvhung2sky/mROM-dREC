#!/usr/bin/env python3
"""Figure for Table 2 (three-arm benchmark) — grouped bars with a consistency band.

Two things in Table 2 are hard to see in the numbers and easy to see in a chart:

  1. Arm 3 lands in a narrow band (0.836-0.912) on all four systems, while the two
     baselines swing over 0.82 and 0.43 in R2 respectively. The shaded band makes that
     invariance the visual subject of the figure.
  2. Neither baseline is reliably the runner-up: Arm 1 beats Arm 2 on the oscillator and
     the frame, Arm 2 beats Arm 1 on the turbine and the tunnel. The ordering inside each
     group flips, which is the argument for needing the corrector at all.

Bars start at zero and the tunnel Arm 1 bar is genuinely negative (worse than predicting
the split mean), so the zero line is drawn explicitly rather than hidden at the axis edge.

    python3 arms_table2.py     ->  arms_table2.pdf / .png
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- Table 2, verbatim
N_SEEDS = 5
SYSTEMS = ["SDOF oscillator", "3-storey steel frame", "Wind turbine", "Buried tunnel"]
ARMS = [
    ("Arm 1 (low-fidelity)",  [+0.7234, +0.7516, +0.2730, -0.0689],
                              [0.0682, 0.0441, 0.0312, 0.0594]),
    ("Arm 2 (data-driven)",   [+0.3014, +0.7065, +0.3061, +0.7292],
                              [0.0460, 0.0351, 0.1310, 0.0768]),
    ("Arm 3 (multi-fidelity)", [+0.9123, +0.8363, +0.8631, +0.8704],
                              [0.0259, 0.0351, 0.0168, 0.0648]),
]

# palette shared with benchmark_systems4.tex and ablation_table5.py
INK, STEEL, TEAL, AMBER = "#1F2A37", "#5B6B7C", "#0D7D8A", "#A85C11"
GREYBG, TEALBG = "#EEF1F4", "#D8ECEE"
FACE = [STEEL, AMBER, TEAL]
ALPHA = [0.45, 0.85, 1.0]                              # Arm 3 is the contribution

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "axes.edgecolor": STEEL, "axes.labelcolor": INK,
    "xtick.color": STEEL, "ytick.color": STEEL,
    "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.09, 3.15))
fig.subplots_adjust(left=0.155, right=0.975, top=0.855, bottom=0.135)

ypos = np.arange(len(SYSTEMS))[::-1]                   # first table row at the top
H = 0.25

for x in np.arange(0.2, 1.01, 0.2):                    # unobtrusive reading guides
    ax.axvline(x, color=STEEL, lw=0.4, alpha=0.25, zorder=1)

for k, (label, mean, sd) in enumerate(ARMS):
    off = H * (1 - k)
    ax.barh(ypos + off, mean, height=H * 0.90, color=FACE[k], alpha=ALPHA[k],
            edgecolor=INK, linewidth=0.35, zorder=3, label=label)
    ax.errorbar(mean, ypos + off, xerr=sd, fmt="none", ecolor=INK, elinewidth=0.7,
                capsize=1.8, capthick=0.7, zorder=4)
    for y, m, s in zip(ypos, mean, sd):
        right = m >= 0
        ax.annotate(f"{m:+.3f}".replace("-", "\u2212"),
                    (m + (s if right else -s), y + off),
                    xytext=(4 if right else -4, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center",
                    fontsize=6.8, color=INK,
                    fontweight="bold" if k == 2 else "normal")

ax.axvline(0.0, color=INK, lw=0.8, zorder=5)

ax.set_xlim(-0.20, 1.06)
ax.set_ylim(-0.55, len(SYSTEMS) - 0.42)
ax.set_xticks(np.arange(-0.2, 1.01, 0.2))
ax.set_xlabel(r"Coefficient of determination $R^2$ on the evaluation split"
              "  (higher is better)")
ax.set_yticks(ypos)
ax.set_yticklabels(SYSTEMS)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.135), ncol=3,
          frameon=False, handlelength=1.5, handleheight=0.9, columnspacing=2.4,
          handletextpad=0.6)

for ext in ("pdf", "png"):
    fig.savefig(f"arms_table2.{ext}", dpi=600, bbox_inches="tight", facecolor="white")

# ---------------------------------------------------------------- self-check
spread = {lab: max(m) - min(m) for lab, m, _ in ARMS}
print(f"{'arm':<26}{'min':>9}{'max':>9}{'across-system spread':>23}")
for lab, m, _ in ARMS:
    print(f"{lab:<26}{min(m):+9.4f}{max(m):+9.4f}{spread[lab]:23.4f}")
print(f"\nArm 3 spread is {spread['Arm 1 (low-fidelity)'] / spread['Arm 3 (multi-fidelity)']:.1f}x "
      f"narrower than Arm 1 and "
      f"{spread['Arm 2 (data-driven)'] / spread['Arm 3 (multi-fidelity)']:.1f}x narrower than Arm 2")

print("\nrunner-up (the better of the two baselines) per system:")
for i, s in enumerate(SYSTEMS):
    a1, a2, a3v = (ARMS[0][1][i], ARMS[1][1][i], ARMS[2][1][i])
    best = "Arm 1" if a1 > a2 else "Arm 2"
    print(f"  {s:<24}{best}   Arm 3 gain over it = {a3v - max(a1, a2):+.4f}")
assert all(ARMS[2][1][i] > max(ARMS[0][1][i], ARMS[1][1][i]) for i in range(len(SYSTEMS))), \
    "Arm 3 no longer wins on every system"
assert (ARMS[0][1][0] > ARMS[1][1][0]) != (ARMS[0][1][3] > ARMS[1][1][3]), \
    "the baseline ordering no longer flips between systems"
print("\nArm 3 wins on all four systems; the baseline ordering flips between them.")
