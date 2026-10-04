#!/usr/bin/env python3
"""Table 2 / Fig. 5 with a fourth arm: MF-IA (input-augmented multi-fidelity baseline).

MF-IA = the Arm 3 script with the residual target switched off: the low-fidelity response is
an input and the network predicts the high-fidelity response directly (Guo et al. 2022;
Conti et al. 2023). Same channels, backbone, loss, epochs, seeds [42,0,1,2,3] and splits.

Arms 1-3 are Table 2 verbatim. MF-IA is read from mfia_per_seed.csv (written by the last
cell of benchmark_mfia_colab.ipynb on Colab, T4). The same file carries the Colab re-run of
Arm 3 on the same seeds, which is the like-for-like comparison: Table 2's Arm 3 was run on
different hardware for some systems.

    python3 make_table2_mfia.py mfia_per_seed.csv  ->  arms_table2_mfia.{pdf,png}, table2_mfia.md
"""
from __future__ import annotations
import csv, statistics as st, sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

N_SEEDS = 5
KEYS    = ["sdof", "frame3", "turbine", "tunnel"]
SYSTEMS = ["SDOF oscillator", "3-storey steel frame", "Wind turbine", "Buried tunnel"]
ARMS = [
    ("Arm 1 (low-fidelity)",  [+0.7234, +0.7516, +0.2730, -0.0689],
                              [0.0682, 0.0441, 0.0312, 0.0594]),
    ("Arm 2 (data-driven)",   [+0.3014, +0.7065, +0.3061, +0.7292],
                              [0.0460, 0.0351, 0.1310, 0.0768]),
    ("Arm 3 (multi-fidelity)", [+0.9123, +0.8363, +0.8631, +0.8704],
                              [0.0259, 0.0351, 0.0168, 0.0648]),
]
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "mfia_per_seed.csv")))
def col(ex, k):
    return [float(r[k]) for r in rows if r["system"] == ex and r[k] not in ("", "None")]
def paired(ex):
    rr = [r for r in rows if r["system"] == ex and r["arm3"] not in ("", "None") and r["mfia"] not in ("", "None")]
    return [float(r["mfia"]) for r in rr], [float(r["arm3"]) for r in rr]
mf  = [col(ex, "mfia") for ex in KEYS]
a3c = [col(ex, "arm3") for ex in KEYS]                 # Colab re-run of Arm 3, same seeds
for ex, v in zip(KEYS, mf):
    assert len(v) == N_SEEDS, f"{ex}: {len(v)}/{N_SEEDS} MF-IA seeds"
ARMS.append(("MF-IA (input-augmented)", [st.mean(v) for v in mf], [st.stdev(v) for v in mf]))

# palette shared with benchmark_systems4.tex and ablation_table5.py
INK, STEEL, TEAL, AMBER = "#1F2A37", "#5B6B7C", "#0D7D8A", "#A85C11"
GREYBG, TEALBG = "#EEF1F4", "#D8ECEE"
PURPLE = "#6B4C9A"
FACE = [STEEL, AMBER, TEAL, PURPLE]
ALPHA = [0.45, 0.85, 1.0, 0.85]                        # Arm 3 is the contribution

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

fig, ax = plt.subplots(figsize=(7.09, 3.9))
fig.subplots_adjust(left=0.155, right=0.975, top=0.855, bottom=0.135)

ypos = np.arange(len(SYSTEMS))[::-1]                   # first table row at the top
H = 0.21

for x in np.arange(0.2, 1.01, 0.2):                    # unobtrusive reading guides
    ax.axvline(x, color=STEEL, lw=0.4, alpha=0.25, zorder=1)

for k, (label, mean, sd) in enumerate(ARMS):
    off = H * (1.5 - k)
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
ax.set_ylim(-0.6, len(SYSTEMS) - 0.35)
ax.set_xticks(np.arange(-0.2, 1.01, 0.2))
ax.set_xlabel(r"Coefficient of determination $R^2$ on the evaluation split"
              "  (higher is better)")
ax.set_yticks(ypos)
ax.set_yticklabels(SYSTEMS)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4,
          frameon=False, handlelength=1.5, handleheight=0.9, columnspacing=1.6,
          handletextpad=0.6)

for ext in ("pdf", "png"):
    fig.savefig(f"arms_table2_mfia.{ext}", dpi=600, bbox_inches="tight", facecolor="white")

# ---------------------------------------------------------------- table + paired check
fmt = lambda m, s: f"{m:+.4f} ± {s:.4f}".replace("-", "−")
L = ["| Structure | Arm 1 (low-fidelity) | Arm 2 (data-driven) | Arm 3 (multi-fidelity) | MF-IA (input-augmented) |",
     "|---|---|---|---|---|"]
for i, s in enumerate(SYSTEMS):
    L.append(f"| {s} | " + " | ".join(fmt(ARMS[k][1][i], ARMS[k][2][i]) for k in range(4)) + " |")
L += ["", "Paired by seed against the Colab re-run of Arm 3 (same seeds, splits, hardware):", "",
      "| Structure | Arm 3 (Colab re-run) | MF-IA | MF-IA − Arm 3 | seeds MF-IA < Arm 3 |", "|---|---|---|---|---|"]
for i, s in enumerate(SYSTEMS):
    pm, pa = paired(KEYS[i]); d = [m - a for m, a in zip(pm, pa)]
    if len(d) < 2:
        L.append(f"| {s} | pending | | | |"); continue
    L.append(f"| {s} (n={len(d)}) | {fmt(st.mean(pa), st.stdev(pa))} | {fmt(st.mean(pm), st.stdev(pm))} | "
             f"{fmt(st.mean(d), st.stdev(d))} | {sum(x < 0 for x in d)}/{len(d)} |")
open("table2_mfia.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
