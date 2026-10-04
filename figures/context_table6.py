#!/usr/bin/env python3
"""Figure for Table 6 (autoregressive context length L).

Plots the PAIRED change against the full context L = 100 on the same seed. The anchor
column of Table 6 (Arm 3 from Table 2) is deliberately NOT drawn: the sweep runs under
the common harness of Section 4.3, whose absolute level sits below the tuned per-example
configuration, so the paired columns are not offsets from the anchor and adding it as a
baseline would invite exactly that misreading.

L = 100 is the reference, so every curve passes through (100, 0) by construction — that
point is the definition of the y axis, not a measurement.

Panel (b) is the same data on a tight axis, because the tunnel's collapse at L = 0
(-0.1987) is 3.7x the next largest effect and 13x the next largest tunnel entry, and would
otherwise flatten the twelve entries the text describes as lying within about half a
standard deviation of zero.

    python3 context_table6.py     ->  context_table6.pdf / .png
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- Table 6, verbatim
L = np.array([0, 12, 25, 50, 100])
SIG = {"*": "$p<0.05$", "†": "$p<0.10$"}          # paired t-test over five seeds
ROWS = [                                           # ordered by |effect| at L = 0
    ("Buried tunnel",        [-0.1987, -0.0231, -0.0152, -0.0085, 0.0],
                             ["*", "*", "*", "", ""]),
    ("Wind turbine",         [-0.0537, -0.0004, +0.0025, +0.0113, 0.0],
                             ["*", "", "", "", ""]),
    ("SDOF oscillator",      [-0.0312, -0.0067, -0.0280, -0.0060, 0.0],
                             ["†", "", "", "", ""]),
    ("3-storey steel frame", [-0.0042, +0.0069, +0.0020, +0.0021, 0.0],
                             ["", "", "", "", ""]),
]

INK, STEEL, TEAL, AMBER = "#1F2A37", "#5B6B7C", "#0D7D8A", "#A85C11"
GREYBG = "#EEF1F4"
COL = [TEAL, AMBER, STEEL, INK]
MRK = ["o", "s", "^", "D"]
LS = ["-", "--", "-.", (0, (1, 1.6))]

ZOOM = (-0.036, 0.016)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": STEEL, "axes.labelcolor": INK,
    "xtick.color": STEEL, "ytick.color": STEEL,
    "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.09, 3.05))
fig.subplots_adjust(left=0.085, right=0.995, top=0.845, bottom=0.185, wspace=0.245)


def draw(ax, zoomed: bool) -> None:
    ax.axhline(0.0, color=INK, lw=0.8, ls=(0, (4, 2.5)), zorder=2)
    for k, (name, d, sig) in enumerate(ROWS):
        ax.plot(L, d, ls=LS[k], color=COL[k], lw=1.2, marker=MRK[k], ms=4.2,
                mec="white", mew=0.6, zorder=4 - 0.1 * k, label=name)
        for x, y, s in zip(L, d, sig):
            if not s:
                continue
            if zoomed and y < ZOOM[0]:
                continue
            ax.annotate(s, (x, y), xytext=(0, 8 if zoomed else (-11 if y < 0 else 8)),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=COL[k], zorder=6)
    ax.set_xticks(L)
    ax.set_xlim(-6, 106)
    ax.set_xlabel("Autoregressive context length $L$ (steps)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=STEEL, lw=0.4, alpha=0.22)
    ax.set_axisbelow(True)


draw(axA, zoomed=False)
axA.set_ylim(-0.215, 0.030)
axA.set_ylabel(r"Paired change in $R^2$ against $L=100$, $\Delta R^2$")
axA.add_patch(Rectangle((-6, ZOOM[0]), 112, ZOOM[1] - ZOOM[0], facecolor=GREYBG,
                        edgecolor=STEEL, lw=0.5, ls=(0, (3, 2)), zorder=0))
axA.annotate("panel (b)", (104, ZOOM[0]), xytext=(0, -9), textcoords="offset points",
             ha="right", va="top", fontsize=7, color=STEEL)
axA.annotate("no context", (0, -0.1987), xytext=(9, 2), textcoords="offset points",
             ha="left", va="center", fontsize=7, color=TEAL)

draw(axB, zoomed=True)
axB.set_ylim(*ZOOM)
axB.set_yticks(np.arange(-0.03, 0.0151, 0.01))
axB.set_ylabel(r"$\Delta R^2$  (detail)")
axB.set_facecolor(GREYBG)
off = [(n, x, y) for n, d, _ in ROWS for x, y in zip(L, d) if y < ZOOM[0]]
axB.text(0.02, 0.985,
         "off scale at $L=0$:  " + ",  ".join(f"{n.split()[-1]} {y:+.4f}".replace("-", "−")
                                              for n, _, y in off),
         transform=axB.transAxes, ha="left", va="top", fontsize=6.6, color=STEEL)

handles, labels = axA.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.545, 1.015), ncol=4,
           frameon=False, handlelength=2.2, columnspacing=1.9, handletextpad=0.55)
for ax, tag in ((axA, "(a)"), (axB, "(b)")):
    ax.text(-0.155, 1.09, tag, transform=ax.transAxes, fontsize=9,
            fontweight="bold", color=INK, va="top")
fig.text(0.995, 0.018, "† $p<0.10$   * $p<0.05$, paired $t$-test over five seeds",
         fontsize=6.8, color=STEEL, ha="right", va="bottom")

for ext in ("pdf", "png"):
    fig.savefig(f"context_table6.{ext}", dpi=600, bbox_inches="tight", facecolor="white")

# ---------------------------------------------------------------- self-check
print(f"{'system':<22}" + "".join(f"{f'L={x}':>11}" for x in L) + "   monotonic?")
for name, d, sig in ROWS:
    mono = all(b >= a for a, b in zip(d, d[1:]))
    print(f"{name:<22}" + "".join(f"{v:+11.4f}" for v in d) + f"   {mono}")
    assert d[-1] == 0.0, f"{name}: L=100 must be the zero reference"
    assert len(d) == len(sig) == len(L)

flat = [v for _, d, _ in ROWS for v in d[:-1]]
assert len(flat) == 16, "Table 6 reports sixteen differences"
big = [v for v in flat if abs(v) > 0.014]
print(f"\n{len(big)} of {len(flat)} differences exceed 0.014 in |dR2|: "
      + ", ".join(f"{v:+.4f}" for v in sorted(big)))
print(f"tunnel L=0 is {abs(ROWS[0][1][0]) / abs(ROWS[1][1][0]):.1f}x the next largest effect")
print("only the tunnel is monotonic in L; the other three are not.")
