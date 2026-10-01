#!/usr/bin/env python3
"""Figure for Table 5 (component ablation) — a paired-difference forest plot.

The table's message is a NULL result: no component removal changes R2 significantly on
any system. A grouped bar chart would hide that; a forest plot with confidence intervals
against a zero reference line shows it directly — every interval crosses zero.

Two intervals are drawn per point:
  thick  = +/- 1 s.d. over the five seeds (the spread quoted in Table 5)
  thin   = 95 % CI of the paired mean, t(4) = 2.776, i.e. 1.24 x s.d.
The CI is the one that answers "is this different from zero", so it is the wider mark.

    python3 ablation_table5.py     ->  ablation_table5.pdf / .png
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

# ---------------------------------------------------------------- Table 5, verbatim
N_SEEDS = 5
SYSTEMS = ["SDOF oscillator", "3-storey steel frame",
           "Wind turbine", "Buried tunnel"]
BASELINE = [0.9123, 0.8363, 0.8631, 0.8704]          # full framework, R2
COMPONENTS = [                                        # label, mean, s.d. per system
    ("removing FiLM",
     [+0.0039, -0.0084, +0.0270, -0.0254],
     [0.0553, 0.0203, 0.0332, 0.0291]),
    ("removing Load Skip",
     [+0.0143, -0.0160, +0.0054, +0.0018],
     [0.0229, 0.0130, 0.0237, 0.0143]),
    (r"Kaiming init $\rightarrow$ zero",
     [-0.0233, +0.0117, -0.0159, -0.0118],
     [0.0286, 0.0280, 0.0205, 0.0155]),
]

# palette shared with the TikZ system figures (benchmark_systems4.tex)
INK, STEEL, TEAL, AMBER = "#1F2A37", "#5B6B7C", "#0D7D8A", "#A85C11"
GREYBG = "#EEF1F4"
COL = [TEAL, AMBER, STEEL]
MRK = ["o", "s", "^"]

TCRIT = stats.t.ppf(0.975, N_SEEDS - 1)               # 2.7764 for n = 5


def pval(mean: float, sd: float) -> float:
    """Two-sided paired t-test against zero, from the reported mean and s.d."""
    return float(2 * stats.t.sf(abs(mean) / (sd / np.sqrt(N_SEEDS)), N_SEEDS - 1))


# ---------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": STEEL, "axes.labelcolor": INK,
    "xtick.color": STEEL, "ytick.color": STEEL,
    "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig = plt.figure(figsize=(7.09, 2.85))                 # 180 mm, Elsevier double column
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.25], wspace=0.06,
                      left=0.175, right=0.965, top=0.80, bottom=0.215)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

ypos = np.arange(len(SYSTEMS))[::-1]                   # first table row at the top
DODGE = 0.24

# ---------------------------------------------------------------- (a) baseline R2
for y in ypos[1::2]:
    axL.axhspan(y - 0.5, y + 0.5, color=GREYBG, lw=0, zorder=0)
for y, r2 in zip(ypos, BASELINE):
    axL.plot([0.80, r2], [y, y], color=STEEL, lw=0.7, alpha=0.45, zorder=1)
    axL.plot(r2, y, "o", ms=5.2, color=INK, zorder=3)
    axL.annotate(f"{r2:.3f}", (r2, y), xytext=(0, 7), textcoords="offset points",
                 ha="center", fontsize=7.5, color=INK)

axL.set_ylim(-0.62, len(SYSTEMS) - 0.38)
axL.set_xlim(0.795, 0.945)
axL.set_xticks([0.80, 0.85, 0.90])
axL.set_xlabel(r"$R^2$, full framework")
axL.set_yticks(ypos)
axL.set_yticklabels(SYSTEMS)
axL.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    axL.spines[s].set_visible(False)

# ---------------------------------------------------------------- (b) forest plot
axR.axvline(0.0, color=INK, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
for y in ypos[1::2]:                                   # alternating row shading
    axR.axhspan(y - 0.5, y + 0.5, color=GREYBG, lw=0, zorder=0)

smallest_p, smallest_xy = 1.0, None
for k, (label, means, sds) in enumerate(COMPONENTS):
    off = DODGE * (1 - k)                              # +0.24 / 0 / -0.24
    for y, m, sd in zip(ypos, means, sds):
        ci = TCRIT * sd / np.sqrt(N_SEEDS)
        yy = y + off
        axR.plot([m - ci, m + ci], [yy, yy], color=COL[k], lw=0.9, zorder=3,
                 solid_capstyle="butt")
        for e in (m - ci, m + ci):                     # CI caps
            axR.plot([e, e], [yy - 0.075, yy + 0.075], color=COL[k], lw=0.9, zorder=3)
        axR.plot([m - sd, m + sd], [yy, yy], color=COL[k], lw=2.6, zorder=4,
                 alpha=0.45, solid_capstyle="butt")
        axR.plot(m, yy, MRK[k], ms=4.4, color=COL[k], mec="white", mew=0.6, zorder=5)
        p = pval(m, sd)
        if p < smallest_p:
            smallest_p, smallest_xy = p, (m + ci, yy)

axR.annotate(f"$p$ = {smallest_p:.3f}", smallest_xy, xytext=(6, -0.5),
             textcoords="offset points", va="center", fontsize=7, color=AMBER)

lim = 0.077
axR.set_xlim(-lim, lim)
axR.set_ylim(-0.62, len(SYSTEMS) - 0.38)
axR.set_yticks(ypos)
axR.set_yticklabels([])
axR.tick_params(axis="y", length=0)
axR.set_xticks(np.arange(-0.06, 0.061, 0.02))
axR.set_xlabel(r"Paired change in $R^2$ against the same-seed baseline, $\Delta R^2$",
                labelpad=27)
for s in ("top", "right", "left"):
    axR.spines[s].set_visible(False)

# direction of the effect, stated in the table's own terms
arrow = dict(arrowstyle="-|>,head_width=0.13,head_length=0.30", color=STEEL, lw=0.6)
for xa, xb, txt in [(0.10, 0.34, "variant is worse"), (0.90, 0.66, "variant is better")]:
    axR.annotate("", xy=(xa, -0.245), xytext=(xb, -0.245), xycoords="axes fraction",
                 arrowprops=arrow, annotation_clip=False)
    axR.text((xa + xb) / 2, -0.225, txt, transform=axR.transAxes, ha="center",
             va="bottom", fontsize=7, color=STEEL)

handles = [Line2D([], [], color=c, marker=m, ms=4.4, mec="white", mew=0.6, lw=1.4,
                  label=lab) for (lab, _, _), c, m in zip(COMPONENTS, COL, MRK)]
axR.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.20),
           ncol=3, frameon=False, handlelength=1.6, columnspacing=1.8,
           handletextpad=0.5)

axL.text(-0.62, 1.16, "(a)", transform=axL.transAxes, fontsize=9,
         fontweight="bold", color=INK, va="top")
axR.text(-0.015, 1.16, "(b)", transform=axR.transAxes, fontsize=9,
         fontweight="bold", color=INK, va="top")

for ext in ("pdf", "png"):
    fig.savefig(f"ablation_table5.{ext}", dpi=600, bbox_inches="tight",
                facecolor="white")

# ---------------------------------------------------------------- self-check
print(f"t_crit(0.975, {N_SEEDS - 1}) = {TCRIT:.4f}   CI half-width = {TCRIT / np.sqrt(N_SEEDS):.4f} x s.d.\n")
print(f"{'component':<28}{'system':<28}{'mean':>9}{'95% CI':>20}{'p':>9}  crosses 0")
n_cross = 0
for label, means, sds in COMPONENTS:
    for sysname, m, sd in zip(SYSTEMS, means, sds):
        ci = TCRIT * sd / np.sqrt(N_SEEDS)
        crosses = (m - ci) < 0 < (m + ci)
        n_cross += crosses
        plain = label.replace("$-$", "-").replace(r"$\rightarrow$", "->")
        print(f"{plain:<28}{sysname:<28}{m:+9.4f}"
              f"  [{m - ci:+.4f}, {m + ci:+.4f}]{pval(m, sd):9.3f}  {crosses}")
assert n_cross == 12, f"only {n_cross}/12 intervals cross zero — the null claim is wrong"
print(f"\nall {n_cross}/12 confidence intervals cross zero; smallest p = {smallest_p:.4f}")
