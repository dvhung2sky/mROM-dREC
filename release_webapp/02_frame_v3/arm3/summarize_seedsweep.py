"""Aggregate seed_sweep/frame3_seed*_arm3.json into a table + CSV.

Reports Arm 1 and Arm 3 per seed, plus the split-invariant `resid_energy_removed`
(fraction of the Arm-1 residual energy the corrector removes).  Arm 1 varies across
seeds purely because the test split changes -- it involves no model -- so comparing
its spread against Arm 3's separates "favourable split" from "better corrector".

Writes seed_sweep/frame3_seed_sweep.csv (new file; touches nothing pre-existing).
"""
import csv
import glob
import json
import os

import numpy as np

rows = []
for p in sorted(glob.glob("seed_sweep/frame3_seed*_arm3.json")):
    r = json.load(open(p))
    if r.get("epochs", 0) < 5:          # skip smoke tests
        continue
    rows.append(r)

if not rows:
    raise SystemExit("no completed seed runs found in seed_sweep/")

rows.sort(key=lambda r: r["seed"])
cols = ["seed", "arm1_R2", "R2", "resid_energy_removed", "alpha_star", "R2_alpha",
        "peak_err_pct", "arm1_peak_err_pct", "RMSE_mm", "arm1_RMSE_mm",
        "R2_persample", "R2_floor1", "R2_floor2", "R2_floor3", "n_test", "epochs"]

hdr = (f"{'seed':>5} {'Arm1':>8} {'Arm3':>8} {'Arm3-Arm1':>10} {'resid rm':>9} "
       f"{'alpha':>6} {'Arm3+a':>8} {'peak%':>7} {'RMSE mm':>8}")
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['seed']:>5} {r['arm1_R2']:>+8.4f} {r['R2']:>+8.4f} "
          f"{r['R2']-r['arm1_R2']:>+10.4f} {r['resid_energy_removed']:>9.4f} "
          f"{r['alpha_star']:>6.2f} {r['R2_alpha']:>+8.4f} "
          f"{r['peak_err_pct']:>7.2f} {r['RMSE_mm']:>8.2f}")

print("-" * len(hdr))
def stat(key):
    v = np.array([r[key] for r in rows], float)
    return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, v.min(), v.max()

print(f"\nn = {len(rows)} seeds     (mean +/- sd [min, max])")
for key, label in [("arm1_R2", "Arm 1  R2"), ("R2", "Arm 3  R2"),
                   ("R2_alpha", "Arm 3  R2 (alpha)"),
                   ("resid_energy_removed", "resid energy removed"),
                   ("peak_err_pct", "Arm 3  peak err %"),
                   ("RMSE_mm", "Arm 3  RMSE mm")]:
    m, s, lo, hi = stat(key)
    print(f"  {label:<22} {m:+.4f} +/- {s:.4f}   [{lo:+.4f}, {hi:+.4f}]")

d = np.array([r["R2"] - r["arm1_R2"] for r in rows], float)
print(f"  {'Arm 3 - Arm 1':<22} {d.mean():+.4f} +/- {d.std(ddof=1):.4f}   "
      f"[{d.min():+.4f}, {d.max():+.4f}]")
print(f"\n  Arm 3 beat Arm 1 on {(d > 0).sum()}/{len(d)} seeds")
a3 = np.array([r["R2"] for r in rows], float)
print(f"  Arm 3 cleared 0.80 on {(a3 > 0.80).sum()}/{len(a3)} seeds; "
      f"cleared 0.8671 on {(a3 >= 0.8671).sum()}/{len(a3)}")

out = "seed_sweep/frame3_seed_sweep.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"\nwrote {out}")
